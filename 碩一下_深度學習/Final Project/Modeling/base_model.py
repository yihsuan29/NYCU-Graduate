import numpy as np
import pandas as pd
import copy

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torch.utils.data import Sampler
import torch
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import root_mean_squared_error,mean_absolute_error

class StockWindowDataset4D(Dataset):
    def __init__(self, df, feature_columns, label_column='label', window_size=10):
        self.df = df.copy()
        self.feature_columns = feature_columns
        self.label_column = label_column
        self.window_size = window_size

        # 取得所有日期排序（這邊只選擇有完整window的日期）
        all_dates = sorted(self.df['date'].unique())
        self.dates = all_dates[window_size:]  # 只從第 window_size 天開始算起，因為要往前8天的視窗

        # 取得所有股票ID
        self.stocks = sorted(self.df['PERMNO'].unique())
        self.N = len(self.stocks)
        self.F = len(feature_columns)

        # 將資料整理成 dict {(date, permno): feature_values}
        # 方便快速取用
        self.data_dict = {}
        for _, row in self.df.iterrows():
            key = (row['date'], row['PERMNO'])
            self.data_dict[key] = row[feature_columns].values.astype(np.float64)

        # label通常是當天或隔天的值，這裡以當天為例
        self.label_dict = {}
        for _, row in self.df.iterrows():
            key = (row['date'], row['PERMNO'])
            self.label_dict[key] = row[label_column]

    def __len__(self):
        return len(self.dates)  # 可做多少天的樣本

    def __getitem__(self, idx):
        date = self.dates[idx]

        # 先取得當天日期的 index
        date_idx = self.df['date'].unique().tolist().index(date)

        # window的日期範圍 (往前推 window_size 天)
        window_dates = self.df['date'].unique()[date_idx - self.window_size: date_idx]

        # 準備 output tensor shape: [N, W, F]
        # W維度是window size，N是股票數量，F是feature數量
        x = np.zeros((self.N, self.window_size, self.F), dtype=np.float64)
        y = np.zeros((self.N,), dtype=np.float64)  # label對應每支股票

        for i, stock in enumerate(self.stocks):
            for j, w_date in enumerate(window_dates):
                key = (w_date, stock)
                if key in self.data_dict:
                    x[i, j, :] = self.data_dict[key]
                else:
                    # 缺失資料補0
                    x[i, j, :] = 0.0

            # 當天label

        # 這邊輸出維度是 [N, W, F]
        # 如果想要最終維度是 [D, N, W, F]，那要用 DataLoader batch
        # 一個batch裡面放多個日期 (D個) 就會變成 (D, N, W, F)
        return torch.tensor(x)
    
    
def calc_ic(pred, label):
    df = pd.DataFrame({'pred':pred, 'label':label})
    ic = df['pred'].corr(df['label'])
    ric = df['pred'].corr(df['label'], method='spearman')
    return ic, ric

def zscore(x):
    return (x - x.mean()).div(x.std())

def drop_extreme(x):
    sorted_tensor, indices = x.sort()
    N = x.shape[0]
    percent_2_5 = int(0.025*N)  
    # Exclude top 2.5% and bottom 2.5% values
    filtered_indices = indices[percent_2_5:-percent_2_5]
    mask = torch.zeros_like(x, device=x.device, dtype=torch.bool)
    mask[filtered_indices] = True
    return mask, x[mask]

def drop_na(x):
    N = x.shape[0]
    mask = ~x.isnan()
    return mask, x[mask]

class DailyBatchSamplerRandom(Sampler):
    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        # calculate number of samples in each batch
        self.daily_count = self.data_source.groupby("date").size().values
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)  # calculate begin index of each batch
        self.daily_index[0] = 0

    def __iter__(self):
        if self.shuffle:
            index = np.arange(len(self.daily_count))
            np.random.shuffle(index)
            for i in index:
                yield np.arange(self.daily_index[i], self.daily_index[i] + self.daily_count[i])
        else:
            for idx, count in zip(self.daily_index, self.daily_count):
                yield np.arange(idx, idx + count)

    def __len__(self):
        return len(self.data_source)


class SequenceModel():
    def __init__(self, n_epochs, lr, GPU=None, seed=None, train_stop_loss_thred=None, save_path = 'model/', save_prefix= ''):
        self.n_epochs = n_epochs
        self.lr = lr
        self.device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.train_stop_loss_thred = train_stop_loss_thred

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
        self.fitted = -1

        self.model = None
        self.train_optimizer = None

        self.save_path = save_path
        self.save_prefix = save_prefix


    def init_model(self):
        if self.model is None:
            raise ValueError("model has not been initialized")

        self.train_optimizer = optim.Adam(self.model.parameters(), self.lr)
        self.model.to(self.device)

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)
        loss = (pred[mask]-label[mask])**2
        return torch.mean(loss)

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        losses = []
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
        for data in progress_bar:
            data = torch.squeeze(data, dim=0)
            '''
            data.shape: (N, T, F)
            N - number of stocks
            T - length of lookback_window, 8
            F - 158 factors + 63 market information + 1 label           
            '''
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            
            # Additional process on labels
            # If you use original data to train, you won't need the following lines because we already drop extreme when we dumped the data.
            # If you use the opensource data to train, use the following lines to drop extreme labels.
            #########################
            # mask, label = drop_extreme(label)
            # feature = feature[mask, :, :]
            # label = zscore(label) # CSZscoreNorm
            #########################

            pred = self.model(feature.float())
            loss = self.loss_fn(pred, label)
            losses.append(loss.item())

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.train_optimizer.step()

        return float(np.mean(losses))

    def test_epoch(self, data_loader):
        self.model.eval()
        losses = []

        for data in data_loader:
            data = torch.squeeze(data, dim=0)
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            # You cannot drop extreme labels for test. 
            # label = zscore(label)
                        
            pred = self.model(feature.float())
            loss = self.loss_fn(pred, label)
            losses.append(loss.item())

        return float(np.mean(losses))

    def _init_data_loader(self, data, feature_columns, shuffle=True, drop_last=True):
        # sampler = DailyBatchSamplerRandom(data, shuffle)
        # dataset = StockWindowDataset4D(data, feature_columns)
        dataset = StockWindowDataset4D(data, feature_columns)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
        return data_loader

    def load_param(self, param_path):
        self.model.load_state_dict(torch.load(param_path, map_location=self.device))
        self.fitted = 'Previously trained.'

    def fit(self, dl_train, dl_valid=None):
        print("training started.")
        # feature_columns = dl_train.get_feature_columns().drop(["PERMNO","TICKER","COMNAM", "date", "SICCD", "NCUSIP","CUSIP"])
        columns_to_remove = ["PERMNO", "date"]
        feature_columns = [col for col in dl_train.columns if col not in columns_to_remove]
        train_loader = self._init_data_loader(dl_train, feature_columns,shuffle=True, drop_last=True)
        best_param = None
        for step in range(self.n_epochs):
            train_loss = self.train_epoch(train_loader, step)
            self.fitted = step
            if dl_valid:
                predictions, metrics = self.predict(dl_valid)
                print("Epoch %d, train_loss %.6f, valid ic %.4f, icir %.3f, rankic %.4f, rankicir %.3f." % (step, train_loss, metrics['IC'],  metrics['ICIR'],  metrics['RIC'],  metrics['RICIR']))
            else: print("Epoch %d, train_loss %.6f" % (step, train_loss))
        

        best_param = copy.deepcopy(self.model.state_dict())
        torch.save(best_param, f'{self.save_path}/{self.save_prefix}_{self.seed}.pkl')
        

        

    def predict(self, dl_test):
        dl_test_filtered = dl_test.groupby('PERMNO').apply(lambda x: x.iloc[10:]).reset_index(drop=True)
        columns_to_remove = ["PERMNO", "date"]
        feature_columns = [col for col in dl_test.columns if col not in columns_to_remove]
        if self.fitted<0:
            raise ValueError("model is not fitted yet!")
        else:
            print('Epoch:', self.fitted)

        test_loader = self._init_data_loader(dl_test, feature_columns, shuffle=False, drop_last=False)

        preds = []
        labels = []
        ic = []
        ric = []

        self.model.eval()
        for data in test_loader:
            data = torch.squeeze(data, dim=0)
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1]
            
            # nan label will be automatically ignored when compute metrics.
            # zscorenorm will not affect the results of ranking-based metrics.

            with torch.no_grad():
                pred = self.model(feature.float()).detach().cpu().numpy()
            preds.append(pred.ravel())
            labels.append(label.detach().numpy())

            daily_ic, daily_ric = calc_ic(pred, label.detach().numpy())
            ic.append(daily_ic)
            ric.append(daily_ric)


        predictions = np.concatenate(preds)
        targets = np.concatenate(labels)
        rmse = root_mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        
        predictions = pd.Series(np.concatenate(preds), index=dl_test_filtered.index)

        metrics = {
            'IC': np.mean(ic),
            'ICIR': np.mean(ic)/np.std(ic),
            'RIC': np.mean(ric),
            'RICIR': np.mean(ric)/np.std(ric),
            'RMSE':np.mean(rmse),
            'MAE':np.mean(mae)
        }

        return predictions, metrics
