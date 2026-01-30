import pandas as pd
import numpy as np




def data_preprocessing(stock_path, market_path, industry_path=None, mapping_path = None, news_path=None):
    
    df_stock = pd.read_parquet(stock_path)
    df_market = pd.read_parquet(market_path)
    df_industry = pd.read_parquet(industry_path) if industry_path else None
    df_mapping = pd.read_csv(mapping_path) if mapping_path else None

    df_stock["date"] = pd.to_datetime(df_stock["date"])
    df_market["date"] = pd.to_datetime(df_market["date"])
    df_industry["date"] = pd.to_datetime(df_industry["date"])
    df_mapping.rename(columns={'Fama': 'fama12'}, inplace=True)
    # feature_num = df_stock.shape[1] - 7  # 計算feature數量以及市場開始的index
    # end_index = feature_num + df_market.shape[1] - 1  #市場結束的index

    
    
    df_merged = pd.merge(df_stock, df_market, on="date", how="left") #合併股票與市場資訊
    df_merged = df_merged.sort_values(by=['PERMNO', 'date']).reset_index(drop=True)
    df_merged = pd.merge(df_merged, df_mapping, on=['PERMNO'], how='left')
    df_merged = pd.merge(df_merged, df_industry, on=['date', 'fama12'], how='left')
    
    df_merged = df_merged[df_merged['date'] >= '2008-01-01'] #撇除2008年之前的資料
    df_merged['label'] = df_merged.groupby('PERMNO')['RET'].shift(-1) #新增label(明天的RET)
    df_merged = df_merged.dropna(subset=['label']).reset_index(drop=True)

    bool_cols = df_merged.select_dtypes(include=['bool']).columns #選擇bool類型的欄位
    df_merged[bool_cols] = df_merged[bool_cols].astype(np.int8) #將bool類型轉換為int8

    df_merged['alpha084'].replace([np.inf, -np.inf], np.nan)
    q99 = df_merged['alpha084'].quantile(0.99)
    df_merged['alpha084'] = df_merged['alpha084'].clip(upper=q99)
    df_merged['alpha084_log'] = np.log(df_merged['alpha084'] + 1e-6)
    cols = list(df_merged.columns)
    idx_084 = cols.index('alpha084')
    cols.remove('alpha084_log')  
    cols.insert(idx_084, 'alpha084_log')  
    cols.remove('alpha084')
    df_merged = df_merged[cols]
    
    df_merged = df_merged.fillna(0) #將NaN值填充為0

    columns_to_remove = ["TICKER", "COMNAM", "SICCD", "NCUSIP", "CUSIP"]
    df_merged = df_merged.drop(columns_to_remove, axis=1)
    # df_merged.to_csv('data/View.csv', index=False)
    
    # i=0
    # for col in df_merged.columns:
    #     print(f"col{i}:{col}")
    #     i+=1
        
    stock_start = df_merged.columns.get_loc('alpha001')
    stock_end = df_merged.columns.get_loc('alpha101')#('alpha101')('total_news_count')
    gate_start = df_merged.columns.get_loc('Return')
    gate_end = df_merged.columns.get_loc('MCQ')#('MCQ') #('StdTurnover_60')
    ind_index = df_merged.columns.get_loc('fama12')
    ind_start = df_merged.columns.get_loc('ind_return')
    ind_end = df_merged.columns.get_loc('Ind_StdReturn_60')
    # print(f"{stock_start, stock_end, gate_start,gate_end, ind_index,ind_start,ind_end}")

    df_train = df_merged[df_merged['date'] < '2022-01-01']
    df_test = df_merged[(df_merged['date'] >= '2022-01-01')]
    #df_test = df_merged[(df_merged['date'] >= '2018-01-01')& (df_merged['date'] <= '2018-12-31')]

    return df_train, df_test,stock_start, stock_end, gate_start,gate_end, ind_index,ind_start,ind_end


# print(df_merged['alpha084'].describe())
# print(df_merged['alpha084'].isna().any().any(), np.isinf(df_merged['alpha084'].to_numpy()).any())
# print(df_merged.isna().any().any(), np.isinf(df_merged.to_numpy()).any())
# df_merged.to_csv('data/View.csv', index=False)