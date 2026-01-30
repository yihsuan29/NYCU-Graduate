'''
Code from MASTER "https://github.com/SJTU-DMTai/MASTER/blob/master"
'''

import torch
from torch import nn
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
import math
import torch.nn.functional as F


from base_model import SequenceModel


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.shape[1], :]


class SAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.temperature = math.sqrt(self.d_model/nhead)

        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        attn_dropout_layer = []
        for i in range(nhead):
            attn_dropout_layer.append(Dropout(p=dropout))
        self.attn_dropout = nn.ModuleList(attn_dropout_layer)

        # input LayerNorm
        self.norm1 = LayerNorm(d_model, eps=1e-5)

        # FFN layerNorm
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.norm1(x)
        q = self.qtrans(x).transpose(0,1)
        k = self.ktrans(x).transpose(0,1)
        v = self.vtrans(x).transpose(0,1)

        dim = int(self.d_model/self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i==self.nhead-1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]

            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)) / self.temperature, dim=-1)
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh).transpose(0, 1))
        att_output = torch.concat(att_output, dim=-1)

        # FFN
        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

        return att_output


class TAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = []
        if dropout > 0:
            for i in range(nhead):
                self.attn_dropout.append(Dropout(p=dropout))
            self.attn_dropout = nn.ModuleList(self.attn_dropout)

        # input LayerNorm
        self.norm1 = LayerNorm(d_model, eps=1e-5)
        # FFN layerNorm
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        # FFN
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.norm1(x)
        q = self.qtrans(x)
        k = self.ktrans(x)
        v = self.vtrans(x)

        dim = int(self.d_model / self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i==self.nhead-1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]
            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)), dim=-1)
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh))
        att_output = torch.concat(att_output, dim=-1)

        # FFN
        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

        return att_output


class Gate(nn.Module):
    def __init__(self, d_input, d_output,  beta=1.0):
        super().__init__()
        self.trans = nn.Linear(d_input, d_output)
        self.d_output =d_output
        self.t = beta

    def forward(self, gate_input):
        output = self.trans(gate_input)
        output = torch.softmax(output/self.t, dim=-1)
        return self.d_output*output
    
    
### Ind Gate ##############################################################################
class IndustryGate(nn.Module):
    def __init__(self, n_industries, d_input, d_output, beta=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_industries, d_output, d_input)) 
        self.bias = nn.Parameter(torch.zeros(n_industries, d_output))           
        self.beta = beta
        self.d_output = d_output

    def forward(self, ind_input, ind_ids):
        B = ind_input.size(0)
        W = self.weight[ind_ids]  
        b = self.bias[ind_ids] 
        
        ind_input = ind_input.unsqueeze(-1)              
        logits = torch.bmm(W, ind_input).squeeze(-1) + b 
        alpha = F.softmax(logits / self.beta, dim=-1)    
        return self.d_output * alpha

###########################################################################################


class TemporalAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z):
        h = self.trans(z) # [N, T, D]
        query = h[:, -1, :].unsqueeze(-1)
        lam = torch.matmul(h, query).squeeze(-1)  # [N, T, D] --> [N, T]
        lam = torch.softmax(lam, dim=1).unsqueeze(1)
        output = torch.matmul(lam, z).squeeze(1)  # [N, 1, T], [N, T, D] --> [N, 1, D]
        return output


class MASTER(nn.Module):
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate,
                 stock_start_index, stock_end_index,beta,
                    gate_input_start_index, gate_input_end_index, 
                    ind_index_column, ind_gate_start_index,ind_gate_end_index):
        super(MASTER, self).__init__()
        # market
        self.stock_start_index = stock_start_index
        self.stock_end_index = stock_end_index
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index+1) # F'
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)
        
        ### Ind Gate ##############################################################################
        self.ind_index_column = ind_index_column
        self.ind_gate_start_index = ind_gate_start_index
        self.ind_gate_end_index = ind_gate_end_index     
        self.d_industry_input = (ind_gate_end_index - ind_gate_start_index +1)
        self.industry_gate = IndustryGate(
            n_industries=12,
            d_input=self.d_industry_input,
            d_output=d_feat,
            beta=beta*3
        )
        ###########################################################################################

        self.layers = nn.Sequential(
            # feature layer
            nn.Linear(d_feat, d_model),
            PositionalEncoding(d_model),
            # intra-stock aggregation
            TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate),
            # inter-stock aggregation
            SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate),
            TemporalAttention(d_model=d_model),
            # decoder
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        src = x[:, :, self.stock_start_index:self.stock_end_index +1] # N, T, D
        gate_input = x[:, -1, self.gate_input_start_index:self.gate_input_end_index+1]
        alpha_market = self.feature_gate(gate_input)
        
        ### Ind Gate ##############################################################################
        # ind_ids = x[:, -1, self.ind_index_column].long() 
        # ind_input = x[:, -1, self.ind_gate_start_index: self.ind_gate_end_index+1] 
        # alpha_industry = self.industry_gate(ind_input, ind_ids) 

        ###########################################################################################    
        alpha = alpha_market  #* alpha_industry
        # alpha = F.softmax(alpha, dim=-1)
        src = src * torch.unsqueeze(alpha, dim=1)
       
        output = self.layers(src).squeeze(-1)

        return output


class MASTERModel(SequenceModel):
    def __init__(
            self, d_feat, d_model, t_nhead, s_nhead, stock_start_index, stock_end_index,
            gate_input_end_index, gate_input_start_index,
            ind_index_column, ind_gate_start_index, ind_gate_end_index,
            T_dropout_rate, S_dropout_rate, beta, **kwargs,
    ):
        super(MASTERModel, self).__init__(**kwargs)
        self.d_model = d_model
        self.d_feat = d_feat
        self.stock_start_index = stock_start_index
        self.stock_end_index = stock_end_index
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        
        ### Ind Gate ##############################################################################
        self.ind_index_column = ind_index_column
        self.ind_gate_start_index = ind_gate_start_index
        self.ind_gate_end_index = ind_gate_end_index      
        ###########################################################################################

        self.T_dropout_rate = T_dropout_rate
        self.S_dropout_rate = S_dropout_rate
        self.t_nhead = t_nhead
        self.s_nhead = s_nhead
        self.beta = beta

        self.init_model()

    def init_model(self):
        self.model = MASTER(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                                   T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,beta=self.beta,
                                   stock_start_index = self.stock_start_index, stock_end_index = self.stock_end_index,
                                   gate_input_start_index=self.gate_input_start_index, gate_input_end_index=self.gate_input_end_index, 
                                   ind_index_column=self.ind_index_column, ind_gate_start_index =self.ind_gate_start_index,
                                   ind_gate_end_index= self.ind_gate_end_index)
        super(MASTERModel, self).init_model()