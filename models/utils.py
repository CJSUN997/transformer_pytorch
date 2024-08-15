"""
@author: Sanchos
@create: 2024-08-14
@desc: this file has base module needed in blocks.py
"""
import torch
from torch import nn 

class Attention(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Attention, self).__init__(*args, **kwargs)

    def forward(self, Q, K ,V):
        return Q


class MultiHeadAttention(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(MultiHeadAttention, self).__init__(*args, **kwargs)

    def forward(self, x):
        return x



class PositionWiseFeedForward(nn.Module):

    def __init__(self, d_model = 512, d_ff = 2048, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.linear2(self.dropout(self.act(self.linear1(x))))
    
class LayerNorm(nn.Module):

    def __init__(self, d_model=512, eps=1e-5):
        super(LayerNorm, self).__init__()
        # 初始化可学习参数 gamma 和beta
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps=eps
        
    def forward(self, x):
        # 计算均值和方差
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # 归一化
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        # 应用gamma和beta
        return self.gamma * x_norm + self.beta
        
    