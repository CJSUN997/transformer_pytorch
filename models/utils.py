"""
@author: Sanchos
@create: 2024-08-14
@desc: this file has base module needed in blocks.py
"""
import torch
import math
from torch import nn 

class Attention(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Attention, self).__init__(*args, **kwargs)

    def forward(self, Q, K ,V):
        return Q


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        d_k = k.shape[-1]
        x = q @ k.transponse(2, 3) / math.sqrt(d_k)

        if mask:
            x = x.masked_fill(mask == 0, -10000)
        return self.softmax(x) @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h_head) -> None:
        super(MultiHeadAttention, self).__init__()
        self.h_head = h_head
        self.attention = ScaledDotProductAttention()
        # the dimension of wq wk is (d_model, d_k = d_model / h_head)
        # and the multi head attention is stacked with {h_head} attention layer
        # so the final dim is (d_model, d_k * h_head = d_model)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)


    def forward(self, Q, K, V, attn_mask=None):
        
        # Linear transform
        q, k, v = self.w_q(Q), self.w_k(K), self.w_v(V)
        q, k, v = self.split(q), self.split(k), self.split(v)
        
        out = self.attention(q, k, v, mask=attn_mask)
        out = self.concat(out)

        return self.w_o(out)
    
    def split(self,x):
        bs, length, d_model = x.shape
        d_tensor = d_model // self.h_head

        return x.view(bs, length, self.h_head, d_tensor).transpose(1,2)
    
    def concat(self, x):
        bs, h, length, d_tensor = x.shape

        d_model = h * d_tensor
        return x.transpose(1, 2).contiguous().view(bs, length, d_model)

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
        
    