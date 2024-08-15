"""
@author: Sanchos
@create: 2024-08-14
"""
from torch import nn

from utils import MultiHeadAttention, PositionWiseFeedForward, LayerNorm

# an encoder layer has two sub-layer connect with  residual connection and LayerNorm
class EncoderLayer(nn.Module):

    def __init__(self, d_model=512, d_ffn=2048, h_head=8, drop_prob=0.1 ):
        super().__init__()
        self.attention = MultiHeadAttention(d_model=d_model,h_head=h_head)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.norm1 = LayerNorm(d_model=d_model)
        self.ffn = PositionWiseFeedForward(d_model=d_model, d_ff=d_ffn, dropout=drop_prob)
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)

    def forward(self, x):
        # sub-layer1
        x = x + self.dropout1(self.attention(q=x, k=x, v=x))
        x = self.norm1(x)
        
        # sub-layer2
        x = x + self.dropout2(self.ffn(x))
        return self.norm2(x)
    

class DecoderLayer(nn.Module):
    def __init__(slef):
        return 
    
    def forward(self,x):
        return x
   




