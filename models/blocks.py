"""
@author: Sanchos
@create: 2024-08-14
"""
from torch import nn

from .utils import MultiHeadAttention, PositionWiseFeedForward, LayerNorm

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

    def forward(self, x, src_mask=None):
        # sub-layer1
        x = x + self.dropout1(self.attention(Q=x, K=x, V=x, atten_mask = src_mask))
        x = self.norm1(x)
        
        # sub-layer2
        x = x + self.dropout2(self.ffn(x))
        return self.norm2(x)
    

class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, h_head=8, d_ffn=2048, drop_prob=0.1 ):
        super(DecoderLayer, self).__init__()

        self.self_attention=MultiHeadAttention(d_model=d_model, h_head=h_head)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.norm1 = LayerNorm(d_model=d_model)

        self.enc_attention=MultiHeadAttention(d_model=d_model, h_head=h_head)
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)

        self.ffn = PositionWiseFeedForward(d_model=d_model, d_ff=d_ffn, dropout=drop_prob)
        self.dropout3 = nn.Dropout(p=drop_prob)
        self.norm3 = LayerNorm(d_model=d_model)
    
    def forward(self, dec_x, enc_x, tgt_mask, src_mask):
        # sublayer1
        x = self.dropout1(self.self_attention(Q=dec_x, K=dec_x, V=dec_x, attn_mask=tgt_mask))
        x = self.norm1(dec_x + x)
        # sublayer2
        x = x + self.dropout2(self.enc_attention(Q=x, K=enc_x, V=enc_x, attn_mask=src_mask))
        x = self.norm2(x)
        # sublayer3
        x = self.norm3(x + self.dropout3(self.ffn(x)))
        
        return x
   




