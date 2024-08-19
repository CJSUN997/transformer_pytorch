"""
@author: Sanchos
@create: 2024-08-15
@desc: this file has base embeddings needed in models.py
"""

import torch
from torch import nn

class PositionalEncoding(nn.Module):
    '''
    positional encoding
    '''

    def __init__(self, d_model,max_len):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.encoding = torch.zeros((max_len, self.d_model))
        self.encoding.requires_grad = False # disable compute gradient
        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)
        index = torch.arange(0, self.d_model, step=2).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (index / self.d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (index / self.d_model)))


    def forward(self, x):
        bs, seq_len = x.shape
        return self.encoding[:seq_len, :] # torch shape [seq_len, d_model]



class TokenEmbedding(nn.Embedding):
    '''
    Token Embedding using torch.nn
    '''    
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)

# construct transformerembedding with PositionalEncoding and TokenEmbedding
class TransformerEmbedding(nn.Module):
    def __init__(self, d_model, max_len, vocab_size, drop_prob):
        super(TransformerEmbedding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len # max length of input

        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_embedding = PositionalEncoding(d_model, max_len)
        self.drop_out = nn.Dropout(p = drop_prob)

    def forward(self, x):
        token = self.token_embedding(x)
        pos = self.pos_embedding(x)
        return self.drop_out(pos + token)



