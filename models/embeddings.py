"""
@author: Sanchos
@create: 2024-08-15
@desc: this file has base embeddings needed in models.py
"""

import torch
from torch.nn import nn

class PositionalEncoding(nn.Module):
    pass

class TokenEmbedding(nn.Embedding):
    pass


# construct transformerembedding with PositionalEncoding and TokenEmbedding
class TransformerEmbedding(nn.Module):

