"""
@author: Sanchos
@create: 2024-08-15
@desc: this file has base module needed in blocks.py
"""

import torch
from torch import nn
from .blocks import *
from .embeddings import *

class Encoder(nn.Module):
    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob):
        super().__init__()
        self.emb = TransformerEmbedding(d_model, max_len, enc_voc_size, drop_prob)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model=d_model, d_ffn=ffn_hidden, h_head=n_head, drop_prob=drop_prob)
            for _ in range(n_layers)
        ])

    def forward(self, x, src_mask):
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x)
        
        return x
        
        

class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob):
        super(Decoder, self).__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        max_len=max_len,
                                        vocab_size=dec_voc_size,
                                        drop_prob=drop_prob)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model=d_model, 
                         h_head=n_head, 
                         d_ffn=ffn_hidden, 
                         drop_prob=drop_prob)
            for _ in range(n_layers)
        ])
        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, tgt, enc, tgt_mask, src_mask):
        tgt = self.emb(tgt)
        
        for layer in self.layers:
            tgt = layer(tgt, enc, tgt_mask, src_mask)

        return self.linear(tgt)


class Transformer(nn.Module):
    def __init__(self, src_pad_idx, tgt_pad_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob):
        super(Transformer, self).__init__()

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.device = 'gpu'
        
        self.encoder = Encoder(enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob)
        self.decoder = Decoder(dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob)

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.mask_tgt_mask(tgt)

        enc_output = self.encoder(src, src_mask)
        output = self.decoder(tgt, enc_output, tgt_mask, src_mask)
        return output
    
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_tgt_mask(self, tgt):
        tgt_pad_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_len = tgt.shape[1]
        tgt_sub_mask = torch.tril(torch.ones(tgt_len, tgt_len)).type(torch.ByteTensor).to(self.device)
        tgt_mask = tgt_pad_mask & tgt_sub_mask

        return tgt_mask
        


