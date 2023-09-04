import torch
import torch.nn as nn
import numpy as np


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.arange(0, d_model, 2).float()
        div_term = div_term * (-np.log(10000.0) / d_model)
        div_term = torch.exp(div_term)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # T, 1, D
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0]]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(LearnedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(max_len, 1, d_model))

    def forward(self, x):
        x = x + self.pe[:x.shape[0]]
        return self.dropout(x)
