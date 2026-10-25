"""Temporal encoders and single-stream sequence baselines (RNN / LSTM / GRU / Transformer)."""
from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: d // 2])
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)]


class TemporalEncoder(nn.Module):
    """Encode [B, L, F] -> [B, H].

    kind: 'rnn' | 'lstm' | 'gru' | 'transformer' | 'mlp' (flattened window).
    Recurrent encoders return the final hidden state of the last layer; the transformer
    returns the representation of the last time step concatenated with the mean over time.
    """

    def __init__(self, n_in: int, hidden: int, kind: str = "gru", layers: int = 1,
                 dropout: float = 0.1, heads: int = 2, lookback: int = 14):
        super().__init__()
        self.kind = kind
        self.out_dim = hidden
        rnn_drop = dropout if layers > 1 else 0.0
        if kind in ("rnn", "lstm", "gru"):
            cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[kind]
            self.net = cls(n_in, hidden, num_layers=layers, batch_first=True, dropout=rnn_drop)
        elif kind == "transformer":
            self.inp = nn.Linear(n_in, hidden)
            self.pos = PositionalEncoding(hidden)
            layer = nn.TransformerEncoderLayer(hidden, heads, dim_feedforward=2 * hidden, dropout=dropout,
                                               batch_first=True, norm_first=True)
            self.net = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.proj = nn.Linear(2 * hidden, hidden)
        elif kind == "mlp":
            self.net = nn.Sequential(nn.Flatten(), nn.Linear(n_in * lookback, hidden), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU())
        else:
            raise ValueError(kind)

    def forward(self, x):
        if self.kind in ("rnn", "gru"):
            _, h = self.net(x)
            return h[-1]
        if self.kind == "lstm":
            _, (h, _) = self.net(x)
            return h[-1]
        if self.kind == "transformer":
            z = self.net(self.pos(self.inp(x)))
            return self.proj(torch.cat([z[:, -1], z.mean(1)], -1))
        return self.net(x)


class SequenceClassifier(nn.Module):
    """Single-stream baseline: all sequence features in one encoder (+ static context)."""

    def __init__(self, n_in: int, n_ctx: int, hidden: int = 64, kind: str = "gru", layers: int = 1,
                 dropout: float = 0.2, heads: int = 2, lookback: int = 14, n_classes: int = 4):
        super().__init__()
        self.encoder = TemporalEncoder(n_in, hidden, kind, layers, dropout, heads, lookback)
        self.n_ctx = n_ctx
        self.head = nn.Sequential(nn.Linear(hidden + n_ctx, hidden), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, n_classes))

    def forward(self, x, ctx=None):
        h = self.encoder(x)
        if self.n_ctx:
            h = torch.cat([h, ctx], -1)
        return self.head(h), None
