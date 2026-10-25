"""ClimateSense Multimodal Attention Fusion Network (MAF-Net).

    climate seq  -> TemporalEncoder A --\
    air seq      -> TemporalEncoder B ---+-> Linear proj + modality embedding -> tokens [B, M, d]
    dryness seq  -> TemporalEncoder C ---+        |
    context      -> MLP D --------------/         v
                          (optional) self-attention among modality tokens (masked)
                                                  v
                  learned query token  --cross-attention-->  fused vector  (attention weights over M)
                                                  v
                  gated residual with masked mean of tokens -> MLP head -> 4 logits

Modality masks: a modality whose inputs are missing for the whole window is excluded via the
attention key-padding mask (never replaced by fictitious numbers).  Modality dropout randomly
masks whole modalities during training.  `fusion='concat'` replaces attention with plain
concatenation of the projected tokens (ablation: "all modalities without attention").
"""
from __future__ import annotations

import torch
from torch import nn

from .sequence import TemporalEncoder


class MAFNet(nn.Module):
    def __init__(self, modality_idx: dict, missing_idx: dict, n_ctx: int = 0, d_model: int = 64,
                 hidden: int = 64, encoder: str = "gru", air_encoder: str = "gru", dry_encoder: str = "gru",
                 layers: int = 1, dropout: float = 0.2, heads: int = 2, lookback: int = 14,
                 fusion: str = "attention", token_self_attention: bool = True,
                 modality_dropout: float = 0.1, n_classes: int = 4):
        super().__init__()
        self.names = list(modality_idx)
        self.modality_idx = {k: torch.as_tensor(v, dtype=torch.long) for k, v in modality_idx.items()}
        self.missing_idx = missing_idx
        self.fusion = fusion
        self.modality_dropout = modality_dropout
        enc_kind = {"climate": encoder, "air": air_encoder, "dryness": dry_encoder}
        small = max(16, hidden // 2)
        self.encoders = nn.ModuleDict()
        self.proj = nn.ModuleDict()
        for m, idx in modality_idx.items():
            h = hidden if m == "climate" else small
            self.encoders[m] = TemporalEncoder(len(idx), h, enc_kind.get(m, "gru"), layers if m == "climate" else 1,
                                               dropout, heads, lookback)
            self.proj[m] = nn.Sequential(nn.Linear(h, d_model), nn.LayerNorm(d_model))
        self.use_ctx = n_ctx > 0
        if self.use_ctx:
            self.names = self.names + ["context"]
            self.ctx_enc = nn.Sequential(nn.Linear(n_ctx, small), nn.GELU(), nn.Linear(small, d_model),
                                         nn.LayerNorm(d_model))
        M = len(self.names)
        self.mod_emb = nn.Parameter(torch.randn(M, d_model) * 0.02)
        if fusion == "attention":
            self.self_attn = (nn.TransformerEncoderLayer(d_model, heads, 2 * d_model, dropout, batch_first=True,
                                                         norm_first=True) if token_self_attention else None)
            self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            self.cross = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
            self.gate = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.Sigmoid())
            head_in = d_model
        elif fusion == "concat":
            head_in = M * d_model
        else:
            raise ValueError(fusion)
        self.head = nn.Sequential(nn.LayerNorm(head_in), nn.Linear(head_in, hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def modality_mask(self, x):
        """True = modality unavailable for the whole window."""
        B = x.size(0)
        masks = []
        for m in self.names:
            if m in self.missing_idx:
                masks.append(x[:, :, self.missing_idx[m]].min(1).values > 0.5)
            else:
                masks.append(torch.zeros(B, dtype=torch.bool, device=x.device))
        return torch.stack(masks, 1)

    def forward(self, x, ctx=None, return_tokens: bool = False):
        tokens = []
        for m in self.modality_idx:
            idx = self.modality_idx[m].to(x.device)
            tokens.append(self.proj[m](self.encoders[m](x.index_select(2, idx))))
        if self.use_ctx:
            tokens.append(self.ctx_enc(ctx))
        T = torch.stack(tokens, 1) + self.mod_emb.unsqueeze(0)
        mask = self.modality_mask(x)
        if self.training and self.modality_dropout > 0:
            drop = torch.rand(mask.shape, device=x.device) < self.modality_dropout
            mask = mask | drop
        all_masked = mask.all(1)
        if all_masked.any():  # always keep the climate token (index 0)
            mask = mask.clone()
            mask[all_masked, 0] = False
        if self.fusion == "concat":
            z = (T * (~mask).unsqueeze(-1)).flatten(1)
            return self.head(z), None
        if self.self_attn is not None:
            T = self.self_attn(T, src_key_padding_mask=mask)
        q = self.query.expand(x.size(0), -1, -1)
        fused, w = self.cross(q, T, T, key_padding_mask=mask, need_weights=True, average_attn_weights=True)
        fused = fused.squeeze(1)
        keep = (~mask).unsqueeze(-1).float()
        mean_tok = (T * keep).sum(1) / keep.sum(1).clamp(min=1)
        g = self.gate(torch.cat([fused, mean_tok], -1))
        z = g * fused + (1 - g) * mean_tok
        out = self.head(z)
        if return_tokens:
            return out, w.squeeze(1), T
        return out, w.squeeze(1)
