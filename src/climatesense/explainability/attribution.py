"""Model-agnostic and gradient-based attributions for the sequence models.

Attention weights describe how the fusion module weighted modality tokens; they are *not*
causal explanations.  Permutation importance and integrated gradients are reported alongside.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import f1_score


@torch.no_grad()
def _proba(model, X, C, bs=2048):
    out = []
    for i in range(0, len(X), bs):
        logits, _ = model(X[i:i + bs], C[i:i + bs])
        out.append(torch.softmax(logits, -1))
    return torch.cat(out)


def permutation_importance(model, X, C, y, feature_groups: dict, repeats=3, seed=0):
    """Drop in macro-F1 when a feature (or group of features) is shuffled across samples.

    The whole 14-day trajectory of the feature is permuted jointly, so temporal structure
    within a window is kept while its association with the target is broken.
    """
    model.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    yv = y.cpu().numpy()
    base = f1_score(yv, _proba(model, X, C).argmax(1).cpu().numpy(), average="macro")
    res = {}
    for name, cols in feature_groups.items():
        drops = []
        for _ in range(repeats):
            Xp = X.clone()
            perm = torch.randperm(len(X), generator=g).to(X.device)
            Xp[:, :, cols] = X[perm][:, :, cols]
            f = f1_score(yv, _proba(model, Xp, C).argmax(1).cpu().numpy(), average="macro")
            drops.append(base - f)
        res[name] = {"mean_drop_macro_f1": float(np.mean(drops)), "std": float(np.std(drops))}
    return float(base), res


def integrated_gradients(model, X, C, target, steps=32):
    """IG w.r.t. the sequence input with an all-zero baseline (= training mean after scaling).

    Returns |attribution| summed over time: [N, F].
    """
    model.eval()
    total = torch.zeros_like(X)
    for a in torch.linspace(1.0 / steps, 1.0, steps):
        xi = (a * X).detach().requires_grad_(True)
        logits, _ = model(xi, C)
        sel = logits.gather(1, target[:, None]).sum()
        grad, = torch.autograd.grad(sel, xi)
        total += grad
    attr = X * total / steps
    return attr.abs().sum(1).detach()
