"""Training loop, model factory and experiment registry for the neural models.

The held-out test split is never touched here: training uses the train split, early stopping
and model selection use the validation split.  Test inference lives in scripts/evaluate_models.py.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from ..data.windows import TL_FEATURES, PanelArrays
from ..evaluation.metrics import compute_metrics
from ..features.engineering import CONTEXT_FEATURES, HUMIDITY_DERIVED, MODALITIES
from ..models import MAFNet, SequenceClassifier
from ..utils import ROOT, atomic_write_json, read_json, set_seed

DEFAULTS = dict(
    model="mafnet", lookback=14, horizon=1, hidden=64, d_model=64, layers=1, dropout=0.2, heads=2,
    lr=1e-3, batch_size=256, optimizer="adamw", weight_decay=1e-4, loss="wce", focal_gamma=2.0,
    epochs=40, patience=6, seed=42, modalities=["climate", "air", "dryness"], use_context=True,
    fusion="attention", encoder="gru", air_encoder="gru", dry_encoder="gru", token_self_attention=True,
    modality_dropout=0.1, drop_features=[], max_missing=3, train_sites=None, grad_clip=1.0,
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# the training loop is GPU-bound; limiting intra-op CPU threads lets several runs share the CPU
torch.set_num_threads(int(os.environ.get("CS_TORCH_THREADS", "3")))


def run_id(cfg: dict) -> str:
    key = json.dumps({k: cfg[k] for k in sorted(cfg) if k not in ("group", "epochs_override")}, sort_keys=True, default=str)
    return hashlib.sha1(key.encode()).hexdigest()[:10]


@lru_cache(maxsize=1)
def get_arrays() -> PanelArrays:
    panel = pd.read_parquet(ROOT / "data/processed/panel.parquet")
    tl = ROOT / "data/processed/drought_transfer_features.parquet"
    if tl.exists():
        panel = panel.merge(pd.read_parquet(tl), on=["site_id", "date"], how="left")
    splits = {k: tuple(pd.Timestamp(x) for x in v) for k, v in read_json(ROOT / "data/manifests/splits.json").items()}
    return PanelArrays(panel, splits)


def feature_sets(cfg: dict, arr: PanelArrays) -> dict:
    """Map each requested modality to its feature indices (minus ablated features) + missing flag."""
    drop = set(cfg.get("drop_features") or [])
    if "humidity" in drop:
        drop |= set(HUMIDITY_DERIVED)
    flags = {"climate": "climate_missing", "air": "air_missing", "dryness": "dry_missing"}
    out = {}
    for m in cfg["modalities"]:
        names = [f for f in MODALITIES[m] if f not in drop]
        if m == "dryness" and cfg.get("transfer_drought"):
            names += TL_FEATURES
        names += [flags[m]]
        out[m] = arr.feature_idx(names)
    return out


def build_model(cfg: dict, arr: PanelArrays) -> nn.Module:
    fs = feature_sets(cfg, arr)
    n_ctx = len(CONTEXT_FEATURES) if cfg["use_context"] else 0
    if cfg["model"] == "mafnet":
        missing_idx = {m: arr.feature_names.index(f) for m, f in
                       {"climate": "climate_missing", "air": "air_missing", "dryness": "dry_missing"}.items() if m in fs}
        return MAFNet(fs, missing_idx, n_ctx, cfg["d_model"], cfg["hidden"], cfg["encoder"], cfg["air_encoder"],
                      cfg["dry_encoder"], cfg["layers"], cfg["dropout"], cfg["heads"], cfg["lookback"],
                      cfg["fusion"], cfg["token_self_attention"], cfg["modality_dropout"])
    n_in = sum(len(v) for v in fs.values())
    return SequenceClassifier(n_in, n_ctx, cfg["hidden"], cfg["model"], cfg["layers"], cfg["dropout"],
                              cfg["heads"], cfg["lookback"])


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight, self.gamma = weight, gamma

    def forward(self, logits, y):
        logp = F.log_softmax(logits, -1)
        ce = F.nll_loss(logp, y, weight=self.weight, reduction="none")
        pt = logp.gather(1, y[:, None]).squeeze(1).exp()
        return ((1 - pt) ** self.gamma * ce).mean()


def make_loss(cfg, y_train):
    counts = np.bincount(y_train, minlength=4).astype(float)
    w = torch.tensor(counts.sum() / (4 * counts), dtype=torch.float32, device=DEVICE)
    w = w / w.mean()
    if cfg["loss"] == "ce":
        return nn.CrossEntropyLoss()
    if cfg["loss"] == "wce":
        return nn.CrossEntropyLoss(weight=w)
    if cfg["loss"] == "focal":
        return FocalLoss(weight=w.sqrt(), gamma=cfg["focal_gamma"])
    raise ValueError(cfg["loss"])


class WindowBatcher:
    """Gathers sliding windows directly from the full panel tensor kept on the device."""

    def __init__(self, arr: PanelArrays, cfg: dict, cols: list[int]):
        self.X = torch.as_tensor(arr.X[:, cols], device=DEVICE)
        self.C = torch.as_tensor(arr.C, device=DEVICE)
        self.y = torch.as_tensor(np.nan_to_num(arr.y, nan=-1).astype(np.int64), device=DEVICE)
        self.offs = torch.arange(-cfg["lookback"] + 1, 1, device=DEVICE)
        self.h = cfg["horizon"]

    def get(self, idx: torch.Tensor):
        return self.X[idx[:, None] + self.offs[None, :]], self.C[idx], self.y[idx + self.h]


def _column_plan(cfg, arr):
    """Columns of arr.X fed to the model; feature indices are remapped into this sub-matrix."""
    fs = feature_sets(cfg, arr)
    cols = sorted({i for v in fs.values() for i in v})
    remap = {c: j for j, c in enumerate(cols)}
    return cols, remap


def build_model_on_cols(cfg, arr):
    """Build a model whose feature indices refer to the column subset given by _column_plan."""
    cols, remap = _column_plan(cfg, arr)

    class _Sub:  # lightweight view exposing the remapped feature names
        feature_names = [arr.feature_names[c] for c in cols]

        def feature_idx(self, names):
            return [self.feature_names.index(n) for n in names]

    model = build_model(cfg, _Sub())
    return model, cols


@torch.no_grad()
def predict(model, batcher: WindowBatcher, idx: np.ndarray, bs: int = 2048):
    model.eval()
    probs, atts, ys = [], [], []
    t_idx = torch.as_tensor(idx, device=DEVICE)
    for i in range(0, len(t_idx), bs):
        x, c, y = batcher.get(t_idx[i:i + bs])
        logits, w = model(x, c)
        probs.append(F.softmax(logits.float(), -1).cpu())
        ys.append(y.cpu())
        if w is not None:
            atts.append(w.float().cpu())
    P = torch.cat(probs).numpy()
    A = torch.cat(atts).numpy() if atts else None
    return P, torch.cat(ys).numpy(), A


def train(cfg_in: dict, group: str = "misc", force: bool = False, verbose: bool = True,
          trial=None) -> dict:
    cfg = {**DEFAULTS, **cfg_in}
    rid = f"{group}-{cfg['model']}-{run_id(cfg)}"
    out_json = ROOT / "experiments" / group / f"{rid}.json"
    if out_json.exists() and not force:
        rec = read_json(out_json)
        if rec.get("status") == "complete":
            return rec
    set_seed(cfg["seed"])
    arr = get_arrays()
    sites = None
    if cfg.get("train_sites") is not None:
        sites = np.asarray(cfg["train_sites"])
    tr = arr.sample_index(cfg["lookback"], cfg["horizon"], "train", cfg["max_missing"], sites)
    va = arr.sample_index(cfg["lookback"], cfg["horizon"], "val", cfg["max_missing"], sites)
    model, cols = build_model_on_cols(cfg, arr)
    model = model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    batcher = WindowBatcher(arr, cfg, cols)
    y_tr = arr.y[tr + cfg["horizon"]].astype(int)
    loss_fn = make_loss(cfg, y_tr)
    opt_cls = torch.optim.AdamW if cfg["optimizer"] == "adamw" else torch.optim.Adam
    opt = opt_cls(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    ckpt = ROOT / "models/checkpoints" / f"{rid}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator(device="cpu").manual_seed(cfg["seed"])
    hist, best, bad, nan_batches = [], -1.0, 0, 0
    t0 = time.time()
    for ep in range(cfg["epochs"]):
        model.train()
        perm = torch.as_tensor(tr[torch.randperm(len(tr), generator=gen).numpy()], device=DEVICE)
        tot, nb = 0.0, 0
        for i in range(0, len(perm), cfg["batch_size"]):
            x, c, y = batcher.get(perm[i:i + cfg["batch_size"]])
            logits, _ = model(x, c)
            loss = loss_fn(logits, y)
            if not torch.isfinite(loss):
                nan_batches += 1
                opt.zero_grad(set_to_none=True)
                if nan_batches > 20:
                    break
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            tot += loss.item()
            nb += 1
        P, Y, _ = predict(model, batcher, va)
        m = compute_metrics(Y, P)
        with torch.no_grad():
            val_loss = float(F.nll_loss(torch.log(torch.as_tensor(P).clamp_min(1e-7)), torch.as_tensor(Y)).item())
        # train metrics on a fixed subsample for the learning curves
        sub = tr[:: max(1, len(tr) // 20000)]
        Pt, Yt, _ = predict(model, batcher, sub)
        mt = compute_metrics(Yt, Pt)
        tr_loss_eval = float(F.nll_loss(torch.log(torch.as_tensor(Pt).clamp_min(1e-7)), torch.as_tensor(Yt)).item())
        lr_now = opt.param_groups[0]["lr"]
        hist.append(dict(epoch=ep + 1, train_loss=tot / max(nb, 1), train_nll=tr_loss_eval, val_nll=val_loss,
                         train_macro_f1=mt["macro_f1"], val_macro_f1=m["macro_f1"],
                         val_balanced_accuracy=m["balanced_accuracy"], val_accuracy=m["accuracy"], lr=lr_now))
        if verbose:
            print(f"  ep{ep + 1:02d} loss={tot / max(nb, 1):.4f} valF1={m['macro_f1']:.4f} valBA={m['balanced_accuracy']:.4f} lr={lr_now:.1e}", flush=True)
        sched.step(m["macro_f1"])
        if m["macro_f1"] > best + 1e-4:
            best, bad = m["macro_f1"], 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
        if trial is not None:
            trial.report(m["macro_f1"], ep)
            if trial.should_prune():
                break
        if bad >= cfg["patience"] or nan_batches > 20:
            break
    train_time = time.time() - t0
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    P, Y, A = predict(model, batcher, va)
    vm = compute_metrics(Y, P)
    rec = dict(run_id=rid, group=group, config=cfg, status="complete", n_params=int(n_params),
               n_train=int(len(tr)), n_val=int(len(va)), best_epoch=int(np.argmax([h["val_macro_f1"] for h in hist]) + 1),
               epochs_run=len(hist), train_time_s=train_time, nan_batches=nan_batches, history=hist,
               val_metrics=vm, checkpoint=str(ckpt.relative_to(ROOT)).replace("\\", "/"),
               device=str(DEVICE), finished_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    np.savez_compressed(ROOT / "results/predictions" / f"{rid}_val.npz", idx=va, proba=P, y=Y,
                        att=A if A is not None else np.zeros(0))
    atomic_write_json(rec, out_json)
    with open(ROOT / "experiments/registry/runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({k: rec[k] for k in ("run_id", "group", "n_params", "train_time_s", "finished_at")}
                           | {"val_macro_f1": vm["macro_f1"], "model": cfg["model"]}) + "\n")
    return rec


def load_trained(rec: dict):
    """Rebuild a trained model + batcher from its registry record (for evaluation/inference)."""
    cfg = rec["config"]
    arr = get_arrays()
    model, cols = build_model_on_cols(cfg, arr)
    model.load_state_dict(torch.load(ROOT / rec["checkpoint"], map_location=DEVICE))
    model.to(DEVICE).eval()
    return model, WindowBatcher(arr, cfg, cols), arr
