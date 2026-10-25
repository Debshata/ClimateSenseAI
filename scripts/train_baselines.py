"""Non-neural baselines on aggregated window features (validation only; test in evaluate_models.py).

  persistence  : predict today's observed class for t+h (the hardest naive baseline)
  majority     : most frequent training class
  climatology  : most frequent training class for the same site and calendar month
  logreg, random_forest, hist_gradient_boosting, mlp (sklearn) on tabular window aggregates
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from climatesense.data.windows import tabular_features
from climatesense.evaluation.metrics import compute_metrics
from climatesense.training.trainer import get_arrays
from climatesense.utils import atomic_write_json, read_json, set_seed

L_DEFAULT, H_DEFAULT = 14, 1


def naive_proba(kind: str, arr, idx, h, train_idx, smooth=0.02):
    """Probabilities for naive rules (one-hot with light smoothing so ROC is defined)."""
    if kind == "persistence":
        y_prev = arr.y[idx]
        # if today is unlabeled (rare: filled day), fall back to the last labeled day in the window
        for k in range(1, 4):
            miss = np.isnan(y_prev)
            y_prev[miss] = arr.y[idx[miss] - k]
        pred = np.nan_to_num(y_prev, nan=0).astype(int)
    elif kind == "majority":
        pred = np.full(len(idx), np.bincount(arr.y[train_idx + h].astype(int)).argmax())
    elif kind == "climatology":
        p = arr.panel
        tr = pd.DataFrame({"site": arr.site_codes[train_idx + h], "m": p.date.values[train_idx + h].astype("datetime64[M]").astype(int) % 12,
                           "y": arr.y[train_idx + h].astype(int)})
        mode = tr.groupby(["site", "m"]).y.agg(lambda s: s.value_counts().idxmax())
        glob = int(np.bincount(tr.y).argmax())
        keys = list(zip(arr.site_codes[idx + h], p.date.values[idx + h].astype("datetime64[M]").astype(int) % 12))
        pred = np.array([mode.get(k, glob) for k in keys])
    P = np.full((len(idx), 4), smooth / 3)
    P[np.arange(len(idx)), pred] = 1 - smooth
    return P


def sklearn_models(seed):
    return {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features="sqrt",
                                                class_weight="balanced_subsample", n_jobs=-1, random_state=seed),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                                                                 l2_regularization=1.0, class_weight="balanced",
                                                                 early_stopping=True, validation_fraction=0.1,
                                                                 random_state=seed),
        "mlp": MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-3, batch_size=256, learning_rate_init=1e-3,
                             early_stopping=True, validation_fraction=0.1, max_iter=60, random_state=seed),
    }


def tabular_split(arr, idx, L, h):
    X, C, y = arr.windows(idx, L, h)
    F, cols = tabular_features(X, C, arr.feature_names)
    return F, y, cols


def main(L=L_DEFAULT, h=H_DEFAULT, seed=42, only=None):
    set_seed(seed)
    arr = get_arrays()
    tr, va = arr.sample_index(L, h, "train"), arr.sample_index(L, h, "val")
    out_dir = ROOT / "experiments/baseline"
    results = {}
    yv = arr.y[va + h].astype(int)
    for kind in ["persistence", "majority", "climatology"]:
        name = f"{kind}_L{L}_h{h}"
        P = naive_proba(kind, arr, va, h, tr)
        results[name] = dict(model=kind, val_metrics=compute_metrics(yv, P), train_time_s=0.0, n_params=0)
        np.savez_compressed(ROOT / "results/predictions" / f"baseline-{name}_val.npz", idx=va, proba=P, y=yv)
        print(name, round(results[name]["val_metrics"]["macro_f1"], 4))
    Ftr, ytr, cols = tabular_split(arr, tr, L, h)
    Fva, _, _ = tabular_split(arr, va, L, h)
    for name, model in sklearn_models(seed).items():
        if only and name not in only:
            continue
        rid = f"{name}_L{L}_h{h}"
        f_json = out_dir / f"baseline-{rid}.json"
        if f_json.exists():
            results[rid] = read_json(f_json)
            print(rid, "cached", round(results[rid]["val_metrics"]["macro_f1"], 4))
            continue
        t0 = time.time()
        Xfit, yfit = Ftr, ytr
        if name == "logreg":  # standardise aggregates for the linear model
            mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
            model.fit((Ftr - mu) / sd, ytr)
            P = model.predict_proba((Fva - mu) / sd)
            model = {"model": model, "mu": mu, "sd": sd}
        else:
            model.fit(Xfit, yfit)
            P = model.predict_proba(Fva)
        dt = time.time() - t0
        est = model["model"] if isinstance(model, dict) else model
        if name == "logreg":
            n_params = int(est.coef_.size + est.intercept_.size)
        elif name == "mlp":
            n_params = int(sum(w.size for w in est.coefs_) + sum(b.size for b in est.intercepts_))
        elif name == "random_forest":
            n_params = int(sum(t.tree_.node_count for t in est.estimators_))  # number of tree nodes
        else:
            n_params = int(sum(p.nodes.shape[0] for pp in est._predictors for p in pp))
        rec = dict(model=name, run_id=f"baseline-{rid}", lookback=L, horizon=h, seed=seed, val_metrics=compute_metrics(yv, P),
                   train_time_s=dt, n_params=n_params, n_params_note="tree nodes" if "forest" in name or "boost" in name else "weights",
                   n_features=len(cols), status="complete")
        joblib.dump(model, ROOT / "models/checkpoints" / f"baseline-{rid}.joblib", compress=3)
        np.savez_compressed(ROOT / "results/predictions" / f"baseline-{rid}_val.npz", idx=va, proba=P, y=yv)
        atomic_write_json(rec, f_json)
        results[rid] = rec
        print(rid, round(rec["val_metrics"]["macro_f1"], 4), f"{dt:.0f}s", flush=True)
    atomic_write_json({k: {"val_macro_f1": v["val_metrics"]["macro_f1"], "val_accuracy": v["val_metrics"]["accuracy"],
                           "val_balanced_accuracy": v["val_metrics"]["balanced_accuracy"]} for k, v in results.items()},
                      out_dir / f"summary_L{L}_h{h}.json")
    return results


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, default=L_DEFAULT)
    ap.add_argument("--horizon", type=int, default=H_DEFAULT)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    main(a.lookback, a.horizon, only=a.only)
