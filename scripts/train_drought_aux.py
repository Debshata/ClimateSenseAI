"""Auxiliary drought benchmark (US counties) and cross-domain transfer of a drought encoder.

1. Benchmark: predict the US Drought Monitor category (None / D0-D1 / D2 / D3-D4) for a county
   from the preceding 30 days of NASA POWER weather, using the dataset's own chronological
   train (2000-2016) / validation (2017-18) / test (2019-20) files.  A random subset of
   counties is used to bound compute.
2. Transfer: the trained encoder uses only variables that also exist in the World Weather
   Repository (temperature, relative humidity, specific humidity, precipitation, wind speed).
   It is applied to every capital-city site's past 30 days to produce "transferred drought
   probabilities", which are added as optional features to the dryness branch.  The US data is
   never joined to capital-city records; only the learned function is transferred.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

from climatesense.evaluation.metrics import compute_metrics
from climatesense.evaluation.plotting import save, setup
from climatesense.utils import atomic_write_json, set_seed

L = 30
N_COUNTIES = 400
SHARED = ["t", "rh", "q", "precip", "wind"]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DR = ROOT / "data/raw/drought_us"
OUT = ROOT / "experiments/drought_aux"
CLASSES = ["None", "D0-D1", "D2", "D3-D4"]


def rh_from_q(q_gkg, t_c, p_kpa):
    q = q_gkg / 1000.0
    e = q * p_kpa / (0.622 + 0.378 * q)
    es = 0.6108 * np.exp(17.27 * t_c / (t_c + 237.3))
    return np.clip(100 * e / es, 0, 100)


def q_from_rh(rh, t_c, p_kpa=101.325):
    e = rh / 100 * 0.6108 * np.exp(17.27 * t_c / (t_c + 237.3))
    return 1000 * 0.622 * e / (p_kpa - 0.378 * e)


def score_to_class(s):
    r = np.round(s)
    return np.select([r <= 0, r <= 2, r <= 3], [0, 1, 2], 3)


def load_part(part, fips):
    cache = OUT / f"{part}_subset.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    chunks = []
    for ch in pd.read_csv(DR / f"{part}_timeseries/{part}_timeseries.csv", chunksize=2_000_000,
                          usecols=["fips", "date", "PRECTOT", "PS", "QV2M", "T2M", "WS10M", "score"]):
        chunks.append(ch[ch.fips.isin(fips)])
    df = pd.concat(chunks)
    df.to_parquet(cache, index=False)
    return df


def to_shared(df):
    return pd.DataFrame({"t": df.T2M, "rh": rh_from_q(df.QV2M, df.T2M, df.PS), "q": df.QV2M,
                         "precip": np.log1p(df.PRECTOT.clip(lower=0)), "wind": df.WS10M})


def windows(df, mu, sd):
    df = df.sort_values(["fips", "date"]).reset_index(drop=True)
    X = ((to_shared(df).values - mu) / sd).astype(np.float32)
    code = df.fips.values
    t = np.arange(len(df))
    ok = df.score.notna().values & (t >= L - 1)
    ok[ok] &= code[t[ok] - L + 1] == code[t[ok]]
    idx = t[ok]
    y = score_to_class(df.score.values[idx]).astype(np.int64)
    return X, idx, y


class DroughtGRU(nn.Module):
    def __init__(self, n_in=5, hidden=64):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.2), nn.Linear(hidden, 4))

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1])


def batches(X, idx, bs, shuffle, gen=None):
    Xt = torch.as_tensor(X, device=DEV)
    order = torch.as_tensor(idx[torch.randperm(len(idx), generator=gen).numpy()] if shuffle else idx, device=DEV)
    offs = torch.arange(-L + 1, 1, device=DEV)
    for i in range(0, len(order), bs):
        j = order[i:i + bs]
        yield Xt[j[:, None] + offs]


@torch.no_grad()
def predict(model, X, idx):
    model.eval()
    return np.concatenate([F.softmax(model(x), -1).cpu().numpy() for x in batches(X, idx, 4096, False)])


def main():
    setup()
    set_seed(42)
    OUT.mkdir(parents=True, exist_ok=True)
    soil = pd.read_csv(DR / "soil_data.csv")
    fips = np.sort(np.random.default_rng(0).choice(soil.fips.values, N_COUNTIES, replace=False))
    parts = {p: load_part(p, fips) for p in ["train", "validation", "test"]}
    sh = to_shared(parts["train"])
    mu, sd = sh.mean().values, sh.std().values
    data = {p: windows(d, mu, sd) for p, d in parts.items()}
    Xtr, itr, ytr = data["train"]
    counts = np.bincount(ytr, minlength=4)
    w = torch.tensor(counts.sum() / (4 * counts), dtype=torch.float32, device=DEV)
    model = DroughtGRU().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ytr_t = torch.as_tensor(ytr, device=DEV)
    gen = torch.Generator().manual_seed(42)
    best, bad, hist = -1, 0, []
    t0 = time.time()
    for ep in range(25):
        model.train()
        perm = torch.randperm(len(itr), generator=gen).numpy()
        Xt = torch.as_tensor(Xtr, device=DEV)
        offs = torch.arange(-L + 1, 1, device=DEV)
        it = torch.as_tensor(itr, device=DEV)
        for i in range(0, len(perm), 512):
            b = torch.as_tensor(perm[i:i + 512], device=DEV)
            x = Xt[it[b][:, None] + offs]
            loss = F.cross_entropy(model(x), ytr_t[b], weight=w)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        Xv, iv, yv = data["validation"]
        m = compute_metrics(yv, predict(model, Xv, iv))
        hist.append({"epoch": ep + 1, "val_macro_f1": m["macro_f1"], "val_balanced_accuracy": m["balanced_accuracy"]})
        print(f"ep{ep + 1} valF1={m['macro_f1']:.4f}", flush=True)
        if m["macro_f1"] > best + 1e-4:
            best, bad = m["macro_f1"], 0
            torch.save(model.state_dict(), OUT / "drought_gru.pt")
        else:
            bad += 1
            if bad >= 5:
                break
    model.load_state_dict(torch.load(OUT / "drought_gru.pt", map_location=DEV))
    res = {"n_counties": N_COUNTIES, "lookback_days": L, "shared_features": SHARED, "classes": CLASSES,
           "train_time_s": time.time() - t0, "history": hist, "standardisation": {"mean": mu.tolist(), "std": sd.tolist()},
           "n_params": sum(p.numel() for p in model.parameters())}
    for p in ["validation", "test"]:
        Xp, ip, yp = data[p]
        P = predict(model, Xp, ip)
        res[f"{p}_metrics"] = compute_metrics(yp, P)
        # persistence-free naive reference: majority class
        res[f"{p}_majority_macro_f1"] = compute_metrics(yp, np.eye(4)[np.full(len(yp), np.bincount(ytr).argmax())] * 0.97 + 0.01)["macro_f1"]
    res["n_samples"] = {p: int(len(d[1])) for p, d in data.items()}
    atomic_write_json(res, OUT / "drought_aux_results.json")
    print("test macroF1", res["test_metrics"]["macro_f1"], "majority", res["test_majority_macro_f1"])

    # confusion matrix figure for the auxiliary benchmark
    cm = np.array(res["test_metrics"]["confusion_matrix"], float)
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xticks(range(4), CLASSES, rotation=30); ax.set_yticks(range(4), CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("US drought benchmark (test 2019–20)")
    save(fig, ROOT / "results/figures/drought_aux_confusion.png")

    # ---------------------------------------------------------------- transfer to capital-city sites
    panel = pd.read_parquet(ROOT / "data/processed/panel.parquet").sort_values(["site_id", "date"]).reset_index(drop=True)
    site_sh = pd.DataFrame({"t": panel.temp_c, "rh": panel.rh, "q": q_from_rh(panel.rh, panel.temp_c),
                            "precip": np.log1p(panel.precip_mm.clip(lower=0)), "wind": panel.wind_kph / 3.6})
    Xs = np.nan_to_num(((site_sh.values - mu) / sd).astype(np.float32))
    code = pd.factorize(panel.site_id)[0]
    t = np.arange(len(panel))
    ok = t >= L - 1
    ok[ok] &= code[t[ok] - L + 1] == code[t[ok]]
    P = np.full((len(panel), 4), np.nan, np.float32)
    P[t[ok]] = predict(model, Xs, t[ok])
    tl = pd.DataFrame({"site_id": panel.site_id, "date": panel.date})
    for k in range(4):
        tl[f"tl_drought_p{k}"] = P[:, k]
    tl["tl_drought_severity"] = (P * np.arange(4)).sum(1)
    tl.to_parquet(ROOT / "data/processed/drought_transfer_features.parquet", index=False)
    res["transfer_feature_summary"] = {"rows_with_features": int(ok.sum()),
                                       "mean_severity": float(np.nanmean(tl.tl_drought_severity)),
                                       "share_argmax_ge_D2": float(np.nanmean(np.nanargmax(np.nan_to_num(P[ok], nan=0), 1) >= 2))}
    atomic_write_json(res, OUT / "drought_aux_results.json")
    print(res["transfer_feature_summary"])


if __name__ == "__main__":
    main()
