"""Chronological splitting, leakage-safe scaling and sliding-window sample construction.

A sample is (site, t): inputs are days t-L+1 .. t, the target is the class observed on day t+h.
A sample belongs to a split only if its *first input day* and its *target day* both fall in
that split's date range, so no input or target crosses a split boundary (implicit embargo of
L + h days).  Scalers are fitted on training-period rows only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..features.engineering import AIR_FEATURES, CLIMATE_FEATURES, CONTEXT_FEATURES, DRY_FEATURES

SEQ_FEATURES = CLIMATE_FEATURES + AIR_FEATURES + DRY_FEATURES
# optional features from the encoder transferred from the US drought benchmark
TL_FEATURES = ["tl_drought_p0", "tl_drought_p1", "tl_drought_p2", "tl_drought_p3", "tl_drought_severity"]
MISSING_FLAGS = ["climate_missing", "air_missing", "dry_missing"]


def split_dates(dates: pd.Series, train_frac=0.70, val_frac=0.15) -> dict:
    """Common calendar cut-offs for all sites."""
    d0, d1 = dates.min(), dates.max()
    n = (d1 - d0).days + 1
    train_end = d0 + pd.Timedelta(days=int(n * train_frac) - 1)
    val_end = d0 + pd.Timedelta(days=int(n * (train_frac + val_frac)) - 1)
    return {
        "train": (d0, train_end),
        "val": (train_end + pd.Timedelta(days=1), val_end),
        "test": (val_end + pd.Timedelta(days=1), d1),
    }


@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray
    columns: list = field(default_factory=list)

    @classmethod
    def fit(cls, df: pd.DataFrame, columns: list) -> "Scaler":
        x = df[columns].astype(float)
        std = x.std().fillna(1.0).replace(0, 1.0)
        return cls(x.mean().fillna(0.0).values, std.values, list(columns))

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        x = (df[self.columns].astype(float).values - self.mean) / self.std
        return np.nan_to_num(x, nan=0.0).astype(np.float32)

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "columns": self.columns}

    @classmethod
    def from_dict(cls, d):
        return cls(np.array(d["mean"]), np.array(d["std"]), d["columns"])


def add_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["climate_missing"] = df["temp_c"].isna().astype(np.float32)
    df["air_missing"] = df[["pm25", "pm10", "o3", "no2"]].isna().all(1).astype(np.float32)
    df["dry_missing"] = df["vpd_kpa"].isna().astype(np.float32)
    return df


class PanelArrays:
    """Dense arrays over the site-day grid, plus window index generation."""

    def __init__(self, panel: pd.DataFrame, splits: dict, scaler: Scaler | None = None,
                 ctx_scaler: Scaler | None = None):
        p = add_missing_flags(panel.sort_values(["site_id", "date"]).reset_index(drop=True))
        self.panel = p
        self.splits = splits
        train_rows = p[p.date <= splits["train"][1]]
        seq = SEQ_FEATURES + [c for c in TL_FEATURES if c in p.columns]
        self.scaler = scaler or Scaler.fit(train_rows, seq)
        self.ctx_scaler = ctx_scaler or Scaler.fit(train_rows.drop_duplicates("site_id"), CONTEXT_FEATURES)
        self.X = np.concatenate([self.scaler.transform(p), p[MISSING_FLAGS].values.astype(np.float32)], 1)
        self.feature_names = list(self.scaler.columns) + MISSING_FLAGS
        self.C = self.ctx_scaler.transform(p)
        self.y = p["risk_class"].values
        self.observed = p["observed"].values.astype(bool)
        self.site_codes, self.site_index = pd.factorize(p["site_id"])
        self.dates = p["date"].values
        self.date_ns = p["date"].values.astype("datetime64[D]")

    def feature_idx(self, names) -> list:
        return [self.feature_names.index(n) for n in names]

    def sample_index(self, lookback: int, horizon: int, split: str, max_missing: int = 3,
                     sites: np.ndarray | None = None) -> np.ndarray:
        """Row indices t of the last input day for every valid sample of `split`."""
        n = len(self.y)
        t = np.arange(n)
        start, tgt = t - lookback + 1, t + horizon
        ok = (start >= 0) & (tgt < n)
        start_c, tgt_c = np.clip(start, 0, n - 1), np.clip(tgt, 0, n - 1)
        ok &= (self.site_codes[start_c] == self.site_codes) & (self.site_codes[tgt_c] == self.site_codes)
        ok &= ~np.isnan(self.y[tgt_c]) & self.observed[tgt_c]
        miss = np.concatenate([[0], np.cumsum(~self.observed)])
        n_missing = miss[t + 1] - miss[start_c]
        ok &= n_missing <= max_missing
        ok &= self.X[:, self.feature_names.index("climate_missing")] == 0  # last input day usable
        lo, hi = (np.datetime64(d.date() if hasattr(d, "date") else d, "D") for d in self.splits[split])
        ok &= (self.date_ns[start_c] >= lo) & (self.date_ns[tgt_c] <= hi)
        if sites is not None:
            ok &= np.isin(self.site_codes, sites)
        return t[ok]

    def windows(self, idx: np.ndarray, lookback: int, horizon: int):
        """Materialise (X_seq [N,L,F], C [N,K], y [N]) for sample indices."""
        offs = np.arange(-lookback + 1, 1)
        X = self.X[idx[:, None] + offs[None, :]]
        return X, self.C[idx], self.y[idx + horizon].astype(np.int64)


def tabular_features(X: np.ndarray, C: np.ndarray, names: list) -> tuple[np.ndarray, list]:
    """Aggregate windows for classical models: last, mean, max, min, std, trend + context."""
    last, mean = X[:, -1], X.mean(1)
    mx, mn, sd = X.max(1), X.min(1), X.std(1)
    trend = X[:, -1] - X[:, 0]
    feats = np.concatenate([last, mean, mx, mn, sd, trend, C], 1)
    cols = [f"{s}_{n}" for s in ["last", "mean", "max", "min", "std", "trend"] for n in names]
    return feats.astype(np.float32), cols + [f"ctx_{c}" for c in CONTEXT_FEATURES]
