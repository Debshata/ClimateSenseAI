"""Project-defined four-class compound environmental habitability-risk label.

Exact equations and sources: docs/label_definition.md.  Hazard scores are computed from the
*observed* (never imputed) values of a site-day; days that were not observed get no label.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.engineering import heat_index_c, vpd_kpa, wet_bulb_stull_c

CLASS_NAMES = ["Low", "Moderate", "High", "Critical"]


def band(x, edges) -> np.ndarray:
    """0 below edges[0], 1 in [e0,e1), 2 in [e1,e2), 3 at/above e2; NaN stays NaN."""
    x = np.asarray(x, float)
    out = np.digitize(x, edges).astype(float)
    out[np.isnan(x)] = np.nan
    return out


def heat_score(t_c, rh, cfg) -> np.ndarray:
    hi = band(heat_index_c(t_c, rh), cfg["heat_index_c"])
    wb = wet_bulb_stull_c(t_c, rh)
    wb_s = np.where(wb >= cfg["wet_bulb_c"]["level3"], 3, np.where(wb >= cfg["wet_bulb_c"]["level2"], 2, 0))
    out = np.fmax(hi, wb_s)
    out[np.isnan(hi)] = np.nan
    return out


def air_score(pm25, pm10, o3, no2, cfg) -> np.ndarray:
    parts = np.vstack([band(pm25, cfg["pm25"]), band(pm10, cfg["pm10"]),
                       band(o3, cfg["o3"]), band(no2, cfg["no2"])])
    with np.errstate(all="ignore"):
        out = np.nanmax(np.where(np.isnan(parts), -1, parts), axis=0)
    out = np.where(np.isnan(parts).all(0), np.nan, out)
    return out


def dryness_score(t_c, rh, wet_frac_30d_count, n_obs_30d, cfg) -> np.ndarray:
    """VPD aridity band, +1 (capped at 3) for a dry spell with no recorded precipitation."""
    base = band(vpd_kpa(t_c, rh), cfg["vpd_kpa"])
    spell = (wet_frac_30d_count == 0) & (n_obs_30d >= cfg["dry_spell_min_obs"])
    out = np.minimum(base + spell.astype(float), 3)
    out[np.isnan(base)] = np.nan
    return out


def compound_class(h, a, d, cfg) -> np.ndarray:
    """Aggregate hazard scores (0-3 each) into the four ordinal classes.

    S = H + A + D;  n2 = number of hazards >= 2;  m = max(H, A, D)
      Critical : S >= critical_sum  or (m == 3 and n2 >= 2)
      High     : S >= high_sum      or  m == 3 or n2 >= 2
      Moderate : S >= moderate_sum  or  m >= 2
      Low      : otherwise
    A missing hazard counts as 0 for S (its availability is tracked by `label_coverage`);
    the label is undefined when heat (temperature/humidity) is missing.
    """
    c = cfg["compound"]
    H, A, D = (np.nan_to_num(np.asarray(v, float), nan=0.0) for v in (h, a, d))
    S = H + A + D
    stack = np.vstack([H, A, D])
    n2 = (stack >= 2).sum(0)
    m = stack.max(0)
    y = np.zeros(len(S), dtype=float)
    y[(S >= c["moderate_sum"]) | (m >= 2)] = 1
    y[(S >= c["high_sum"]) | (m == 3) | (n2 >= 2)] = 2
    y[(S >= c["critical_sum"]) | ((m == 3) & (n2 >= 2))] = 3
    y[np.isnan(np.asarray(h, float))] = np.nan
    return y


def label_panel(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add hazard scores (from observed values) and the compound label to the site-day panel."""
    out = df.copy()
    obs = out["observed"] == 1
    t, rh = out["temp_c_obs"].values, out["rh_obs"].values
    g = out.groupby("site_id")
    win = cfg["dry_spell_days"]
    wet_obs = (out["precip_mm_obs"] > 0).astype(float).where(out["precip_mm_obs"].notna())
    out["_wet"] = wet_obs
    out["_wet_count30"] = g["_wet"].transform(lambda x: x.rolling(win, min_periods=1).sum())
    out["_n_obs30"] = g["_wet"].transform(lambda x: x.rolling(win, min_periods=1).count())
    h = heat_score(t, rh, cfg)
    a = air_score(out["pm25_obs"].values, out["pm10_obs"].values, out["o3_obs"].values,
                  out["no2_obs"].values, cfg)
    d = dryness_score(t, rh, out["_wet_count30"].values, out["_n_obs30"].values, cfg)
    h[~obs.values], a[~obs.values], d[~obs.values] = np.nan, np.nan, np.nan
    out["heat_score_obs"], out["air_score_obs"], out["dry_score_obs"] = h, a, d
    out["risk_class"] = compound_class(h, a, d, cfg)
    out["label_coverage"] = (np.isfinite(np.vstack([h, a, d]))).mean(0)
    # input-side copies of the scores (computed on forward-filled inputs, used as features)
    out["heat_score"] = heat_score(out["temp_c"].values, out["rh"].values, cfg)
    out["air_score"] = air_score(out["pm25"].values, out["pm10"].values, out["o3"].values, out["no2"].values, cfg)
    out["dry_score"] = dryness_score(out["temp_c"].values, out["rh"].values, out["_wet_count30"].values,
                                     out["_n_obs30"].values, cfg)
    return out.drop(columns=["_wet", "_wet_count30", "_n_obs30"])
