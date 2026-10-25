"""Physically based and temporal features.  Every temporal feature uses only past/current days."""
from __future__ import annotations

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------- thermodynamics

def sat_vapour_pressure_kpa(t_c):
    """Tetens/Magnus saturation vapour pressure over water (kPa)."""
    return 0.6108 * np.exp(17.27 * t_c / (t_c + 237.3))


def dewpoint_c(t_c, rh):
    a, b = 17.27, 237.3
    rh = np.clip(rh, 1e-3, 100)
    g = np.log(rh / 100.0) + a * t_c / (b + t_c)
    return b * g / (a - g)


def vpd_kpa(t_c, rh):
    """Vapour-pressure deficit: atmospheric evaporative demand (aridity) indicator."""
    return sat_vapour_pressure_kpa(t_c) * (1 - np.clip(rh, 0, 100) / 100.0)


def heat_index_c(t_c, rh):
    """US NWS heat index (Rothfusz regression with Steadman fallback and NWS adjustments).

    The regression is only applied where the simple Steadman estimate averaged with T is >= 80 F,
    exactly as in the NWS algorithm; below that range the simple formula is returned, so cold
    conditions never receive spurious regression values.
    """
    t = np.asarray(t_c, float) * 9 / 5 + 32
    r = np.asarray(rh, float)
    simple = 0.5 * (t + 61.0 + (t - 68.0) * 1.2 + r * 0.094)
    hi = (-42.379 + 2.04901523 * t + 10.14333127 * r - 0.22475541 * t * r - 6.83783e-3 * t * t
          - 5.481717e-2 * r * r + 1.22874e-3 * t * t * r + 8.5282e-4 * t * r * r - 1.99e-6 * t * t * r * r)
    with np.errstate(invalid="ignore"):
        adj1 = ((13 - r) / 4) * np.sqrt(np.clip((17 - np.abs(t - 95.0)) / 17, 0, None))
    hi = np.where((r < 13) & (t >= 80) & (t <= 112), hi - adj1, hi)
    adj2 = ((r - 85) / 10) * ((87 - t) / 5)
    hi = np.where((r > 85) & (t >= 80) & (t <= 87), hi + adj2, hi)
    use_reg = (simple + t) / 2 >= 80
    out = np.where(use_reg, hi, simple)
    out = np.where(np.isnan(t) | np.isnan(r), np.nan, out)
    return (out - 32) * 5 / 9


def wet_bulb_stull_c(t_c, rh):
    """Stull (2011) empirical wet-bulb temperature at ~1013 hPa; valid RH 5-99 %, T -20..50 C."""
    t = np.asarray(t_c, float)
    r = np.clip(np.asarray(rh, float), 5, 99)
    return (t * np.arctan(0.151977 * np.sqrt(r + 8.313659)) + np.arctan(t + r) - np.arctan(r - 1.676331)
            + 0.00391838 * r ** 1.5 * np.arctan(0.023101 * r) - 4.686035)


# ----------------------------------------------------------------------------- feature groups

CLIMATE_FEATURES = [
    "temp_c", "feels_like_c", "rh", "dewpoint_c", "heat_index_c", "wet_bulb_c", "wind_kph",
    "gust_kph", "pressure_hpa", "cloud", "visibility_km", "uv_index", "temp_anom_bk",
    "obs_hour", "doy_sin", "doy_cos", "heat_score",
]
AIR_FEATURES = [
    "log_co", "log_o3", "log_no2", "log_so2", "log_pm25", "log_pm10", "us_epa_index",
    "gb_defra_index", "air_score",
]
DRY_FEATURES = [
    "vpd_kpa", "log_precip", "wet_flag", "precip_sum_7d", "precip_sum_30d", "wet_frac_7d",
    "wet_frac_30d", "days_since_wet", "dry_score",
]
CONTEXT_FEATURES = [
    "lat_scaled", "abs_lat_scaled", "lon_sin", "lon_cos", "log_population", "bk_normal_annual",
    "bk_seasonal_amplitude", "bk_warming_recent_minus_normal",
]
HUMIDITY_DERIVED = ["feels_like_c", "rh", "dewpoint_c", "heat_index_c", "wet_bulb_c", "heat_score",
                    "vpd_kpa", "dry_score"]
MODALITIES = {"climate": CLIMATE_FEATURES, "air": AIR_FEATURES, "dryness": DRY_FEATURES}


def add_daily_features(panel: pd.DataFrame, sites: pd.DataFrame, ffill_max: int = 3) -> pd.DataFrame:
    """Derive physical features on the site-day grid.

    Raw observations are forward-filled for at most `ffill_max` days (past information only) to
    build model *inputs*; columns suffixed `_obs` keep the un-filled values for labelling.
    """
    df = panel.sort_values(["site_id", "date"]).copy()
    raw_cols = ["temp_c", "feels_like_c", "rh", "wind_kph", "gust_kph", "pressure_hpa", "precip_mm",
                "cloud", "visibility_km", "uv_index", "co", "o3", "no2", "so2", "pm25", "pm10",
                "us_epa_index", "gb_defra_index", "obs_hour"]
    for c in raw_cols:
        df[c + "_obs"] = df[c]
    df[raw_cols] = df.groupby("site_id")[raw_cols].ffill(limit=ffill_max)
    df["filled"] = (df["observed"] == 0) & df["temp_c"].notna()

    df["dewpoint_c"] = dewpoint_c(df.temp_c, df.rh)
    df["heat_index_c"] = heat_index_c(df.temp_c, df.rh)
    df["wet_bulb_c"] = wet_bulb_stull_c(df.temp_c, df.rh)
    df["vpd_kpa"] = vpd_kpa(df.temp_c, df.rh)
    for p in ["co", "o3", "no2", "so2", "pm25", "pm10"]:
        df["log_" + p] = np.log1p(df[p])
    df["log_precip"] = np.log1p(df.precip_mm)

    doy = df.date.dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    s = sites.set_index("site_id")
    month = df.date.dt.month
    normal = np.full(len(df), np.nan)
    for m in range(1, 13):
        col = f"bk_normal_m{m:02d}"
        if col in s:
            sel = (month == m).values
            normal[sel] = df.loc[sel, "site_id"].map(s[col]).values
    df["temp_anom_bk"] = df.temp_c - normal

    # past-only precipitation persistence features (precip_mm is the provider's snapshot value)
    obs_precip = df["precip_mm_obs"]
    df["wet_flag"] = (obs_precip > 0).astype(float).where(obs_precip.notna())
    g = df.groupby("site_id")
    for w in (7, 30):
        df[f"precip_sum_{w}d"] = g["precip_mm_obs"].transform(lambda x: x.rolling(w, min_periods=max(3, w // 3)).sum())
        df[f"wet_frac_{w}d"] = g["wet_flag"].transform(lambda x: x.rolling(w, min_periods=max(3, w // 3)).mean())
    df["precip_sum_7d"] = np.log1p(df["precip_sum_7d"])
    df["precip_sum_30d"] = np.log1p(df["precip_sum_30d"])

    def _days_since(x):
        out, last = np.full(len(x), np.nan), None
        for i, v in enumerate(x.values):
            if v == 1:
                last = i
            out[i] = (i - last) if last is not None else np.nan
        return pd.Series(np.minimum(out, 60), index=x.index)

    df["days_since_wet"] = g["wet_flag"].transform(_days_since)

    # static context
    df["lat_scaled"] = df.site_id.map(s["lat"]) / 90.0
    df["abs_lat_scaled"] = df["lat_scaled"].abs()
    lon = np.radians(df.site_id.map(s["lon"]))
    df["lon_sin"], df["lon_cos"] = np.sin(lon), np.cos(lon)
    df["log_population"] = np.log10(df.site_id.map(s["population"]).astype(float))
    for c in ["bk_normal_annual", "bk_seasonal_amplitude", "bk_warming_recent_minus_normal"]:
        df[c] = df.site_id.map(s[c]).astype(float)
    return df
