"""Load the Kaggle sources and build the harmonised site-day panel.

The World Weather Repository provides one *snapshot* observation per capital-city
location per day.  A "site" is defined as (country, lat, lon rounded to 0.1 deg) so that
cosmetic renames of the same location (e.g. Rangoon / Yangon) are merged, while
genuinely different locations that the provider substituted over time are not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import ROOT, haversine_km, load_yaml

WEATHER_NUMERIC = [
    "temperature_celsius", "feels_like_celsius", "humidity", "wind_kph", "gust_kph",
    "pressure_mb", "precip_mm", "cloud", "visibility_km", "uv_index",
    "air_quality_Carbon_Monoxide", "air_quality_Ozone", "air_quality_Nitrogen_dioxide",
    "air_quality_Sulphur_dioxide", "air_quality_PM2.5", "air_quality_PM10",
    "air_quality_us-epa-index", "air_quality_gb-defra-index",
]
RENAME = {
    "temperature_celsius": "temp_c", "feels_like_celsius": "feels_like_c", "humidity": "rh",
    "pressure_mb": "pressure_hpa", "air_quality_Carbon_Monoxide": "co",
    "air_quality_Ozone": "o3", "air_quality_Nitrogen_dioxide": "no2",
    "air_quality_Sulphur_dioxide": "so2", "air_quality_PM2.5": "pm25", "air_quality_PM10": "pm10",
    "air_quality_us-epa-index": "us_epa_index", "air_quality_gb-defra-index": "gb_defra_index",
}
# physically plausible ranges; values outside are treated as sensor/provider errors (set NaN)
VALID_RANGE = {
    "temp_c": (-60, 60), "feels_like_c": (-80, 75), "rh": (0, 100), "wind_kph": (0, 300),
    "gust_kph": (0, 400), "pressure_hpa": (850, 1090), "precip_mm": (0, 500), "cloud": (0, 100),
    "visibility_km": (0, 100), "uv_index": (0, 20), "co": (0, 1e5), "o3": (0, 1000),
    "no2": (0, 2000), "so2": (0, 3000), "pm25": (0, 3000), "pm10": (0, 8000),
    "us_epa_index": (1, 6), "gb_defra_index": (1, 10),
}


def load_weather_raw(path=None) -> pd.DataFrame:
    cfg = load_yaml("data")
    path = path or ROOT / cfg["paths"]["weather_csv"]
    return pd.read_csv(path)


def clean_weather(raw: pd.DataFrame, round_deg: float = 0.1) -> tuple[pd.DataFrame, dict]:
    """Normalise schema, replace sentinels/implausible values with NaN, aggregate to site-day."""
    audit = {"raw_rows": len(raw)}
    df = raw[["country", "location_name", "latitude", "longitude", "last_updated"] + WEATHER_NUMERIC].copy()
    df = df.rename(columns=RENAME)
    df["date"] = pd.to_datetime(df["last_updated"]).dt.normalize()  # local calendar date
    df["obs_hour"] = pd.to_datetime(df["last_updated"]).dt.hour + pd.to_datetime(df["last_updated"]).dt.minute / 60
    n_invalid = {}
    for c, (lo, hi) in VALID_RANGE.items():
        bad = (df[c] <= -9999) | (df[c] < lo) | (df[c] > hi)
        n_invalid[c] = int(bad.sum())
        df.loc[bad, c] = np.nan
    audit["invalid_values_set_nan"] = n_invalid
    df["site_lat"] = (df["latitude"] / round_deg).round() * round_deg
    df["site_lon"] = (df["longitude"] / round_deg).round() * round_deg
    df["site_id"] = (df["country"].str.strip() + "|" + df["site_lat"].round(1).astype(str) + "|"
                     + df["site_lon"].round(1).astype(str))
    audit["exact_duplicate_rows"] = int(df.duplicated(["site_id", "last_updated"]).sum())
    df = df.drop_duplicates(["site_id", "last_updated"])
    num = list(RENAME.values()) + ["wind_kph", "gust_kph", "precip_mm", "cloud", "visibility_km", "uv_index", "obs_hour"]
    num = list(dict.fromkeys(num))
    agg = {c: "mean" for c in num}
    agg.update({"location_name": "first", "country": "first", "latitude": "mean", "longitude": "mean"})
    before = len(df)
    day = df.groupby(["site_id", "date"], as_index=False).agg(agg)
    audit["same_day_records_averaged"] = int(before - len(day))
    # integer AQ indices: averaging two same-day records can give .5 -> keep as ceiling (conservative)
    for c in ["us_epa_index", "gb_defra_index"]:
        day[c] = np.ceil(day[c])
    return day, audit


def select_sites(day: pd.DataFrame, min_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats = day.groupby("site_id").agg(
        country=("country", "first"), location=("location_name", "first"),
        lat=("latitude", "mean"), lon=("longitude", "mean"), n_days=("date", "nunique"),
        first=("date", "min"), last=("date", "max"))
    stats["span_days"] = (stats["last"] - stats["first"]).dt.days + 1
    stats["kept"] = stats["n_days"] >= min_days
    return day[day.site_id.isin(stats.index[stats.kept])].copy(), stats.reset_index()


def to_daily_grid(day: pd.DataFrame) -> pd.DataFrame:
    """Reindex every site to a continuous daily calendar; unobserved days get observed=0."""
    out = []
    for sid, g in day.groupby("site_id"):
        g = g.set_index("date").sort_index()
        idx = pd.date_range(g.index.min(), g.index.max(), freq="D")
        g = g.reindex(idx)
        g["observed"] = g["temp_c"].notna().astype(np.int8)
        g["site_id"] = sid
        for c in ["country", "location_name"]:
            g[c] = g[c].ffill().bfill()
        out.append(g.rename_axis("date").reset_index())
    return pd.concat(out, ignore_index=True)


# ----------------------------------------------------------------------------- spatial joins

def load_world_cities(path=None) -> pd.DataFrame:
    cfg = load_yaml("data")
    return pd.read_csv(path or ROOT / cfg["paths"]["world_cities_csv"])


def join_world_cities(sites: pd.DataFrame, wc: pd.DataFrame, max_km: float) -> pd.DataFrame:
    """Nearest World Cities entry within max_km (coordinates first; names only as a check)."""
    d = haversine_km(sites["lat"].values[:, None], sites["lon"].values[:, None],
                     wc["lat"].values[None, :], wc["lng"].values[None, :])
    j = d.argmin(1)
    dist = d[np.arange(len(sites)), j]
    m = wc.iloc[j].reset_index(drop=True)
    res = sites.reset_index(drop=True).copy()
    res["wc_city"] = m["city_ascii"].values
    res["wc_country"] = m["country"].values
    res["iso3"] = m["iso3"].values
    res["wc_capital"] = m["capital"].values
    res["population"] = m["population"].values
    res["wc_distance_km"] = dist
    res["wc_matched"] = dist <= max_km
    res.loc[~res.wc_matched, ["wc_city", "wc_country", "iso3", "wc_capital", "population"]] = np.nan
    res["wc_country_agrees"] = res["wc_country"].str.lower().str[:5] == res["country"].str.lower().str[:5]
    # ambiguity: second-nearest candidate almost as close as the nearest
    d_sorted = np.sort(d, axis=1)
    res["wc_ambiguous"] = (d_sorted[:, 1] - d_sorted[:, 0] < 2.0) & res["wc_matched"]
    return res


def _parse_coord(s: pd.Series) -> pd.Series:
    v = s.str[:-1].astype(float)
    return v.where(s.str[-1].isin(["N", "E"]), -v)


def berkeley_climatology(path=None, normal=(1961, 1990), recent=(2000, 2012)) -> pd.DataFrame:
    """Per Berkeley city: monthly normals for the reference period and recent-period warming."""
    cfg = load_yaml("data")
    df = pd.read_csv(path or ROOT / cfg["paths"]["berkeley_city_csv"],
                     usecols=["dt", "AverageTemperature", "City", "Country", "Latitude", "Longitude"])
    df = df.dropna(subset=["AverageTemperature"])
    df["year"] = df["dt"].str[:4].astype(int)
    df["month"] = df["dt"].str[5:7].astype(int)
    df["key"] = df["City"] + "|" + df["Country"] + "|" + df["Latitude"] + "|" + df["Longitude"]
    ref = df[df.year.between(*normal)]
    normals = ref.pivot_table(index="key", columns="month", values="AverageTemperature", aggfunc="mean")
    counts = ref.groupby("key").size()
    normals = normals[(counts.reindex(normals.index) >= 0.8 * 12 * (normal[1] - normal[0] + 1)).values]
    normals = normals.dropna()
    normals.columns = [f"bk_normal_m{m:02d}" for m in normals.columns]
    rec = df[df.year.between(*recent)].groupby("key")["AverageTemperature"].mean()
    refm = ref.groupby("key")["AverageTemperature"].mean()
    out = normals.copy()
    out["bk_normal_annual"] = normals.mean(1)
    out["bk_seasonal_amplitude"] = normals.max(1) - normals.min(1)
    out["bk_warming_recent_minus_normal"] = (rec - refm).reindex(out.index)
    meta = df.drop_duplicates("key").set_index("key")
    out["bk_city"] = meta.loc[out.index, "City"].values
    out["bk_country"] = meta.loc[out.index, "Country"].values
    out["bk_lat"] = _parse_coord(meta.loc[out.index, "Latitude"]).values
    out["bk_lon"] = _parse_coord(meta.loc[out.index, "Longitude"]).values
    return out.reset_index(drop=True)


def join_berkeley(sites: pd.DataFrame, bk: pd.DataFrame, max_km: float) -> pd.DataFrame:
    d = haversine_km(sites["lat"].values[:, None], sites["lon"].values[:, None],
                     bk["bk_lat"].values[None, :], bk["bk_lon"].values[None, :])
    j = d.argmin(1)
    dist = d[np.arange(len(sites)), j]
    m = bk.iloc[j].reset_index(drop=True)
    res = sites.reset_index(drop=True).copy()
    for c in bk.columns:
        res[c] = m[c].values
    res["bk_distance_km"] = dist
    res["bk_matched"] = dist <= max_km
    bk_cols = [c for c in bk.columns]
    res.loc[~res.bk_matched, bk_cols] = np.nan
    return res
