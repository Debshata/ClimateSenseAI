"""Dataset quality audit and exploratory figures for all four Kaggle sources + the labelled panel.

Writes data/manifests/dataset_quality.json and figures under results/figures/eda/.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from climatesense.evaluation.plotting import MUTED, RISK_COLORS, SERIES, save, setup
from climatesense.labeling.compound_risk import CLASS_NAMES
from climatesense.utils import atomic_write_json, load_yaml

FIG = ROOT / "results/figures/eda"


def profile(df: pd.DataFrame, name: str, date_col=None, id_cols=None) -> dict:
    p = {"rows": int(len(df)), "columns": list(map(str, df.columns)),
         "dtypes": {c: str(t) for c, t in df.dtypes.items()},
         "missing_pct": {c: round(float(v) * 100, 3) for c, v in df.isna().mean().items() if v > 0},
         "duplicate_rows": int(df.duplicated().sum())}
    if date_col:
        d = pd.to_datetime(df[date_col], errors="coerce")
        p["date_min"], p["date_max"] = str(d.min().date()), str(d.max().date())
    if id_cols:
        p["duplicate_ids"] = int(df.duplicated(id_cols).sum())
    num = df.select_dtypes("number")
    p["numeric_summary"] = num.describe().T.round(3).to_dict(orient="index")
    print(f"{name}: {p['rows']} rows, {len(p['columns'])} cols")
    return p


def main():
    setup()
    dcfg = load_yaml("data")
    q = {}
    w = pd.read_csv(ROOT / dcfg["paths"]["weather_csv"])
    q["weather"] = profile(w, "weather", "last_updated", ["country", "location_name", "last_updated"])
    q["weather"]["n_countries_raw"] = int(w.country.nunique())
    q["weather"]["n_locations_raw"] = int(w.location_name.nunique())
    q["weather"]["sentinel_minus9999_counts"] = {c: int((w[c] <= -9999).sum()) for c in w.select_dtypes("number") if (w[c] <= -9999).any()}
    wc = pd.read_csv(ROOT / dcfg["paths"]["world_cities_csv"])
    q["world_cities"] = profile(wc, "world_cities", id_cols=["id"])
    bk = pd.read_csv(ROOT / dcfg["paths"]["berkeley_city_csv"])
    q["berkeley_by_city"] = profile(bk, "berkeley", "dt", ["dt", "City", "Country", "Latitude", "Longitude"])
    q["berkeley_by_city"]["n_cities"] = int(bk.groupby(["City", "Country", "Latitude", "Longitude"]).ngroups)
    dr_dir = ROOT / dcfg["paths"]["drought_dir"]
    soil = pd.read_csv(dr_dir / "soil_data.csv")
    q["drought_us_soil"] = profile(soil, "drought soil", id_cols=["fips"])
    dr = {}
    for part in ["train", "validation", "test"]:
        d = pd.read_csv(dr_dir / f"{part}_timeseries/{part}_timeseries.csv", usecols=["fips", "date", "score"])
        dr[part] = {"rows": int(len(d)), "date_min": d.date.min(), "date_max": d.date.max(),
                    "counties": int(d.fips.nunique()), "score_non_null": int(d.score.notna().sum()),
                    "score_hist_rounded": {str(int(k)): int(v) for k, v in d.score.dropna().round().value_counts().sort_index().items()}}
    head = pd.read_csv(dr_dir / "train_timeseries/train_timeseries.csv", nrows=5)
    q["drought_us_timeseries"] = {"columns": list(head.columns), "splits": dr,
                                  "note": "US counties (FIPS); weekly USDM score, daily NASA POWER weather"}
    atomic_write_json(q, ROOT / "data/manifests/dataset_quality.json")

    # ------------------------------------------------------------------ figures
    panel = pd.read_parquet(ROOT / "data/processed/panel.parquet")
    sites = pd.read_csv(ROOT / "data/processed/sites.csv")
    obs = panel[panel.observed == 1]

    fig, axs = plt.subplots(1, 3, figsize=(10, 2.8))
    for ax, col, lab in zip(axs, ["temp_c_obs", "rh_obs", "heat_index_c"], ["Temperature (°C)", "Relative humidity (%)", "Heat index (°C)"]):
        ax.hist(obs[col].dropna(), bins=60, color=SERIES[0], edgecolor="white", linewidth=0.3)
        ax.set_xlabel(lab); ax.set_ylabel("Site-days")
    fig.suptitle("Distribution of thermal variables (observed site-days)", fontsize=10)
    save(fig, FIG / "thermal_distributions.png")

    fig, axs = plt.subplots(1, 4, figsize=(11, 2.8))
    for ax, (col, lab, g) in zip(axs, [("pm25_obs", "PM2.5 (µg/m³)", 15), ("pm10_obs", "PM10 (µg/m³)", 45),
                                       ("o3_obs", "O3 (µg/m³)", 100), ("no2_obs", "NO2 (µg/m³)", 25)]):
        v = obs[col].dropna()
        ax.hist(np.log10(v.clip(lower=0.1)), bins=60, color=SERIES[1], edgecolor="white", linewidth=0.3)
        ax.axvline(np.log10(g), color=MUTED, ls="--", lw=1)
        ax.text(np.log10(g), ax.get_ylim()[1] * 0.9, " WHO AQG", color=MUTED, fontsize=7)
        ax.set_xlabel(f"log10 {lab}")
    axs[0].set_ylabel("Site-days")
    fig.suptitle("Pollutant concentrations (log scale) with WHO 2021 guideline levels", fontsize=10)
    save(fig, FIG / "pollution_distributions.png")

    cols = ["temp_c_obs", "rh_obs", "wind_kph_obs", "pressure_hpa_obs", "precip_mm_obs", "uv_index_obs", "vpd_kpa",
            "heat_index_c", "pm25_obs", "pm10_obs", "o3_obs", "no2_obs", "co_obs", "so2_obs", "risk_class"]
    corr = obs[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    names = [c.replace("_obs", "") for c in cols]
    ax.set_xticks(range(len(cols)), names, rotation=60, ha="right"); ax.set_yticks(range(len(cols)), names)
    ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.1f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(corr.iloc[i, j]) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Spearman ρ")
    ax.set_title("Spearman correlation of observed variables and risk class")
    save(fig, FIG / "correlation_heatmap.png")

    # missingness heatmap: fraction missing per site over time (monthly)
    raw_cols = ["temp_c_obs", "rh_obs", "pm25_obs", "pm10_obs", "o3_obs", "no2_obs", "co_obs", "so2_obs", "precip_mm_obs"]
    miss = panel.assign(month=panel.date.dt.to_period("M"))
    mm = miss.groupby("month")[raw_cols].apply(lambda g: g.isna().mean())
    fig, ax = plt.subplots(figsize=(9, 3))
    im = ax.imshow(mm.T.values, aspect="auto", cmap="Greys", vmin=0, vmax=max(0.05, mm.values.max()))
    ax.set_yticks(range(len(raw_cols)), [c.replace("_obs", "") for c in raw_cols])
    xt = np.arange(0, len(mm), 3)
    ax.set_xticks(xt, [str(mm.index[i]) for i in xt], rotation=45, ha="right"); ax.grid(False)
    fig.colorbar(im, ax=ax, label="Fraction missing")
    ax.set_title("Missingness by variable and month (site-day grid, incl. unobserved days)")
    save(fig, FIG / "missingness_heatmap.png")

    # site map coloured by share of High+Critical days
    share = obs.groupby("site_id").risk_class.apply(lambda s: (s >= 2).mean()).rename("hc")
    s2 = sites.set_index("site_id").join(share)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    sc = ax.scatter(s2.lon, s2.lat, c=s2.hc, cmap="OrRd", s=22, edgecolor="#52514e", linewidth=0.4, vmin=0, vmax=s2.hc.max())
    ax.set_xlim(-180, 180); ax.set_ylim(-60, 75); ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    fig.colorbar(sc, ax=ax, label="Share of High/Critical days")
    ax.set_title(f"{len(s2)} capital-city sites retained (World Weather Repository)")
    save(fig, FIG / "site_map_high_critical_share.png")

    # temporal trends: daily mean temperature and PM2.5 across sites, by hemisphere
    nh = obs[obs.site_id.map(sites.set_index("site_id").lat) >= 0]
    sh = obs[obs.site_id.map(sites.set_index("site_id").lat) < 0]
    fig, axs = plt.subplots(2, 1, figsize=(9, 4.5), sharex=True)
    for d, lab, c in [(nh, "Northern hemisphere", SERIES[0]), (sh, "Southern hemisphere", SERIES[1])]:
        t = d.groupby("date").temp_c_obs.mean().rolling(7, min_periods=1).mean()
        p = d.groupby("date").pm25_obs.median().rolling(7, min_periods=1).mean()
        axs[0].plot(t.index, t.values, color=c, label=lab); axs[1].plot(p.index, p.values, color=c, label=lab)
    axs[0].set_ylabel("Mean temp (°C)"); axs[1].set_ylabel("Median PM2.5 (µg/m³)")
    axs[0].legend(loc="lower left"); axs[0].set_title("Cross-site temporal trends (7-day smoothed)")
    sp = pd.read_json(ROOT / "data/manifests/splits.json", typ="series")
    for ax in axs:
        for k in ["val", "test"]:
            ax.axvline(pd.Timestamp(sp[k][0]), color=MUTED, ls=":", lw=1)
    axs[1].text(pd.Timestamp(sp["val"][0]), axs[1].get_ylim()[1] * 0.9, " val", color=MUTED, fontsize=7)
    axs[1].text(pd.Timestamp(sp["test"][0]), axs[1].get_ylim()[1] * 0.9, " test", color=MUTED, fontsize=7)
    save(fig, FIG / "temporal_trends.png")

    # seasonal distribution of risk classes (by month, NH sites)
    m = obs.assign(month=obs.date.dt.month, hemi=np.where(obs.site_id.map(sites.set_index("site_id").lat) >= 0, "NH", "SH"))
    fig, axs = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
    for ax, hemi in zip(axs, ["NH", "SH"]):
        ct = pd.crosstab(m[m.hemi == hemi].month, m[m.hemi == hemi].risk_class, normalize="index")
        bottom = np.zeros(len(ct))
        for k in range(4):
            v = ct.get(float(k), pd.Series(0, index=ct.index)).values
            ax.bar(ct.index, v, bottom=bottom, color=RISK_COLORS[k], label=CLASS_NAMES[k], edgecolor="white", linewidth=1)
            bottom += v
        ax.set_title(f"{hemi} sites: class share by calendar month"); ax.set_xlabel("Month"); ax.set_xticks(range(1, 13))
    axs[0].set_ylabel("Share of site-days"); axs[1].legend(loc="upper right", bbox_to_anchor=(1.3, 1))
    save(fig, FIG / "seasonal_class_distribution.png")

    # dryness indicators
    fig, axs = plt.subplots(1, 3, figsize=(10, 2.8))
    axs[0].hist(obs.vpd_kpa.dropna(), bins=60, color=SERIES[2], edgecolor="white", linewidth=0.3); axs[0].set_xlabel("VPD (kPa)")
    axs[1].hist(obs.wet_frac_30d.dropna(), bins=30, color=SERIES[2], edgecolor="white", linewidth=0.3); axs[1].set_xlabel("Share of wet days (30 d)")
    axs[2].hist(obs.days_since_wet.dropna(), bins=30, color=SERIES[2], edgecolor="white", linewidth=0.3); axs[2].set_xlabel("Days since recorded precipitation")
    axs[0].set_ylabel("Site-days"); fig.suptitle("Hydro-climatic dryness indicators (proxy modality)", fontsize=10)
    save(fig, FIG / "dryness_indicators.png")

    # class distribution by split
    sp_counts = obs.groupby(["split_by_date", "risk_class"]).size().unstack()
    sp_counts = sp_counts.loc[["train", "val", "test"]]
    fig, ax = plt.subplots(figsize=(6, 3))
    x = np.arange(3)
    for k in range(4):
        ax.bar(x + (k - 1.5) * 0.2, sp_counts[float(k)], width=0.18, color=RISK_COLORS[k], label=CLASS_NAMES[k])
    ax.set_xticks(x, ["Train", "Validation", "Test"]); ax.set_ylabel("Site-days"); ax.legend(ncol=4, loc="upper right")
    ax.set_title("Compound-risk class distribution by chronological split")
    save(fig, FIG / "class_distribution_by_split.png")

    # Berkeley Earth context: recent warming at matched sites
    s3 = sites.dropna(subset=["bk_warming_recent_minus_normal"])
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(s3.bk_warming_recent_minus_normal, bins=25, color=SERIES[6], edgecolor="white")
    ax.set_xlabel("2000–2012 mean minus 1961–1990 normal (°C)"); ax.set_ylabel("Sites")
    ax.set_title("Long-term warming at matched Berkeley Earth city series")
    save(fig, FIG / "berkeley_warming.png")

    # US drought auxiliary: score distribution
    h = pd.Series(dr["train"]["score_hist_rounded"]).astype(int)
    fig, ax = plt.subplots(figsize=(5, 2.8))
    ax.bar(["None", "D0", "D1", "D2", "D3", "D4"][: len(h)], h.values, color=SERIES[3])
    ax.set_ylabel("County-weeks (train 2000–2016)"); ax.set_title("US Drought Monitor categories (auxiliary dataset)")
    save(fig, FIG / "us_drought_score_distribution.png")
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
