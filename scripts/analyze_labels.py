"""Label validity: sensitivity of the compound-risk label to its design choices.

Computed BEFORE model selection and independent of any model output.  Writes
results/analysis/label_sensitivity.json and figures under results/figures/labels/.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score

from climatesense.evaluation.plotting import RISK_COLORS, SERIES, save, setup
from climatesense.labeling.compound_risk import CLASS_NAMES, air_score, compound_class, dryness_score, heat_score
from climatesense.utils import atomic_write_json, load_yaml

FIG = ROOT / "results/figures/labels"


def relabel(df, cfg):
    h = heat_score(df.temp_c_obs.values, df.rh_obs.values, cfg)
    a = air_score(df.pm25_obs.values, df.pm10_obs.values, df.o3_obs.values, df.no2_obs.values, cfg)
    g = df.groupby("site_id")
    wet = (df.precip_mm_obs > 0).astype(float).where(df.precip_mm_obs.notna())
    cnt = wet.groupby(df.site_id).transform(lambda x: x.rolling(cfg["dry_spell_days"], min_periods=1).sum()).values
    nob = wet.groupby(df.site_id).transform(lambda x: x.rolling(cfg["dry_spell_days"], min_periods=1).count()).values
    d = dryness_score(df.temp_c_obs.values, df.rh_obs.values, cnt, nob, cfg)
    return h, a, d


def main():
    setup()
    base_cfg = load_yaml("preprocessing")["labels"]
    panel = pd.read_parquet(ROOT / "data/processed/panel.parquet")
    df = panel[panel.observed == 1].copy()
    y0 = df.risk_class.values.astype(int)
    h, a, d = df.heat_score_obs.values, df.air_score_obs.values, df.dry_score_obs.values
    out = {"n_site_days": int(len(df)),
           "base_distribution": {CLASS_NAMES[k]: float(np.mean(y0 == k)) for k in range(4)}}

    variants = {}

    def add(name, y):
        y = np.asarray(y).astype(int)
        variants[name] = {"distribution": {CLASS_NAMES[k]: float(np.mean(y == k)) for k in range(4)},
                          "agreement_with_base": float(np.mean(y == y0)),
                          "quadratic_kappa_vs_base": float(cohen_kappa_score(y0, y, weights="quadratic"))}

    for delta in (-1, 1):
        c = copy.deepcopy(base_cfg)
        for k in ("critical_sum", "high_sum", "moderate_sum"):
            c["compound"][k] += delta
        add(f"class_boundaries_{'+' if delta > 0 else ''}{delta}", compound_class(h, a, d, c))
    add("without_dryness", compound_class(h, a, np.zeros_like(d), base_cfg))
    add("without_air", compound_class(h, np.zeros_like(a), d, base_cfg))
    # US-EPA index (provider-computed) instead of WHO-based air score: map 1-2->0,3->1,4->2,5-6->3
    epa = df.us_epa_index_obs.map({1: 0, 2: 0, 3: 1, 4: 2, 5: 3, 6: 3}).values.astype(float)
    add("air_from_us_epa_index", compound_class(h, epa, d, base_cfg))
    # stricter / looser heat thresholds (+/-1 C on all heat-index bands)
    for dt in (-1.0, 1.0):
        c = copy.deepcopy(base_cfg)
        c["heat_index_c"] = [v + dt for v in c["heat_index_c"]]
        hh, _, _ = relabel(df, c)
        add(f"heat_index_bands_{dt:+.0f}C", compound_class(hh, a, d, c))
    c = copy.deepcopy(base_cfg)
    c["vpd_kpa"] = [v * 1.25 for v in c["vpd_kpa"]]
    _, _, dd = relabel(df, c)
    add("vpd_bands_x1.25", compound_class(h, a, dd, c))
    out["variants"] = variants

    # agreement between WHO-based air score and the provider's US-EPA index
    ok = ~np.isnan(a) & ~np.isnan(epa)
    out["air_score_vs_us_epa_index"] = {"exact_agreement": float(np.mean(a[ok] == epa[ok])),
                                        "spearman": float(pd.Series(a[ok]).corr(pd.Series(epa[ok]), method="spearman"))}
    # temporal stability: class shares per quarter
    q = df.assign(q=df.date.dt.to_period("Q").astype(str)).groupby("q").risk_class.value_counts(normalize=True).unstack().fillna(0)
    out["quarterly_class_share"] = {str(i): {CLASS_NAMES[int(k)]: float(v) for k, v in r.items()} for i, r in q.iterrows()}
    # day-to-day persistence of the label (autocorrelation that a forecaster can exploit)
    nxt = df.groupby("site_id").risk_class.shift(-1)
    same = (df.date.groupby(df.site_id).shift(-1) - df.date).dt.days == 1
    out["next_day_label_persistence"] = float(np.mean(df.risk_class[same] == nxt[same]))
    # which hazard drives High/Critical days
    hc = y0 >= 2
    out["drivers_high_critical"] = {"heat_ge2": float(np.mean(np.nan_to_num(h[hc]) >= 2)),
                                    "air_ge2": float(np.mean(np.nan_to_num(a[hc]) >= 2)),
                                    "dryness_ge2": float(np.mean(np.nan_to_num(d[hc]) >= 2))}
    # regional plausibility: top/bottom sites by High+Critical share
    # sites are keyed by site_id; the provider occasionally renames a location for a single day,
    # so the representative name is the modal one
    name = df.groupby("site_id").agg(location=("location_name", lambda s: s.mode().iloc[0]),
                                     country=("country", lambda s: s.mode().iloc[0]))
    hc_share = df.groupby("site_id").risk_class.apply(lambda s: float((s >= 2).mean()))
    share = pd.Series(hc_share.values, index=pd.MultiIndex.from_arrays(
        [hc_share.index, name.loc[hc_share.index, "location"], name.loc[hc_share.index, "country"]])).sort_values()
    out["sites_highest_hc_share"] = [dict(zip(["site_id", "location", "country"], k), share=v) for k, v in share.tail(10)[::-1].items()]
    out["sites_lowest_hc_share"] = [dict(zip(["site_id", "location", "country"], k), share=v) for k, v in share.head(10).items()]
    # sampling-time bias: the provider snapshot is taken at ~the same UTC instant for all sites,
    # so the local observation hour (and hence the diurnal phase of temperature) depends on longitude
    sites = pd.read_csv(ROOT / "data/processed/sites.csv").set_index("site_id")
    lonbin = pd.cut(df.site_id.map(sites.lon), [-180, -90, -30, 30, 90, 180])
    hb = df.groupby(lonbin, observed=True).agg(median_local_hour=("obs_hour_obs", "median"),
                                               share_heat_ge2=("heat_score_obs", lambda v: float((v >= 2).mean())),
                                               share_high_critical=("risk_class", lambda v: float((v >= 2).mean())),
                                               n=("risk_class", "size"))
    out["observation_hour_bias_by_longitude"] = {str(k): {c: float(v) for c, v in r.items()} for k, r in hb.iterrows()}
    atomic_write_json(out, ROOT / "results/analysis/label_sensitivity.json")

    # figure: distribution under variants
    names = ["base"] + list(variants)
    dist = np.array([[out["base_distribution"][c] for c in CLASS_NAMES]] +
                    [[variants[n]["distribution"][c] for c in CLASS_NAMES] for n in variants])
    fig, ax = plt.subplots(figsize=(8, 3.6))
    left = np.zeros(len(names))
    for k in range(4):
        ax.barh(names, dist[:, k], left=left, color=RISK_COLORS[k], label=CLASS_NAMES[k], edgecolor="white", linewidth=1)
        left += dist[:, k]
    for i, n in enumerate(names[1:], 1):
        ax.text(1.01, i, f"κw={variants[n]['quadratic_kappa_vs_base']:.2f}", va="center", fontsize=7)
    ax.set_xlim(0, 1.12); ax.invert_yaxis(); ax.set_xlabel("Share of site-days"); ax.grid(axis="y", visible=False)
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.45, -0.32))
    ax.set_title("Label sensitivity: class distribution under alternative definitions")
    save(fig, FIG / "label_sensitivity.png")

    # figure: hazard score composition
    fig, axs = plt.subplots(1, 3, figsize=(9, 2.6), sharey=True)
    for ax, v, nm, c in zip(axs, [h, a, d], ["Heat", "Air quality", "Dryness (proxy)"], SERIES[:3]):
        cnt = pd.Series(v).value_counts(normalize=True).sort_index()
        ax.bar(cnt.index.astype(int).astype(str), cnt.values, color=c)
        ax.set_title(f"{nm} hazard score"); ax.set_xlabel("Score")
    axs[0].set_ylabel("Share of site-days")
    save(fig, FIG / "hazard_scores.png")
    print({k: v["quadratic_kappa_vs_base"] for k, v in variants.items()})
    print(out["air_score_vs_us_epa_index"], out["next_day_label_persistence"], out["drivers_high_critical"])


if __name__ == "__main__":
    main()
