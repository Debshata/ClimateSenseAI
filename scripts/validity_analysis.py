"""Validity / robustness checks on FROZEN predictions (run after evaluate_models.py).

1. Observation-hour bias: correlation of local observation hour with labels and predictions;
   performance by longitude band (MAF-Net vs persistence).
2. Air-quality non-stationarity: PM2.5 distributions in the three provider regimes; macro-F1 by
   month over the validation + test periods.
3. Label persistence: performance on stable days vs class transitions, and recall of High/Critical
   *onsets* (today < High, tomorrow >= High), which persistence can never predict.
Nothing here feeds back into model selection.
Outputs: results/analysis/validity_checks.json, results/figures/validity/*.png
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, spearmanr
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score

from climatesense.evaluation.plotting import MUTED, SERIES, save, setup
from climatesense.training.trainer import get_arrays
from climatesense.utils import atomic_write_json, read_json

FIG = ROOT / "results/figures/validity"
PRED = ROOT / "results/predictions"
# boundaries = largest 7-day steps of the cross-site median PM2.5 (2024-09-05: x2.6; 2025-10-01: /1.7)
REGIMES = [("R1: to 2024-09-04", None, "2024-09-04"), ("R2: 2024-09-05 to 2025-09-30", "2024-09-05", "2025-09-30"),
           ("R3: from 2025-10-01", "2025-10-01", None)]


def mf1(y, p):
    return float(f1_score(y, p, average="macro", labels=[0, 1, 2, 3], zero_division=0))


def main():
    setup()
    arr = get_arrays()
    p = arr.panel
    st = read_json(ROOT / "results/metrics/selected_model_test.json")
    sel = st["selected_run"]
    d = np.load(PRED / f"{sel}_test.npz")
    pers = np.load(PRED / "baseline-persistence_L14_h1_test.npz")
    hgb = np.load(PRED / "baseline-hist_gradient_boosting_L14_h1_test.npz")
    assert np.array_equal(d["idx"], pers["idx"]) and np.array_equal(d["idx"], hgb["idx"])
    idx, y = d["idx"], d["y"]
    pm, pp, ph = d["proba"].argmax(1), pers["proba"].argmax(1), hgb["proba"].argmax(1)
    sites = pd.read_csv(ROOT / "data/processed/sites.csv").set_index("site_id")
    tgt = idx + 1
    df = pd.DataFrame({"site": p.site_id.values[tgt], "date": p.date.values[tgt], "y": y, "maf": pm, "pers": pp, "hgb": ph,
                       "y_today": arr.y[idx], "hour": p.obs_hour_obs.values[tgt]})
    df["lon"] = df.site.map(sites.lon)
    out = {"selected_run": sel, "n_test": int(len(df))}

    # ---- 1. observation hour / longitude
    o = p[p.observed == 1]
    rho_lab = spearmanr(o.obs_hour_obs, o.risk_class, nan_policy="omit")
    rho_heat = spearmanr(o.obs_hour_obs, o.heat_score_obs, nan_policy="omit")
    rho_pred = spearmanr(df.hour, df.maf, nan_policy="omit")
    out["obs_hour_correlation"] = {"spearman_hour_vs_label_all_days": float(rho_lab.statistic),
                                   "spearman_hour_vs_heat_score_all_days": float(rho_heat.statistic),
                                   "spearman_hour_vs_predicted_class_test": float(rho_pred.statistic)}
    bins = [-180, -90, -30, 30, 90, 180]
    names = ["Americas W (<-90°)", "Americas E (-90°…-30°)", "Europe/Africa (-30°…30°)", "W/S Asia (30°…90°)", "E Asia/Oceania (>90°)"]
    df["band"] = pd.cut(df.lon, bins, labels=names)
    rows = []
    for b, g in df.groupby("band", observed=True):
        rows.append(dict(band=str(b), n=len(g), sites=g.site.nunique(), median_local_hour=float(g.hour.median()),
                         share_high_critical=float((g.y >= 2).mean()), macro_f1_mafnet=mf1(g.y, g.maf),
                         macro_f1_persistence=mf1(g.y, g.pers), acc_mafnet=float((g.y == g.maf).mean()),
                         acc_persistence=float((g.y == g.pers).mean())))
    lon = pd.DataFrame(rows)
    out["by_longitude_band"] = lon.to_dict(orient="records")

    # ---- 2. PM2.5 regimes and performance over time
    reg_rows, samples = [], {}
    for name, a, b in REGIMES:
        m = pd.Series(True, index=o.index)
        if a:
            m &= o.date >= a
        if b:
            m &= o.date <= b
        v = o.loc[m, "pm25_obs"].dropna()
        samples[name] = v.values
        reg_rows.append(dict(regime=name, n=len(v), median=float(v.median()), p90=float(v.quantile(.9)),
                             share_above_who_aqg=float((v > 15).mean()),
                             share_air_score_ge2=float((o.loc[m, "air_score_obs"] >= 2).mean())))
    ks = {f"{a[:2]} vs {b[:2]}": float(ks_2samp(samples[a], samples[b]).statistic)
          for a, b in [(REGIMES[0][0], REGIMES[1][0]), (REGIMES[1][0], REGIMES[2][0]), (REGIMES[0][0], REGIMES[2][0])]}
    out["pm25_regimes"] = {"regimes": reg_rows, "ks_statistic": ks,
                           "note": "training spans R1-R3 (to 2026-01-21); validation and test lie entirely in R3"}
    v = np.load(PRED / f"{sel}_val.npz")
    vdf = pd.DataFrame({"date": p.date.values[v["idx"] + 1], "y": v["y"], "maf": v["proba"].argmax(1),
                        "pers": arr.y[v["idx"]]})
    vdf["pers"] = vdf.pers.fillna(0).astype(int)
    allp = pd.concat([vdf.assign(split="val"), df[["date", "y", "maf", "pers"]].assign(split="test")])
    allp["month"] = pd.to_datetime(allp.date).dt.to_period("M").astype(str)
    monthly = allp.groupby("month").apply(lambda g: pd.Series({"n": len(g), "split": g.split.iloc[0],
                                                               "macro_f1_mafnet": mf1(g.y, g.maf), "macro_f1_persistence": mf1(g.y, g.pers)}))
    out["monthly_val_test"] = monthly.reset_index().to_dict(orient="records")

    # ---- 3. persistence and transitions
    yt = df.y_today
    stable = (yt == df.y)
    trans = ~stable & yt.notna()
    onset = (yt < 2) & (df.y >= 2)
    offset = (yt >= 2) & (df.y < 2)
    def block(mask, label):
        g = df[mask]
        return {"subset": label, "n": int(mask.sum()), "share": float(mask.mean()),
                "acc_mafnet": float((g.y == g.maf).mean()), "acc_hgb": float((g.y == g.hgb).mean()),
                "acc_persistence": float((g.y == g.pers).mean())}
    out["persistence_analysis"] = {
        "test_label_persistence": float(stable.mean()),
        "subsets": [block(stable, "stable (class unchanged)"), block(trans, "transition (class changes)"),
                    block(onset, "onset of High/Critical"), block(offset, "end of High/Critical")],
        "onset_recall_high_or_critical": {"mafnet": float((df[onset].maf >= 2).mean()), "hgb": float((df[onset].hgb >= 2).mean()),
                                          "persistence": float((df[onset].pers >= 2).mean())},
        "false_alarm_rate_hc_on_stable_low_moderate": {
            "mafnet": float((df[(yt < 2) & (df.y < 2)].maf >= 2).mean()),
            "hgb": float((df[(yt < 2) & (df.y < 2)].hgb >= 2).mean()), "persistence": 0.0},
        "balanced_accuracy_on_transitions": {"mafnet": float(balanced_accuracy_score(df[trans].y, df[trans].maf)),
                                             "persistence": float(balanced_accuracy_score(df[trans].y, df[trans].pers))},
    }
    # minority-class recall for High+Critical combined
    hc = df.y >= 2
    out["high_critical_detection"] = {k: float(recall_score(hc, df[k] >= 2)) for k in ["maf", "hgb", "pers"]}
    atomic_write_json(out, ROOT / "results/analysis/validity_checks.json")

    # ---- figures
    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(lon))
    ax.bar(x - 0.2, lon.macro_f1_mafnet, 0.38, color=SERIES[0], label="MAF-Net")
    ax.bar(x + 0.2, lon.macro_f1_persistence, 0.38, color=SERIES[1], label="Persistence")
    for i, r in lon.iterrows():
        ax.text(i, max(r.macro_f1_mafnet, r.macro_f1_persistence) + 0.01, f"{r.median_local_hour:.0f} h\nHC {r.share_high_critical:.0%}",
                ha="center", fontsize=6.5, color=MUTED)
    ax.set_xticks(x, [n.replace(" (", "\n(") for n in lon.band], fontsize=7); ax.set_ylabel("Test macro-F1")
    ax.set_ylim(0, 1.05); ax.legend(loc="upper left", ncol=2)
    ax.set_title("Test macro-F1 by longitude band (labels: median local obs. hour, High/Critical share)")
    save(fig, FIG / "performance_by_longitude.png")

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
    axs[0].boxplot([np.log10(np.clip(samples[r[0]], 0.1, None)) for r in REGIMES], showfliers=False)
    axs[0].set_xticks([1, 2, 3], [r[0].replace(": ", ":\n") for r in REGIMES], fontsize=7)
    axs[0].set_ylabel("log10 PM2.5 (µg/m³)"); axs[0].set_title("PM2.5 by provider regime (all sites)")
    mm = monthly.reset_index()
    axs[1].plot(mm.month, mm.macro_f1_mafnet, marker="o", ms=4, label="MAF-Net")
    axs[1].plot(mm.month, mm.macro_f1_persistence, marker="o", ms=4, label="Persistence")
    first_test = mm[mm.split == "test"].index.min()
    axs[1].axvline(first_test - 0.5, color=MUTED, ls=":", lw=1); axs[1].text(first_test - 0.4, axs[1].get_ylim()[0] + 0.01, "test", color=MUTED, fontsize=7)
    axs[1].set_ylabel("Macro-F1"); axs[1].set_title("Monthly macro-F1 (validation → test)")
    axs[1].tick_params(axis="x", rotation=45, labelsize=7); axs[1].legend()
    save(fig, FIG / "pm25_regimes_and_monthly_performance.png")

    sub = pd.DataFrame(out["persistence_analysis"]["subsets"])
    fig, ax = plt.subplots(figsize=(6.5, 3))
    x = np.arange(len(sub))
    for j, (k, lab) in enumerate([("acc_mafnet", "MAF-Net"), ("acc_hgb", "Gradient boosting"), ("acc_persistence", "Persistence")]):
        ax.bar(x + (j - 1) * 0.27, sub[k], 0.25, color=SERIES[j], label=lab)
    ax.set_xticks(x, [f"{s}\n(n={n:,})" for s, n in zip(sub.subset, sub.n)], fontsize=7)
    ax.set_ylabel("Accuracy"); ax.set_title("Accuracy on stable days vs class transitions (test)"); ax.legend(fontsize=7)
    save(fig, FIG / "transitions_vs_stable.png")
    print(lon.round(3).to_string()); print(pd.DataFrame(reg_rows).round(3).to_string(), ks)
    print(sub.round(3).to_string()); print(out["persistence_analysis"]["onset_recall_high_or_critical"], out["high_critical_detection"])


if __name__ == "__main__":
    main()
