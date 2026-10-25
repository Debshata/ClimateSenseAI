"""Explainability for the selected MAF-Net: permutation importance, integrated gradients,
attention by class, city case studies and risk timelines (held-out test period).

Run after evaluate_models.py.  Outputs results/analysis/*.json and results/figures/explain/*.png
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from climatesense.evaluation.plotting import MUTED, RISK_COLORS, SERIES, save, setup
from climatesense.explainability.attribution import integrated_gradients, permutation_importance
from climatesense.features.engineering import AIR_FEATURES, CLIMATE_FEATURES, DRY_FEATURES, HUMIDITY_DERIVED
from climatesense.labeling.compound_risk import CLASS_NAMES
from climatesense.training.trainer import feature_sets, load_trained
from climatesense.utils import atomic_write_json, read_json

FIG = ROOT / "results/figures/explain"
OUT = ROOT / "results/analysis"


def main():
    setup()
    sel = read_json(ROOT / "results/metrics/selected_model_test.json")["selected_run"]
    rec = read_json(ROOT / "experiments" / sel.split("-")[0] / f"{sel}.json")
    model, batcher, arr = load_trained(rec)
    d = np.load(ROOT / "results/predictions" / f"{sel}_test.npz")
    idx, P, Y, A = d["idx"], d["proba"], d["y"], d["att"]
    names = [arr.feature_names[c] for c in sorted({i for v in feature_sets(rec["config"], arr).values() for i in v})]
    rng = np.random.default_rng(0)
    sub = np.sort(rng.choice(len(idx), min(8000, len(idx)), replace=False))
    t_idx = torch.as_tensor(idx[sub], device=batcher.X.device)
    X, C, y = batcher.get(t_idx)

    # ---- permutation importance: single features and modality groups
    feats = {n: [i] for i, n in enumerate(names) if not n.endswith("_missing")}
    groups = {
        "Climate (all)": [names.index(n) for n in CLIMATE_FEATURES if n in names],
        "Air quality (all)": [names.index(n) for n in AIR_FEATURES if n in names],
        "Dryness proxy (all)": [names.index(n) for n in DRY_FEATURES if n in names],
        "Humidity-derived": [names.index(n) for n in HUMIDITY_DERIVED if n in names],
    }
    base, imp_f = permutation_importance(model, X, C, y, feats, repeats=2)
    _, imp_g = permutation_importance(model, X, C, y, groups, repeats=3)
    # context permutation
    Cp = C[torch.randperm(len(C), generator=torch.Generator().manual_seed(0)).to(C.device)]
    from sklearn.metrics import f1_score
    with torch.no_grad():
        f_ctx = f1_score(y.cpu().numpy(), torch.softmax(model(X, Cp)[0], -1).argmax(1).cpu().numpy(), average="macro")
    imp_g["Static context"] = {"mean_drop_macro_f1": float(base - f_ctx), "std": 0.0}

    # ---- integrated gradients for samples predicted High/Critical
    pred_sub = P[sub].argmax(1)
    hc = np.where(pred_sub >= 2)[0][:1500]
    ig = integrated_gradients(model, X[hc], C[hc], torch.as_tensor(pred_sub[hc], device=X.device)).cpu().numpy()
    ig_mean = pd.Series(ig.mean(0), index=names).drop([n for n in names if n.endswith("_missing")]).sort_values(ascending=False)

    # ---- attention by predicted & true class
    att = {}
    mod_names = ["climate", "air", "dryness"] + (["context"] if rec["config"]["use_context"] else [])
    mod_names = [m for m in mod_names if m in rec["config"]["modalities"] or m == "context"]
    if A.size:
        for k in range(4):
            att[CLASS_NAMES[k]] = {"by_pred": A[P.argmax(1) == k].mean(0).tolist(), "by_true": A[Y == k].mean(0).tolist(),
                                   "n_pred": int((P.argmax(1) == k).sum())}

    res = {"selected_run": sel, "subset_size": int(len(sub)), "base_macro_f1_subset": base,
           "permutation_importance_features": dict(sorted(imp_f.items(), key=lambda kv: -kv[1]["mean_drop_macro_f1"])),
           "permutation_importance_groups": imp_g,
           "integrated_gradients_high_critical_mean_abs": ig_mean.round(5).to_dict(),
           "attention_modalities": mod_names, "attention_by_class": att,
           "note": "Attention weights and attributions describe model behaviour, not causal effects."}

    # ---- case studies (real test samples predicted High/Critical)
    p = arr.panel
    rows = []
    pred = P.argmax(1)
    cand = np.where(pred >= 2)[0]
    correct = [i for i in cand if Y[i] == pred[i]]
    wrong = [i for i in cand if Y[i] != pred[i]]
    pick, seen = [], set()
    for pool, n in [(correct, 4), (wrong, 2)]:
        for i in rng.permutation(pool):
            s = arr.site_codes[idx[i]]
            if s in seen:
                continue
            pick.append(i); seen.add(s)
            if len([j for j in pick if (j in correct) == (pool is correct)]) >= n:
                break
    for i in pick:
        t = idx[i]
        r_t, r_y = p.iloc[t], p.iloc[t + 1]
        rows.append({
            "target_date": str(r_y.date.date()), "last_input_date": str(r_t.date.date()),
            "city": r_y.location_name, "country": r_y.country,
            "actual": CLASS_NAMES[int(Y[i])], "predicted": CLASS_NAMES[int(pred[i])], "correct": bool(Y[i] == pred[i]),
            "probabilities": {CLASS_NAMES[k]: round(float(P[i, k]), 3) for k in range(4)},
            "signals_last_input_day": {"temp_c": round(float(r_t.temp_c), 1), "rh_pct": round(float(r_t.rh), 0),
                                       "heat_index_c": round(float(r_t.heat_index_c), 1), "pm25": round(float(r_t.pm25), 1),
                                       "pm10": round(float(r_t.pm10), 1), "o3": round(float(r_t.o3), 1),
                                       "vpd_kpa": round(float(r_t.vpd_kpa), 2), "days_since_wet": None if np.isnan(r_t.days_since_wet) else int(r_t.days_since_wet),
                                       "hazards_today(H,A,D)": [None if np.isnan(v) else int(v) for v in (r_t.heat_score_obs, r_t.air_score_obs, r_t.dry_score_obs)],
                                       "class_today": None if np.isnan(r_t.risk_class) else CLASS_NAMES[int(r_t.risk_class)]},
            "hazards_target_day(H,A,D)": [None if np.isnan(v) else int(v) for v in (r_y.heat_score_obs, r_y.air_score_obs, r_y.dry_score_obs)],
            "attention": {m: round(float(w), 3) for m, w in zip(mod_names, A[i])} if A.size else None,
        })
    res["case_studies"] = rows
    atomic_write_json(res, OUT / "explainability.json")

    # ---- figures
    top = list(res["permutation_importance_features"].items())[:15]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh([k for k, _ in top][::-1], [v["mean_drop_macro_f1"] for _, v in top][::-1],
            xerr=[v["std"] for _, v in top][::-1], color=SERIES[0], error_kw={"ecolor": MUTED, "lw": 1})
    ax.set_xlabel("Drop in macro-F1 when permuted"); ax.set_title("Permutation importance (selected MAF-Net, test subset)")
    ax.grid(axis="y", visible=False)
    save(fig, FIG / "permutation_importance_features.png")

    fig, ax = plt.subplots(figsize=(5.5, 2.6))
    gi = sorted(imp_g.items(), key=lambda kv: kv[1]["mean_drop_macro_f1"])
    ax.barh([k for k, _ in gi], [v["mean_drop_macro_f1"] for _, v in gi], xerr=[v["std"] for _, v in gi], color=SERIES[1],
            error_kw={"ecolor": MUTED, "lw": 1})
    ax.set_xlabel("Drop in macro-F1 when permuted"); ax.set_title("Modality-group permutation importance"); ax.grid(axis="y", visible=False)
    save(fig, FIG / "permutation_importance_groups.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    t15 = ig_mean.head(15)
    ax.barh(t15.index[::-1], t15.values[::-1], color=SERIES[2])
    ax.set_xlabel("Mean |integrated gradient| (summed over 14 days)")
    ax.set_title("Integrated gradients for predicted High/Critical"); ax.grid(axis="y", visible=False)
    save(fig, FIG / "integrated_gradients_high_critical.png")

    if att:
        fig, ax = plt.subplots(figsize=(6, 3))
        M = np.array([att[c]["by_pred"] for c in CLASS_NAMES])
        x = np.arange(4)
        for j, m in enumerate(mod_names):
            ax.bar(x + (j - (len(mod_names) - 1) / 2) * 0.2, M[:, j], width=0.18, color=SERIES[j], label=m)
        ax.set_xticks(x, [f"{c}\n(n={att[c]['n_pred']})" for c in CLASS_NAMES]); ax.set_ylabel("Mean attention weight")
        ax.set_title("Modality attention by predicted class (test)"); ax.legend(ncol=len(mod_names), loc="upper center", bbox_to_anchor=(0.5, -0.25))
        save(fig, FIG / "attention_by_class.png")

    # risk timelines for 3 sites with the most High/Critical test days
    dfp = pd.DataFrame({"site": arr.site_codes[idx], "date": p.date.values[idx + 1], "y": Y, "pred": pred,
                        "p_hc": P[:, 2] + P[:, 3]})
    top_sites = dfp.groupby("site").y.apply(lambda s: (s >= 2).mean()).sort_values(ascending=False).index[[0, 5, 15]]
    fig, axs = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
    for ax, s in zip(axs, top_sites):
        g = dfp[dfp.site == s].sort_values("date")
        ax.step(g.date, g.y, where="mid", color=MUTED, lw=1.5, label="Observed class")
        ax.scatter(g.date, g.pred, s=8, color=SERIES[1], label="Predicted class (t+1)", zorder=3)
        ax2 = ax.twinx()
        ax2.fill_between(g.date, 0, g.p_hc, color=SERIES[0], alpha=0.15, step="mid", label="P(High or Critical)")
        ax2.set_ylim(0, 1); ax2.set_ylabel("P(H/C)", color=SERIES[0]); ax2.grid(False)
        ax.set_yticks(range(4), CLASS_NAMES); ax.set_ylim(-0.4, 3.4)
        site_id = arr.site_index[s]
        ax.set_title(f"{p.loc[p.site_id == site_id, 'location_name'].iloc[0]}, {p.loc[p.site_id == site_id, 'country'].iloc[0]}", fontsize=9)
    axs[0].legend(loc="upper left", ncol=2, fontsize=7)
    fig.suptitle("One-day-ahead risk timelines on the held-out test period", fontsize=10)
    save(fig, FIG / "risk_timelines.png")
    print("base", base, "groups", {k: round(v["mean_drop_macro_f1"], 4) for k, v in imp_g.items()})
    print(pd.DataFrame(rows)[["city", "target_date", "actual", "predicted"]])


if __name__ == "__main__":
    main()
