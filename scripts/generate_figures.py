"""Evaluation figures from saved artefacts (no training, no new test inference).

Writes results/figures/*.png and the CSV data behind each figure to results/figures/data/.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from climatesense.evaluation.plotting import MUTED, RISK_COLORS, SERIES, TEXT, save, setup
from climatesense.labeling.compound_risk import CLASS_NAMES
from climatesense.utils import read_json

FIG = ROOT / "results/figures"
DATA = FIG / "data"
PRED = ROOT / "results/predictions"


def confusion(cm, title, path, normalise):
    cm = np.asarray(cm, float)
    M = cm / cm.sum(1, keepdims=True) if normalise else cm
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1 if normalise else M.max())
    ax.grid(False)
    for i in range(4):
        for j in range(4):
            txt = f"{M[i, j]:.2f}" if normalise else f"{int(M[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > (0.5 if normalise else M.max() / 2) else TEXT)
    ax.set_xticks(range(4), CLASS_NAMES); ax.set_yticks(range(4), CLASS_NAMES)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class"); ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Row-normalised rate" if normalise else "Samples")
    save(fig, path)


def roc_pr(y, P, title_suffix):
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
    rows = []
    for k in range(4):
        yk = (y == k).astype(int)
        fpr, tpr, _ = roc_curve(yk, P[:, k])
        prec, rec, _ = precision_recall_curve(yk, P[:, k])
        a_roc, a_pr = auc(fpr, tpr), auc(rec, prec)
        axs[0].plot(fpr, tpr, color=RISK_COLORS[k] if k else "#b9b6ad", label=f"{CLASS_NAMES[k]} (AUC={a_roc:.3f})")
        axs[1].plot(rec, prec, color=RISK_COLORS[k] if k else "#b9b6ad", label=f"{CLASS_NAMES[k]} (AP≈{a_pr:.3f})")
        axs[1].axhline(yk.mean(), color=RISK_COLORS[k] if k else "#b9b6ad", ls=":", lw=1)
        rows.append(dict(cls=CLASS_NAMES[k], roc_auc=a_roc, pr_auc=a_pr, prevalence=yk.mean()))
    axs[0].plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1)
    axs[0].set_xlabel("False positive rate"); axs[0].set_ylabel("True positive rate")
    axs[0].set_title(f"One-vs-rest ROC curves {title_suffix}")
    axs[1].set_xlabel("Recall"); axs[1].set_ylabel("Precision")
    axs[1].set_title(f"Precision-recall curves {title_suffix}")
    axs[0].legend(loc="lower right", fontsize=7)
    axs[1].legend(loc="lower left", fontsize=7)
    pd.DataFrame(rows).to_csv(DATA / "roc_pr_auc_selected.csv", index=False)
    return fig


def architecture(path):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 52); ax.axis("off")

    def box(x, y, w, h, text, color, fs=7.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=color, ec=MUTED, lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=TEXT)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))

    rows = [("Climate sequence\n14 d × 18 features", "Temporal encoder A\n(GRU/LSTM/Transformer)", 42),
            ("Air-quality sequence\n14 d × 10 features", "Temporal encoder B\n(GRU)", 31),
            ("Dryness-proxy seq.\n14 d × 10 features", "Temporal encoder C\n(GRU)", 20),
            ("Static context\n8 features", "Context MLP D", 9)]
    for inp, enc, y in rows:
        box(1, y, 17, 7, inp, "#eef4fc")
        box(23, y, 18, 7, enc, "#fdf0e9")
        box(46, y, 11, 7, "Linear proj.\n+ LayerNorm\n+ modality emb.", "#f4f3ef", 6.5)
        arrow(18.5, y + 3.5, 22.5, y + 3.5); arrow(41.5, y + 3.5, 45.5, y + 3.5); arrow(57.5, y + 3.5, 61.5, 27)
    box(62, 30, 16, 12, "Masked self-attention\namong modality tokens\n(missing modality →\nkey-padding mask)", "#e9f7f1", 6.8)
    box(62, 13, 16, 12, "Learned query\ncross-attention pooling\n→ modality weights α\n+ gated residual", "#e9f7f1", 6.8)
    arrow(70, 29.5, 70, 25.5)
    box(83, 13, 15, 12, "MLP head\n→ softmax\nLow / Moderate /\nHigh / Critical", "#fbeaea", 6.8)
    arrow(78.5, 19, 82.5, 19)
    ax.text(50, 50, "ClimateSense Multimodal Attention Fusion Network (MAF-Net)", ha="center", fontsize=10, color=TEXT)
    ax.text(50, 2.5, "Training: class-weighted CE / focal loss, AdamW, ReduceLROnPlateau, early stopping on validation macro-F1, modality dropout",
            ha="center", fontsize=7, color=MUTED)
    save(fig, path)


def main():
    setup()
    DATA.mkdir(parents=True, exist_ok=True)
    tm = read_json(ROOT / "results/metrics/test_metrics.json")
    sel = read_json(ROOT / "results/metrics/selected_model_test.json")
    rid = sel["selected_run"]
    rec = read_json(ROOT / "experiments" / rid.split("-")[0] / f"{rid}.json")
    d = np.load(PRED / f"{rid}_test.npz")
    y, P = d["y"], d["proba"]
    cm = tm[rid]["test_metrics"]["confusion_matrix"]
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(DATA / "confusion_selected.csv")
    confusion(cm, "MAF-Net confusion matrix (test, counts)", FIG / "confusion_matrix_raw.png", False)
    confusion(cm, "MAF-Net confusion matrix (test, normalised)", FIG / "confusion_matrix_normalized.png", True)
    pers = tm["baseline-persistence_L14_h1"]["test_metrics"]["confusion_matrix"]
    confusion(pers, "Persistence baseline (test, normalised)", FIG / "confusion_matrix_persistence.png", True)
    save(roc_pr(y, P, "(MAF-Net, test)"), FIG / "roc_pr_curves.png")

    # training curves of the selected run
    h = pd.DataFrame(rec["history"])
    h.to_csv(DATA / "training_history_selected.csv", index=False)
    fig, axs = plt.subplots(1, 3, figsize=(11, 3))
    axs[0].plot(h.epoch, h.train_nll, label="Train"); axs[0].plot(h.epoch, h.val_nll, label="Validation")
    axs[0].set_ylabel("Cross-entropy (unweighted)"); axs[0].set_title("Loss")
    axs[1].plot(h.epoch, h.train_macro_f1, label="Train"); axs[1].plot(h.epoch, h.val_macro_f1, label="Validation")
    axs[1].axvline(rec["best_epoch"], color=MUTED, ls=":", lw=1); axs[1].set_ylabel("Macro-F1"); axs[1].set_title("Macro-F1 (best epoch dotted)")
    axs[2].plot(h.epoch, h.lr, color=SERIES[2]); axs[2].set_yscale("log"); axs[2].set_ylabel("Learning rate"); axs[2].set_title("LR schedule")
    for ax in axs:
        ax.set_xlabel("Epoch")
    axs[0].legend(); axs[1].legend()
    save(fig, FIG / "training_curves.png")

    # model comparison
    mc = pd.read_csv(ROOT / "results/tables/model_comparison_h1.csv")
    mc = mc[mc.model != "Majority class"].sort_values("macro_f1")
    mc.to_csv(DATA / "model_comparison.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    yy = np.arange(len(mc))
    ax.barh(yy + 0.2, mc.macro_f1, height=0.38, xerr=mc.macro_f1_std.fillna(0), color=SERIES[0], label="Test macro-F1",
            error_kw={"ecolor": MUTED, "lw": 1})
    ax.barh(yy - 0.2, mc.balanced_accuracy, height=0.38, color=SERIES[1], label="Test balanced accuracy")
    ax.set_yticks(yy, mc.model); ax.set_xlim(max(0, mc.macro_f1.min() - 0.1), min(1, mc[["macro_f1", "balanced_accuracy"]].max().max() + 0.03))
    for i, (v, sd) in enumerate(zip(mc.macro_f1, mc.macro_f1_std.fillna(0))):
        ax.text(v + sd + 0.003, i + 0.2, f"{v:.3f}", va="center", fontsize=6.5, color=TEXT)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2); ax.grid(axis="y", visible=False)
    ax.set_title("Held-out test performance, horizon t+1 (mean ± s.d. over seeds)")
    save(fig, FIG / "model_comparison.png")

    # compact single-column version for the IEEE report
    short = {"MAF-Net (proposed, attention fusion)": "MAF-Net (proposed)", "MAF-Net w/o attention (concat fusion)": "MAF-Net, concat fusion",
             "Transformer encoder (tuned)": "Transformer (tuned)", "Gradient boosting (HistGB)": "Gradient boosting",
             "Persistence (today's class)": "Persistence", "Site-month climatology": "Climatology"}
    mcc = mc[mc.model != "Site-month climatology"]
    fig, ax = plt.subplots(figsize=(3.6, 3.9))
    yy = np.arange(len(mcc))
    ax.barh(yy, mcc.macro_f1, height=0.7, xerr=mcc.macro_f1_std.fillna(0), color=[SERIES[1] if m.startswith("MAF-Net (p") else SERIES[0] for m in mcc.model],
            error_kw={"ecolor": MUTED, "lw": 1})
    for i, (v, sd) in enumerate(zip(mcc.macro_f1, mcc.macro_f1_std.fillna(0))):
        ax.text(v + sd + 0.003, i, f"{v:.3f}", va="center", fontsize=7, color=TEXT)
    ax.set_yticks(yy, [short.get(m, m) for m in mcc.model], fontsize=7.5)
    ax.set_xlim(0.5, 0.775); ax.set_xlabel("Test macro-F1 (mean ± s.d. over seeds)"); ax.grid(axis="y", visible=False)
    ax.set_title("Held-out test, h = 1", fontsize=9)
    save(fig, FIG / "model_comparison_column.png")

    # ablation
    ab = pd.read_csv(ROOT / "results/tables/ablation.csv").sort_values("macro_f1")
    ab.to_csv(DATA / "ablation.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    yy = np.arange(len(ab))
    colors = [SERIES[1] if v == "all_attention" else SERIES[0] for v in ab.variant]
    ax.barh(yy + 0.2, ab.macro_f1, height=0.38, xerr=ab.macro_f1_std, color=colors, error_kw={"ecolor": MUTED, "lw": 1}, label="Test")
    ax.barh(yy - 0.2, ab.val_macro_f1, height=0.38, xerr=ab.val_macro_f1_std, color="#b9b6ad", error_kw={"ecolor": MUTED, "lw": 1}, label="Validation")
    ax.set_yticks(yy, ab.variant.str.replace("_", " ")); ax.set_xlim(max(0, ab[["macro_f1", "val_macro_f1"]].min().min() - 0.05), ab[["macro_f1", "val_macro_f1"]].max().max() + 0.02)
    ax.set_xlabel("Macro-F1 (mean ± s.d., 3 seeds)"); ax.grid(axis="y", visible=False); ax.legend(loc="lower right")
    ax.set_title("Ablation study (full MAF-Net highlighted)")
    save(fig, FIG / "ablation.png")

    # horizons
    hz = pd.read_csv(ROOT / "results/tables/horizons.csv")
    hz.to_csv(DATA / "horizons.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    for j, (m, g) in enumerate(hz.groupby("model", sort=False)):
        g = g.sort_values("horizon")
        ax.errorbar(g.horizon, g.macro_f1, yerr=g.macro_f1_std.fillna(0), marker="o", ms=5, color=SERIES[j], label=m, capsize=2)
    ax.set_xticks([1, 3, 7]); ax.set_xlabel("Forecast horizon (days)"); ax.set_ylabel("Test macro-F1")
    ax.set_title("Skill decay with forecast horizon"); ax.legend()
    save(fig, FIG / "horizons.png")

    # calibration
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    for k in range(4):
        fr, mp = calibration_curve((y == k).astype(int), P[:, k], n_bins=10, strategy="quantile")
        ax.plot(mp, fr, marker="o", ms=4, color=RISK_COLORS[k] if k else "#b9b6ad", label=CLASS_NAMES[k])
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1)
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency"); ax.set_title("Reliability diagram (MAF-Net, test)")
    ax.legend(fontsize=7)
    save(fig, FIG / "calibration.png")

    # optuna trials
    fig, ax = plt.subplots(figsize=(6.5, 3))
    for j, m in enumerate(["mafnet", "gru", "transformer"]):
        f = ROOT / f"experiments/tuning/{m}_trials.json"
        if f.exists():
            t = pd.DataFrame(read_json(f))
            t.to_csv(DATA / f"optuna_{m}.csv", index=False)
            t = t[t.value.notna()]
            ax.scatter(t.number, t.value, s=14, color=SERIES[j], label=m)
            ax.plot(t.number, t.value.cummax(), color=SERIES[j], lw=1)
    ax.set_xlabel("Trial"); ax.set_ylabel("Validation macro-F1"); ax.set_title("Optuna TPE search (line = running best)")
    ax.legend()
    save(fig, FIG / "optuna_trials.png")

    # look-back sensitivity (validation: selection-relevant; test: reported for completeness)
    lb = pd.read_csv(ROOT / "results/tables/lookback.csv").sort_values("lookback")
    lb.to_csv(DATA / "lookback.csv", index=False)
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.errorbar(lb.lookback, lb.val_macro_f1, yerr=lb.val_macro_f1_std, marker="o", color=SERIES[0], capsize=2, label="Validation")
    ax.errorbar(lb.lookback, lb.macro_f1, yerr=lb.macro_f1_std, marker="s", color=SERIES[1], capsize=2, label="Test")
    ax.set_xticks(lb.lookback); ax.set_xlabel("Look-back window (days)"); ax.set_ylabel("Macro-F1 (3 seeds)")
    ax.set_title("MAF-Net look-back sensitivity"); ax.legend()
    save(fig, FIG / "lookback.png")

    # geographic hold-out
    sp = pd.read_csv(ROOT / "results/tables/spatial_holdout.csv")
    sp.to_csv(DATA / "spatial_holdout.csv", index=False)
    lab = {"mafnet": "MAF-Net", "gru": "GRU", "persistence": "Persistence"}
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    x = np.arange(len(sp))
    ax.bar(x - 0.2, sp.heldout_macro_f1, 0.38, yerr=sp.get("heldout_macro_f1_std", pd.Series(0, index=sp.index)).fillna(0),
           color=SERIES[0], label="Unseen sites (20 %)", error_kw={"ecolor": MUTED, "lw": 1})
    ax.bar(x + 0.2, sp.seen_sites_macro_f1, 0.38, color=SERIES[2], label="Training sites")  # NaN for persistence: not drawn
    ax.set_xticks(x, [lab.get(m, m) for m in sp.model]); ax.set_ylabel("Test macro-F1")
    ax.set_ylim(0, 1); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2); ax.set_title("Spatial generalisation (test period)")
    save(fig, FIG / "spatial_holdout.png")

    architecture(FIG / "architecture_mafnet.png")
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
