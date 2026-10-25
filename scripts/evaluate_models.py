"""Single held-out test evaluation of all finalised configurations.

Run once, after tuning/selection is frozen (experiments/tuning/selected_configs.json and the
seed/ablation/horizon/spatial stages).  Nothing computed here feeds back into model selection.

Outputs: results/metrics/test_metrics.json, results/tables/*.csv|md, results/predictions/*_test.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd

from climatesense.data.windows import tabular_features
from climatesense.evaluation.metrics import bootstrap_ci, compute_metrics, paired_bootstrap_diff
from climatesense.training.trainer import get_arrays, load_trained, predict
from climatesense.utils import atomic_write_json, read_json
sys.path.insert(0, str(ROOT / "scripts"))
from train_baselines import naive_proba  # noqa: E402

PRED = ROOT / "results/predictions"
TAB = ROOT / "results/tables"
MODALITY_LABEL = {"climate": "C", "air": "A", "dryness": "D"}


def modalities_str(cfg):
    if cfg["model"] != "mafnet":
        return "C+A+D (single stream)" + ("+ctx" if cfg.get("use_context") else "")
    s = "+".join(MODALITY_LABEL[m] for m in cfg["modalities"])
    if cfg.get("use_context"):
        s += "+ctx"
    if cfg.get("drop_features"):
        s += " −" + ",".join(cfg["drop_features"])
    if cfg.get("transfer_drought"):
        s += "+TL"
    return s


def eval_neural(rec, split_sites=None):
    """Predict the test split with the stored best-validation checkpoint."""
    model, batcher, arr = load_trained(rec)
    cfg = rec["config"]
    idx = arr.sample_index(cfg["lookback"], cfg["horizon"], "test", cfg["max_missing"], split_sites)
    P, Y, A = predict(model, batcher, idx)
    tag = rec["run_id"] + ("" if split_sites is None else "_heldout_sites")
    np.savez_compressed(PRED / f"{tag}_test.npz", idx=idx, proba=P, y=Y, att=A if A is not None else np.zeros(0))
    return compute_metrics(Y, P), idx, P, Y


def main():
    arr = get_arrays()
    out = {}
    rows = []

    # -------------------------------------------------- naive + classical baselines (h = 1, 3, 7)
    for h in (1, 3, 7):
        tr = arr.sample_index(14, h, "train")
        te = arr.sample_index(14, h, "test")
        yte = arr.y[te + h].astype(int)
        for kind in ["persistence", "majority", "climatology"]:
            P = naive_proba(kind, arr, te, h, tr)
            name = f"baseline-{kind}_L14_h{h}"
            m = compute_metrics(yte, P)
            out[name] = dict(model=kind, horizon=h, test_metrics=m, group="baseline")
            np.savez_compressed(PRED / f"{name}_test.npz", idx=te, proba=P, y=yte)
        Fte = None
        for kind in ["logreg", "random_forest", "hist_gradient_boosting", "mlp"]:
            path = ROOT / "models/checkpoints" / f"baseline-{kind}_L14_h{h}.joblib"
            if not path.exists():
                continue
            if Fte is None:
                X, C, _ = arr.windows(te, 14, h)
                Fte, _ = tabular_features(X, C, arr.feature_names)
            mdl = joblib.load(path)
            P = (mdl["model"].predict_proba((Fte - mdl["mu"]) / mdl["sd"]) if isinstance(mdl, dict) else mdl.predict_proba(Fte))
            name = f"baseline-{kind}_L14_h{h}"
            rec = read_json(ROOT / "experiments/baseline" / f"{name}.json")
            out[name] = dict(model=kind, horizon=h, test_metrics=compute_metrics(yte, P), group="baseline",
                             val_macro_f1=rec["val_metrics"]["macro_f1"], train_time_s=rec["train_time_s"],
                             n_params=rec["n_params"])
            np.savez_compressed(PRED / f"{name}_test.npz", idx=te, proba=P, y=yte)
        print("baselines h", h, "done", flush=True)

    # -------------------------------------------------- neural runs
    spatial = read_json(ROOT / "experiments/spatial_split.json") if (ROOT / "experiments/spatial_split.json").exists() else None
    for group in ["multimodal", "final", "ablation", "horizons", "lookback", "spatial"]:
        for f in sorted((ROOT / "experiments" / group).glob("*.json")):
            rec = read_json(f)
            if not isinstance(rec, dict) or rec.get("status") != "complete":
                continue
            entry = dict(model=rec["config"]["model"], group=group, horizon=rec["config"]["horizon"],
                         seed=rec["config"]["seed"], val_macro_f1=rec["val_metrics"]["macro_f1"],
                         n_params=rec["n_params"], train_time_s=rec["train_time_s"],
                         modalities=modalities_str(rec["config"]))
            if group == "spatial":
                held = np.array(spatial["held_out_site_codes"])
                entry["test_metrics_heldout_sites"] = eval_neural(rec, held)[0]
                entry["test_metrics"] = eval_neural(rec, np.array(spatial["train_site_codes"]))[0]
            else:
                entry["test_metrics"] = eval_neural(rec)[0]
            out[rec["run_id"]] = entry
        print(group, "done", flush=True)

    # spatial baselines on held-out sites (HGB trained on all sites is NOT comparable; use persistence)
    if spatial:
        held = np.array(spatial["held_out_site_codes"])
        te = arr.sample_index(14, 1, "test", sites=held)
        P = naive_proba("persistence", arr, te, 1, arr.sample_index(14, 1, "train"))
        out["baseline-persistence_heldout_sites"] = dict(model="persistence", group="spatial", horizon=1,
                                                         test_metrics_heldout_sites=compute_metrics(arr.y[te + 1].astype(int), P))
    atomic_write_json(out, ROOT / "results/metrics/test_metrics.json")
    build_tables(out)


def _agg(entries, key="test_metrics"):
    ms = ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1",
          "roc_auc_macro_ovr", "pr_auc_macro", "quadratic_kappa"]
    d = {}
    for m in ms:
        v = [e[key][m] for e in entries if e[key][m] is not None]
        d[m] = np.mean(v) if v else np.nan
        d[m + "_std"] = np.std(v, ddof=1) if len(v) > 1 else 0.0
    d["val_macro_f1"] = np.mean([e.get("val_macro_f1", np.nan) for e in entries])
    d["val_macro_f1_std"] = np.std([e.get("val_macro_f1", np.nan) for e in entries], ddof=1) if len(entries) > 1 else 0.0
    d["train_time_s"] = np.mean([e.get("train_time_s", 0) for e in entries])
    d["n_params"] = entries[0].get("n_params", 0)
    d["n_runs"] = len(entries)
    return d


def build_tables(out):
    TAB.mkdir(parents=True, exist_ok=True)
    fsum = read_json(ROOT / "experiments/final_val_summary.json")
    names = {"mafnet": "MAF-Net (proposed, attention fusion)", "mafnet_concat": "MAF-Net w/o attention (concat fusion)",
             "gru": "GRU (tuned)", "transformer": "Transformer encoder (tuned)", "lstm": "LSTM", "rnn": "Vanilla RNN"}
    rows = []
    for key in ["persistence", "majority", "climatology", "logreg", "random_forest", "hist_gradient_boosting", "mlp"]:
        e = out.get(f"baseline-{key}_L14_h1")
        if e:
            r = _agg([{**e, "val_macro_f1": e.get("val_macro_f1", np.nan)}])
            r.update(model={"persistence": "Persistence (today's class)", "majority": "Majority class",
                            "climatology": "Site-month climatology", "logreg": "Logistic regression",
                            "random_forest": "Random forest", "hist_gradient_boosting": "Gradient boosting (HistGB)",
                            "mlp": "MLP (sklearn)"}[key], modalities="C+A+D (aggregated)" if key not in ("persistence", "majority", "climatology") else "label history" if key == "persistence" else "—")
            rows.append(r)
    for key, v in fsum.items():
        es = [out[r] for r in v["run_ids"] if r in out]
        r = _agg(es)
        r.update(model=names[key], modalities=es[0]["modalities"])
        rows.append(r)
    df = pd.DataFrame(rows)
    cols = ["model", "modalities", "n_params", "n_runs", "val_macro_f1", "val_macro_f1_std", "accuracy", "accuracy_std",
            "balanced_accuracy", "balanced_accuracy_std", "macro_precision", "macro_recall", "macro_f1", "macro_f1_std",
            "weighted_f1", "roc_auc_macro_ovr", "pr_auc_macro", "quadratic_kappa", "train_time_s"]
    df[cols].to_csv(TAB / "model_comparison_h1.csv", index=False)
    with open(TAB / "model_comparison_h1.md", "w", encoding="utf-8") as f:
        f.write(df[cols].round(4).to_markdown(index=False))

    # ablation table
    ab = read_json(ROOT / "experiments/ablation/summary.json")
    arows = []
    for name, v in ab.items():
        es = [out[r] for r in v["run_ids"] if r in out]
        r = _agg(es)
        r.update(variant=name, modalities=es[0]["modalities"])
        arows.append(r)
    adf = pd.DataFrame(arows)
    acols = ["variant", "modalities", "n_params", "val_macro_f1", "val_macro_f1_std", "macro_f1", "macro_f1_std",
             "balanced_accuracy", "accuracy", "roc_auc_macro_ovr", "pr_auc_macro"]
    adf[acols].to_csv(TAB / "ablation.csv", index=False)
    with open(TAB / "ablation.md", "w", encoding="utf-8") as f:
        f.write(adf[acols].round(4).to_markdown(index=False))

    # horizons
    hrows = []
    hs = read_json(ROOT / "experiments/horizons_summary.json")
    for h in (1, 3, 7):
        for kind, lab in [("persistence", "Persistence"), ("hist_gradient_boosting", "Gradient boosting")]:
            e = out.get(f"baseline-{kind}_L14_h{h}")
            if e:
                hrows.append(dict(horizon=h, model=lab, **{k: v for k, v in _agg([e]).items() if not k.startswith("val")}))
        for kind, lab in [("mafnet", "MAF-Net"), ("gru", "GRU")]:
            ids = fsum[kind]["run_ids"][:3] if h == 1 else hs.get(f"{kind}_h{h}", {}).get("run_ids", [])
            es = [out[r] for r in ids if r in out]
            if es:
                hrows.append(dict(horizon=h, model=lab, **_agg(es)))
    hdf = pd.DataFrame(hrows)
    hdf.to_csv(TAB / "horizons.csv", index=False)
    with open(TAB / "horizons.md", "w", encoding="utf-8") as f:
        f.write(hdf[["horizon", "model", "n_runs", "accuracy", "balanced_accuracy", "macro_f1", "macro_f1_std",
                     "roc_auc_macro_ovr"]].round(4).to_markdown(index=False))

    # lookback & spatial
    lrows = []
    for L, ids in [(14, fsum["mafnet"]["run_ids"][:3])] + [(int(k.split("L")[1]), v["run_ids"]) for k, v in read_json(ROOT / "experiments/lookback_summary.json").items()]:
        es = [out[r] for r in ids if r in out]
        lrows.append(dict(lookback=L, **_agg(es)))
    pd.DataFrame(lrows).sort_values("lookback").to_csv(TAB / "lookback.csv", index=False)
    srows = []
    for kind, v in read_json(ROOT / "experiments/spatial_summary.json").items():
        es = [out[r] for r in v["run_ids"] if r in out]
        a = _agg(es, "test_metrics_heldout_sites")
        b = _agg(es, "test_metrics")
        srows.append(dict(model=kind, heldout_macro_f1=a["macro_f1"], heldout_macro_f1_std=a["macro_f1_std"],
                          heldout_balanced_accuracy=a["balanced_accuracy"], heldout_accuracy=a["accuracy"],
                          seen_sites_macro_f1=b["macro_f1"], seen_sites_balanced_accuracy=b["balanced_accuracy"]))
    p = out.get("baseline-persistence_heldout_sites")
    if p:
        m = p["test_metrics_heldout_sites"]
        srows.append(dict(model="persistence", heldout_macro_f1=m["macro_f1"], heldout_balanced_accuracy=m["balanced_accuracy"],
                          heldout_accuracy=m["accuracy"]))
    pd.DataFrame(srows).to_csv(TAB / "spatial_holdout.csv", index=False)

    # ---------------------------------------------- selected model, CIs and paired comparisons
    mf = fsum["mafnet"]
    vals = {r: out[r]["val_macro_f1"] for r in mf["run_ids"]}
    sel = max(vals, key=vals.get)  # selection by validation macro-F1 only
    arr = get_arrays()
    d = np.load(PRED / f"{sel}_test.npz")
    groups = arr.site_codes[d["idx"]]
    stats = {"selected_run": sel, "selection_rule": "highest validation macro-F1 among the 5 MAF-Net seeds",
             "test_metrics": out[sel]["test_metrics"],
             "ci95_macro_f1_site_block_bootstrap": bootstrap_ci(d["y"], d["proba"], "macro_f1", 1000, 0, groups),
             "ci95_accuracy_site_block_bootstrap": bootstrap_ci(d["y"], d["proba"], "accuracy", 1000, 0, groups),
             "ci95_balanced_accuracy_site_block_bootstrap": bootstrap_ci(d["y"], d["proba"], "balanced_accuracy", 1000, 0, groups),
             "paired_vs": {}}
    comps = {"persistence": "baseline-persistence_L14_h1", "hist_gradient_boosting": "baseline-hist_gradient_boosting_L14_h1",
             "random_forest": "baseline-random_forest_L14_h1"}
    for k in ["gru", "mafnet_concat", "transformer"]:
        ids = fsum[k]["run_ids"]
        comps[k] = max(ids, key=lambda r: out[r]["val_macro_f1"])
    for k, rid in comps.items():
        e = np.load(PRED / f"{rid}_test.npz")
        assert np.array_equal(e["idx"], d["idx"]), f"sample mismatch with {k}"
        stats["paired_vs"][k] = {"run": rid, **paired_bootstrap_diff(d["y"], d["proba"], e["proba"], 1000, 0, groups)}
    atomic_write_json(stats, ROOT / "results/metrics/selected_model_test.json")
    print(df[["model", "val_macro_f1", "macro_f1", "balanced_accuracy", "accuracy", "roc_auc_macro_ovr"]].round(4).to_string())
    print(adf[["variant", "val_macro_f1", "macro_f1"]].round(4).to_string())
    print(stats["ci95_macro_f1_site_block_bootstrap"], {k: v["diff"] for k, v in stats["paired_vs"].items()})


if __name__ == "__main__":
    if "--tables-only" in sys.argv:
        build_tables(read_json(ROOT / "results/metrics/test_metrics.json"))
    else:
        main()
