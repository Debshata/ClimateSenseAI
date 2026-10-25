"""Staged neural experiments (all selection on the validation split; resumable via run-id caching).

  python scripts/run_experiments.py --stage neural_baselines
  python scripts/run_experiments.py --stage tune        # Optuna: MAF-Net 24 trials, GRU 10, Transformer 10
  python scripts/run_experiments.py --stage seeds       # 5 seeds for the selected configurations
  python scripts/run_experiments.py --stage ablation    # modality / fusion ablations, 3 seeds each
  python scripts/run_experiments.py --stage horizons    # h = 3, 7
  python scripts/run_experiments.py --stage lookback    # L = 7, 30
  python scripts/run_experiments.py --stage spatial     # held-out sites
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import optuna

from climatesense.training.trainer import DEFAULTS, get_arrays, train
from climatesense.utils import atomic_write_json, read_json

SEEDS5 = [42, 7, 123, 2024, 31337]
SEEDS3 = SEEDS5[:3]
TUNE_EPOCHS, TUNE_PATIENCE = 20, 5
FINAL_EPOCHS, FINAL_PATIENCE = 40, 6
SELECTED = ROOT / "experiments/tuning/selected_configs.json"


def neural_baselines():
    out = {}
    for model in ["rnn", "lstm", "gru", "transformer"]:
        r = train(dict(model=model, epochs=FINAL_EPOCHS, patience=FINAL_PATIENCE), group="multimodal")
        out[model] = r["val_metrics"]["macro_f1"]
        print(model, out[model], flush=True)
    for fusion in ["attention", "concat"]:
        r = train(dict(model="mafnet", fusion=fusion, epochs=FINAL_EPOCHS, patience=FINAL_PATIENCE), group="multimodal")
        out[f"mafnet_{fusion}"] = r["val_metrics"]["macro_f1"]
        print("mafnet", fusion, out[f"mafnet_{fusion}"], flush=True)
    atomic_write_json(out, ROOT / "experiments/multimodal/initial_summary.json")


def _space(trial, model):
    cfg = dict(
        hidden=trial.suggest_categorical("hidden", [32, 64, 128]),
        layers=trial.suggest_categorical("layers", [1, 2]),
        dropout=trial.suggest_categorical("dropout", [0.1, 0.2, 0.3]),
        lr=trial.suggest_categorical("lr", [3e-3, 1e-3, 3e-4]),
        batch_size=trial.suggest_categorical("batch_size", [128, 256]),
        weight_decay=trial.suggest_categorical("weight_decay", [0.0, 1e-4, 1e-3]),
        loss=trial.suggest_categorical("loss", ["ce", "wce", "focal"]),
    )
    if model == "mafnet":
        cfg.update(
            encoder=trial.suggest_categorical("encoder", ["gru", "lstm", "transformer"]),
            d_model=trial.suggest_categorical("d_model", [32, 64, 128]),
            heads=trial.suggest_categorical("heads", [1, 2, 4]),
            modality_dropout=trial.suggest_categorical("modality_dropout", [0.0, 0.1, 0.2]),
            token_self_attention=trial.suggest_categorical("token_self_attention", [True, False]),
        )
    elif model == "transformer":
        cfg.update(heads=trial.suggest_categorical("heads", [1, 2, 4]))
    return cfg


def tune(model: str, n_trials: int, sampler_seed: int = 42):
    """Resumable Optuna study; several worker processes may share it (each with its own sampler
    seed).  MaxTrialsCallback caps the total number of finished trials across workers."""
    storage = f"sqlite:///{(ROOT / 'experiments/tuning/optuna.db').as_posix()}"
    study = optuna.create_study(study_name=f"{model}_h1_L14", storage=storage, direction="maximize",
                                load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=sampler_seed),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=4))
    done = len([t for t in study.trials if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)])

    def objective(trial):
        cfg = dict(model=model, epochs=TUNE_EPOCHS, patience=TUNE_PATIENCE, **_space(trial, model))
        r = train(cfg, group="tuning", trial=trial, verbose=False)
        trial.set_user_attr("run_id", r["run_id"])
        trial.set_user_attr("n_params", r["n_params"])
        print(f"[{model}] trial {trial.number}: valF1={r['val_metrics']['macro_f1']:.4f} {trial.params}", flush=True)
        return r["val_metrics"]["macro_f1"]

    states = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    if n_trials - done > 0:
        study.optimize(objective, n_trials=n_trials - done,
                       callbacks=[optuna.study.MaxTrialsCallback(n_trials, states=states)])
    # trials left RUNNING by a killed worker (restart to cap CPU threads) are reported as interrupted
    rows = [dict(number=t.number, value=t.value, state="INTERRUPTED" if t.state.name == "RUNNING" else t.state.name,
                 params=t.params, run_id=t.user_attrs.get("run_id"),
                 n_params=t.user_attrs.get("n_params")) for t in study.trials]
    atomic_write_json(rows, ROOT / f"experiments/tuning/{model}_trials.json")
    return study.best_params, study.best_value


def stage_tune():
    sel = {}
    for model, n in [("mafnet", 24), ("gru", 10), ("transformer", 10)]:
        bp, bv = tune(model, n)
        sel[model] = dict(params=bp, val_macro_f1=bv)
        print("BEST", model, bv, bp, flush=True)
    atomic_write_json(sel, SELECTED)


def selected(model):
    s = read_json(SELECTED)[model]["params"]
    return dict(model=model, epochs=FINAL_EPOCHS, patience=FINAL_PATIENCE, **s)


def stage_seeds():
    plan = [("mafnet", selected("mafnet"), SEEDS5),
            ("mafnet_concat", {**selected("mafnet"), "fusion": "concat"}, SEEDS5),
            ("gru", selected("gru"), SEEDS5),
            ("transformer", selected("transformer"), SEEDS3),
            ("lstm", dict(model="lstm", epochs=FINAL_EPOCHS, patience=FINAL_PATIENCE, **{k: v for k, v in read_json(SELECTED)["gru"]["params"].items()}), SEEDS3),
            ("rnn", dict(model="rnn", epochs=FINAL_EPOCHS, patience=FINAL_PATIENCE), SEEDS3)]
    summary = {}
    for name, cfg, seeds in plan:
        recs = [train({**cfg, "seed": s}, group="final") for s in seeds]
        f1 = [r["val_metrics"]["macro_f1"] for r in recs]
        summary[name] = dict(run_ids=[r["run_id"] for r in recs], val_macro_f1_mean=float(np.mean(f1)),
                             val_macro_f1_std=float(np.std(f1, ddof=1)) if len(f1) > 1 else 0.0)
        print(name, summary[name], flush=True)
    atomic_write_json(summary, ROOT / "experiments/final_val_summary.json")


ABLATIONS = {
    "climate_only": dict(modalities=["climate"], use_context=False),
    "air_only": dict(modalities=["air"], use_context=False),
    "dryness_only": dict(modalities=["dryness"], use_context=False),
    "climate_air": dict(modalities=["climate", "air"]),
    "climate_dryness": dict(modalities=["climate", "dryness"]),
    "all_concat_no_attention": dict(fusion="concat"),
    "all_attention_no_context": dict(use_context=False),
    "all_attention": dict(),
    "no_humidity": dict(drop_features=["humidity"]),
    "no_air_quality": dict(modalities=["climate", "dryness"]),
    "no_dryness": dict(modalities=["climate", "air"]),
    "with_modality_dropout": dict(modality_dropout=0.1),  # selected config uses 0.0
    "with_transferred_drought_features": dict(transfer_drought=True),
}


def stage_ablation(reverse: bool = False):
    """reverse=True lets a second worker process the same variants from the other end; completed
    runs are skipped through the run-id cache, so the two workers meet in the middle."""
    base = selected("mafnet")
    summary = {}
    items = list(ABLATIONS.items())
    for name, delta in (items[::-1] if reverse else items):
        recs = [train({**base, **delta, "seed": s}, group="ablation") for s in SEEDS3]
        f1 = [r["val_metrics"]["macro_f1"] for r in recs]
        summary[name] = dict(delta=delta, run_ids=[r["run_id"] for r in recs], val_macro_f1_mean=float(np.mean(f1)),
                             val_macro_f1_std=float(np.std(f1, ddof=1)))
        print(name, summary[name]["val_macro_f1_mean"], flush=True)
    if not reverse:
        atomic_write_json(dict(sorted(summary.items(), key=lambda kv: list(ABLATIONS).index(kv[0]))),
                          ROOT / "experiments/ablation/summary.json")


def stage_horizons():
    summary = {}
    for h in [3, 7]:
        for name, cfg in [("mafnet", selected("mafnet")), ("gru", selected("gru"))]:
            recs = [train({**cfg, "horizon": h, "seed": s}, group="horizons") for s in SEEDS3]
            summary[f"{name}_h{h}"] = dict(run_ids=[r["run_id"] for r in recs],
                                           val_macro_f1_mean=float(np.mean([r["val_metrics"]["macro_f1"] for r in recs])))
            print(name, h, summary[f"{name}_h{h}"], flush=True)
    atomic_write_json(summary, ROOT / "experiments/horizons_summary.json")


def stage_lookback():
    summary = {}
    for L in [7, 30]:
        recs = [train({**selected("mafnet"), "lookback": L, "seed": s}, group="lookback") for s in SEEDS3]
        summary[f"mafnet_L{L}"] = dict(run_ids=[r["run_id"] for r in recs],
                                       val_macro_f1_mean=float(np.mean([r["val_metrics"]["macro_f1"] for r in recs])))
        print(L, summary[f"mafnet_L{L}"], flush=True)
    atomic_write_json(summary, ROOT / "experiments/lookback_summary.json")


def spatial_holdout_sites(seed=0, frac=0.2):
    arr = get_arrays()
    n = len(arr.site_index)
    rng = np.random.default_rng(seed)
    held = np.sort(rng.choice(n, int(round(frac * n)), replace=False))
    train_sites = np.setdiff1d(np.arange(n), held)
    return train_sites, held


def stage_spatial():
    train_sites, held = spatial_holdout_sites()
    atomic_write_json({"held_out_site_codes": held.tolist(), "train_site_codes": train_sites.tolist(),
                       "held_out_site_ids": [str(get_arrays().site_index[i]) for i in held]},
                      ROOT / "experiments/spatial_split.json")
    summary = {}
    for name, cfg in [("mafnet", selected("mafnet")), ("gru", selected("gru"))]:
        recs = [train({**cfg, "train_sites": train_sites.tolist(), "seed": s}, group="spatial") for s in SEEDS3]
        summary[name] = dict(run_ids=[r["run_id"] for r in recs],
                             val_macro_f1_mean=float(np.mean([r["val_metrics"]["macro_f1"] for r in recs])))
        print(name, summary[name], flush=True)
    atomic_write_json(summary, ROOT / "experiments/spatial_summary.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--models", nargs="*", default=["gru", "transformer"])
    ap.add_argument("--sampler-seed", type=int, default=42)
    args = ap.parse_args()
    stage = args.stage
    if stage == "tune_side":  # tune baseline models in a parallel process (same Optuna storage)
        budget = {"mafnet": 24, "gru": 10, "transformer": 10}
        for m in args.models:
            print("BEST", m, tune(m, budget[m], args.sampler_seed), flush=True)
        sys.exit(0)
    {"neural_baselines": neural_baselines, "tune": stage_tune, "seeds": stage_seeds, "ablation": stage_ablation,
     "horizons": stage_horizons, "ablation_rev": lambda: stage_ablation(reverse=True), "lookback": stage_lookback, "spatial": stage_spatial}[stage]()
