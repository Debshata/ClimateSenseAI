# Reproducibility Guide

## 1. Environment

Reference machine: Windows 11 Pro, Intel Xeon E-2176M (12 logical CPUs), 15.8 GB RAM,
NVIDIA Quadro P3200 (6 GB, compute capability 6.1), Python 3.13.7, PyTorch 2.14.1 + CUDA 12.6.
Full details: `docs/environment_info.json` (written by `scripts/setup_environment.py`).

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/python -m pip install -r requirements-dev.txt
.venv/Scripts/python scripts/setup_environment.py
```

The CUDA 12.6 wheel is used deliberately: the CUDA 12.8+ wheels no longer ship kernels for
Pascal (sm_61) GPUs. Everything also runs on CPU (slower). Mixed precision is **not** used:
Pascal GPUs have very low FP16 throughput, so AMP would slow training down.

## 2. Credentials

Create `.env` in the project root (git-ignored) with your own Kaggle token:

```
KAGGLE_API_TOKEN=<your token>
```

## 3. Pipeline (each step is idempotent / resumable)

| Step | Command | Main outputs |
|---|---|---|
| Download | `python scripts/download_datasets.py` | `data/raw/*`, `data/manifests/download_manifest.json` |
| Harmonise + label | `python scripts/prepare_data.py --force` | `data/processed/panel.parquet`, `sites.csv`, `data/manifests/data_audit.json`, `splits.json` |
| EDA / audit | `python scripts/inspect_datasets.py` | `data/manifests/dataset_quality.json`, `results/figures/eda/` |
| Label validity | `python scripts/analyze_labels.py` | `results/analysis/label_sensitivity.json` |
| Classical baselines | `python scripts/train_baselines.py --horizon 1` (and `--horizon 3/7 --only hist_gradient_boosting logreg`) | `experiments/baseline/` |
| Drought benchmark + transfer | `python scripts/train_drought_aux.py` | `experiments/drought_aux/`, `data/processed/drought_transfer_features.parquet` |
| Neural experiments | `python scripts/run_experiments.py --stage {neural_baselines,tune,seeds,ablation,horizons,lookback,spatial}` | `experiments/{multimodal,tuning,final,ablation,horizons,lookback,spatial}/` |
| Held-out evaluation (once) | `python scripts/evaluate_models.py` | `results/metrics/`, `results/tables/` |
| Explainability | `python scripts/explain_model.py` | `results/analysis/explainability.json`, `results/figures/explain/` |
| Figures | `python scripts/generate_figures.py` | `results/figures/` |
| Notebooks | `python scripts/build_notebooks.py` | `notebooks/*.ipynb` (executed) |
| Report | `python scripts/generate_report.py` | `reports/final/ClimateSenseAI_IEEE_Final_Report.docx` (+ PDF) |
| Tests | `python -m pytest` | — |
| Deliverable check | `python scripts/verify_deliverables.py` | `docs/deliverable_check.json` |

`make all` runs the whole chain where `make` is available.

## 4. Determinism

* Seeds: Python, NumPy and PyTorch are seeded per run (`set_seed`); cuDNN deterministic mode is
  on and benchmark mode off. The window shuffling uses a seeded CPU generator.
* Every neural run has a run-id = hash of its full configuration; re-running a stage skips runs
  whose JSON record is complete, so interrupted stages resume without recomputation.
* Remaining nondeterminism: some CUDA kernels (e.g. RNN backward, scatter/gather) are not
  bit-wise deterministic, so re-training on another GPU/driver can give slightly different
  numbers. Variation across seeds is reported (mean ± s.d.) for this reason.
* Inference is deterministic: notebook 11 and `tests/test_metrics.py` reload the selected
  checkpoint and reproduce the stored test probabilities.

## 5. Data versions

The Kaggle weather repository is updated daily. The experiments used **version 1105**
(version 1106 was empty on 2026-10-25). To reproduce exactly, download that version:

```python
kagglehub.dataset_download("nelgiriyewithana/global-weather-repository/versions/1105")
```

Other versions used: US drought v5, World Cities v12, Berkeley Earth v2.

## 6. What is and is not in Git

Committed: code, configs, notebooks (executed), docs, figures, tables, metrics, experiment
records (JSON), the selected model checkpoint (`models/best/`), the report.
Not committed: raw/processed data (re-downloadable), all intermediate checkpoints, `.env`.
