# Final Execution Status

Generated 2026-10-25 18:17 by `scripts/write_docs.py` from saved artefacts and live checks.

## A. Completion summary

| Phase | Status | Evidence |
|---|---|---|
| 1 Repository & environment | done | `.venv` (Python 3.13.7, torch 2.14.1+cu126), `docs/environment_info.json` |
| 2 Dataset acquisition (four Kaggle datasets) | done | `data/manifests/download_manifest.json` |
| 2b Official sources (NASA POWER, SPEIbase, MODIS, OpenAQ, EM-DAT) | not used: the project was restricted to the four Kaggle datasets | `docs/dataset_documentation.md` |
| 3 Exploration & audit | done | `data/manifests/dataset_quality.json`, `results/figures/eda/` |
| 4 Harmonisation | done | `data/manifests/data_audit.json` |
| 5 Risk labelling + sensitivity | done | `docs/label_definition.md`, `results/analysis/label_sensitivity.json` |
| 6 Baselines | done | `experiments/baseline/`, `experiments/multimodal/` |
| 7 Proposed model (MAF-Net) | done | `src/climatesense/models/fusion.py` |
| 8 Hyper-parameter tuning | done | `experiments/tuning/` (Optuna) |
| 9 Evaluation, ablation, explainability, validity checks | done | `results/` |
| 10 Notebooks | 11/11 executed without errors | `notebooks/validation_manifest.json` |
| 11 Documentation | done | `README.md`, `docs/` |
| 12 IEEE report | done (DOCX + PDF via MS Word, visually inspected) | `reports/final/` |
| 13 Quality assurance | tests: `................................                                         [100%]` | `tests/` |
| 14 GitHub | pushed successfully; remote main = 75a67c6 verified with git ls-remote on 2026-10-25 18:16 | §G |

## B. Datasets

| Dataset | Handle / version | Records | Coverage | Usage |
|---|---|---|---|---|
| World Weather Repository | `nelgiriyewithana/global-weather-repository` v1105 (v1106 was empty) | 170,733 rows × 41 cols | 2024-05-16 → 2026-10-12; 185 of 279 sites kept | inputs + labels |
| World Cities | `juanmah/world-cities` v12 | 50,250 | global | coordinates, ISO3, population (182/185 sites matched) |
| Berkeley Earth (by city) | `berkeleyearth/climate-change-earth-surface-temperature-data` v2 | 8,599,212 | 1743-11-01 → 2013-09-01 | 1961–90 normals, warming (156/185 matched) |
| US drought | `cdminix/us-drought-meteorological-data` v5 | 23,841,468 | 2000–2020, 3 108 US counties | auxiliary benchmark (400 counties) + transfer; never joined to capitals |

Harmonised panel: 160,497 observed site-days (161,908 grid rows). Columns of every dataset: `data/manifests/dataset_quality.json`.

## C. Models trained

128 neural runs (measured training time 6.5 GPU-hours), by stage: tuning 41, ablation 33, final 24, horizons 12, multimodal 6, lookback 6, spatial 6. Classical: persistence, majority, site-month climatology, logistic regression, random forest, HistGradientBoosting, MLP (h = 1; logistic regression and HistGB also for h = 3, 7). Auxiliary drought GRU. Details: `docs/experiment_log.md`.

## D. Results (validation used for selection; held-out test reported once)

| model                                 | modalities                |   n_runs | VAL macro-F1   |   TEST acc. |   TEST macro-F1 |   ± s.d. |   TEST bal. acc. |   TEST ROC-AUC |
|:--------------------------------------|:--------------------------|---------:|:---------------|------------:|----------------:|---------:|-----------------:|---------------:|
| Persistence (today's class)           | label history             |        1 | —              |      0.7976 |          0.7265 |   0      |           0.727  |         0.8218 |
| Majority class                        | —                         |        1 | —              |      0.6144 |          0.1903 |   0      |           0.25   |         0.5    |
| Site-month climatology                | —                         |        1 | —              |      0.614  |          0.5192 |   0      |           0.5859 |         0.7272 |
| Logistic regression                   | C+A+D (aggregated)        |        1 | 0.7291         |      0.8015 |          0.7297 |   0      |           0.727  |         0.9384 |
| Random forest                         | C+A+D (aggregated)        |        1 | 0.7405         |      0.8059 |          0.7398 |   0      |           0.7392 |         0.9445 |
| Gradient boosting (HistGB)            | C+A+D (aggregated)        |        1 | 0.7397         |      0.8023 |          0.7337 |   0      |           0.7436 |         0.9444 |
| MLP (sklearn)                         | C+A+D (aggregated)        |        1 | 0.7181         |      0.7922 |          0.7004 |   0      |           0.6852 |         0.9357 |
| MAF-Net (proposed, attention fusion)  | C+A+D+ctx                 |        5 | 0.7422         |      0.8063 |          0.7414 |   0.0026 |           0.747  |         0.941  |
| MAF-Net w/o attention (concat fusion) | C+A+D+ctx                 |        5 | 0.7406         |      0.8012 |          0.7375 |   0.0038 |           0.7468 |         0.9391 |
| GRU (tuned)                           | C+A+D (single stream)+ctx |        5 | 0.741          |      0.7982 |          0.7391 |   0.0029 |           0.7536 |         0.9437 |
| Transformer encoder (tuned)           | C+A+D (single stream)+ctx |        3 | 0.7438         |      0.8097 |          0.7403 |   0.0019 |           0.7377 |         0.9455 |
| LSTM                                  | C+A+D (single stream)+ctx |        3 | 0.739          |      0.7934 |          0.7316 |   0.0102 |           0.7499 |         0.9415 |
| Vanilla RNN                           | C+A+D (single stream)+ctx |        3 | 0.7399         |      0.7983 |          0.7358 |   0.0045 |           0.7517 |         0.9419 |

Auxiliary US drought benchmark (test 2019–20): macro-F1 0.348 (majority 0.213).

## E. Selected model

* Run `final-mafnet-4ac9637255`: MAF-Net, the seed with the highest validation macro-F1 among five seeds.
* Climate encoder transformer, hidden 32, d_model 32, heads 2, dropout 0.3, modality dropout 0.0, loss focal, lr 0.001, batch 256, look-back 14, horizon 1.
* Trainable parameters: 33,268. Checkpoint `models/best/mafnet_best.pt`; record `models/best/model_record.json`; frozen predictions `models/best/predictions_val.npz`, `predictions_test.npz`.
* Test: accuracy 0.8070, balanced accuracy 0.7436, macro P/R/F1 0.7416/0.7436/0.7415, weighted F1 0.8084, ROC-AUC 0.9411, PR-AUC 0.7871; 95 % site-block bootstrap CI for macro-F1 0.7139–0.7616.
* Protocol and disclosure: selection used validation data only, and the frozen configurations were scored once on 2026-06-03 → 2026-10-12. The first evaluation run stopped with a feature-shape error inside the classical-baseline block, before any neural model was scored. The h = 3/7 classical baselines had inadvertently been trained with the optional transfer features included; they were retrained on the same training data with the documented feature set and the evaluation was rerun. No architecture, feature, label or hyper-parameter choice was changed after seeing test results.

## F. Deliverables

* Notebooks: `notebooks/01_dataset_acquisition.ipynb` … `notebooks/11_final_results_and_inference.ipynb`, `notebooks/validation_manifest.json`
* README: `README.md`; docs: `docs/*.md`
* Figures: `results/figures/` (data behind figures: `results/figures/data/`)
* Metrics: `results/metrics/test_metrics.json`, `results/metrics/selected_model_test.json`; tables: `results/tables/`
* Checkpoint: `models/best/mafnet_best.pt` (other checkpoints stay local in `models/checkpoints/`, git-ignored)
* Report: `reports/final/ClimateSenseAI_IEEE_Final_Report.docx`, `reports/final/ClimateSenseAI_IEEE_Final_Report.pdf`

## G. GitHub

* Repository: https://github.com/Debshata/ClimateSenseAI.git, branch `main`
* Push: pushed successfully; remote main = 75a67c6 verified with git ls-remote on 2026-10-25 18:16
* Commit history checked for AI co-author trailers before pushing (0 found).

## H. Quality verification

* Tests: `................................                                         [100%]` (pipeline, alignment, leakage, labels, models, metrics, checkpoint reproduction).
* Notebooks: 01 ok, 02 ok, 03 ok, 04 ok, 05 ok, 06 ok, 07 ok, 08 ok, 09 ok, 10 ok, 11 ok.
* Leakage controls: window-admission rule, training-only scalers, past-only rolling features, no oversampling (unit-tested).
* Report: generated from artefacts, converted by MS Word to PDF and inspected (two-column IEEE A4).

## I. Limitations

See `docs/limitations.md`. Key points: no observed SPEI, soil moisture or NDVI (dryness is a proxy); one daily snapshot at a longitude-dependent local hour; provider PM2.5 regime shifts; project-defined labels without external validation; small gains over persistence that did not transfer to unseen cities; a single four-month test period.

## J. Reproduction commands

```bash
python -m venv .venv && .venv/Scripts/python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/python -m pip install -r requirements-dev.txt
python scripts/download_datasets.py
python scripts/prepare_data.py --force && python scripts/inspect_datasets.py && python scripts/analyze_labels.py
python scripts/train_baselines.py --horizon 1   # then --horizon 3 / 7 --only hist_gradient_boosting logreg
python scripts/train_drought_aux.py
for s in neural_baselines tune seeds ablation horizons lookback spatial; do python scripts/run_experiments.py --stage $s; done
python scripts/evaluate_models.py && python scripts/predict.py --export
python scripts/explain_model.py && python scripts/validity_analysis.py && python scripts/generate_figures.py
python scripts/build_notebooks.py && python scripts/generate_report.py && python scripts/write_docs.py
python -m pytest
```