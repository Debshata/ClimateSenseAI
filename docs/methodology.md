# Methodology

This document follows the submission format required by the assignment: objective, approach,
modifications/enhancements, and how effectiveness is validated. Numerical results are in
`docs/final_execution_status.md`, the README and the IEEE report; every number there is generated
from files in `results/` and `experiments/`.

## 1. Objective

Forecast, one day ahead (secondary: 3 and 7 days), a **project-defined four-class compound
environmental habitability-risk level** (Low / Moderate / High / Critical) for capital-city sites
from the preceding 14 days of meteorological, air-quality and dryness observations, and test
whether a **multimodal attention-fusion network** identifies compound conditions better than
(a) models built on individual modalities, (b) the same network without attention, and
(c) naive, classical and single-stream deep-learning baselines.

Hypotheses: **H1** multimodal > unimodal; **H2** attention fusion > concatenation fusion;
**H3** learned models > persistence forecast.

## 2. Approach (pipeline)

1. **Acquisition** — four Kaggle datasets via `kagglehub` with provenance manifest
   (`scripts/download_datasets.py`).
2. **Audit and EDA** — schemas, missingness, duplicates, sentinels, coverage, distributions,
   observation-hour and air-quality regime analysis (`scripts/inspect_datasets.py`).
3. **Harmonisation** — sites = (country, coordinates rounded to 0.1°); same-day aggregation;
   ≥ 500 observed days; continuous daily grid; haversine joins to World Cities (≤ 50 km) and
   Berkeley Earth (≤ 150 km) (`src/climatesense/data/harmonize.py`).
4. **Features** — physically based (heat index, wet-bulb, dew point, VPD, Berkeley temperature
   anomaly) and past-only temporal features (`src/climatesense/features/engineering.py`).
5. **Labels** — hazard scores from published thresholds, compound aggregation, sensitivity
   analysis before modelling (`docs/label_definition.md`).
6. **Splitting** — common calendar cut-offs 70/15/15 %; a sample is admitted only if its first
   input day and its target day are in the same split (implicit L + h embargo); scalers fitted
   on training rows only (`src/climatesense/data/windows.py`).
7. **Models** — naive (persistence, majority, site-month climatology), classical
   (logistic regression, random forest, HistGradientBoosting, MLP on 242 window aggregates),
   single-stream sequence models (RNN, LSTM, GRU, Transformer), and the proposed MAF-Net.
8. **Tuning** — Optuna TPE + median pruning on validation macro-F1 (24 / 10 / 10 trials for
   MAF-Net / GRU / Transformer).
9. **Multi-seed retraining** — 5 seeds (MAF-Net, MAF-Net-concat, GRU), 3 seeds (Transformer,
   LSTM, RNN); selection of the MAF-Net checkpoint by validation macro-F1.
10. **Single held-out test evaluation** of all frozen configurations (`scripts/evaluate_models.py`):
    metrics, site-block bootstrap CIs, paired bootstrap differences.
11. **Ablations** (12 distinct configurations × 3 seeds), **horizons** (3, 7 days), **look-back** (7, 30 days),
    **spatial hold-out** (20 % unseen sites), **auxiliary US-drought benchmark and transfer**.
12. **Explainability** — permutation importance (features, modality groups, context),
    integrated gradients, attention by class, case studies, risk timelines.

## 3. Proposed model: ClimateSense MAF-Net

See `docs/model_architecture.md` for shapes and parameter counts.

* Branch A — climate sequence (17 features + missing flag) → GRU / LSTM / Transformer encoder.
* Branch B — air-quality sequence (9 + flag) → GRU encoder.
* Branch C — dryness-proxy sequence (9 + flag) → GRU encoder.
* Branch D — static context (8) → MLP.
* Projection to a common d-dimensional token space + LayerNorm + learned modality embedding.
* Optional masked self-attention among modality tokens.
* Learned-query multi-head cross-attention pooling with a **modality key-padding mask**
  (a modality whose inputs are missing for the whole window is excluded, not imputed).
* Gated residual between the attended vector and the masked mean token.
* MLP head → 4 logits → softmax.

## 4. Modifications and enhancements relative to standard approaches

| Enhancement | Purpose | Tested by |
|---|---|---|
| Branch-wise temporal encoders per modality | modality-specific dynamics | unimodal and leave-one-out ablations |
| Learned-query cross-attention over modality tokens with key-padding masks | adaptive weighting, principled missing-modality handling | concat-fusion variant (5 seeds) and ablation |
| Gated residual fusion | stabilise attention pooling | part of full model |
| Modality dropout | robustness to missing / unreliable modalities, regularisation | `no_modality_dropout` ablation |
| Missingness flags per modality | distinguish observed vs. imputed days | — |
| Cross-domain transferred drought representation (US benchmark → capitals) | inject drought knowledge without joining mismatched records | `with_transferred_drought_features` ablation |
| Leakage-safe window admission rule + training-only scalers | honest future evaluation | unit tests `tests/test_data_leakage.py` |
| Site definition by coordinates (not names) | prevents false joins and merges cosmetic renames | `tests/test_data_alignment.py` |

## 5. Validation of effectiveness

* **Primary metric:** macro-F1 (minority High/Critical classes); also accuracy, balanced
  accuracy, macro precision/recall, weighted F1, OvR ROC-AUC and PR-AUC, log-loss, Brier,
  quadratic κ, ordinal error.
* **Selection only on validation;** the test period (2026-06-03 → 2026-10-12) is evaluated once.
* **Comparison against persistence**, which is strong because labels persist day to day.
* **Uncertainty:** mean ± s.d. over seeds; site-block bootstrap CIs; paired bootstrap of
  macro-F1 differences on identical samples.
* **Generalisation:** forecast horizons 3/7 days; unseen sites; look-back sensitivity.
* **Sanity:** automated tests for leakage, alignment, labels, model shapes, attention masks,
  metrics and checkpoint reproducibility.
* **Honesty:** if a simpler model matches or beats MAF-Net, it is reported as such.
