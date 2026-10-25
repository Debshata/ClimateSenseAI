# Limitations

These limitations bound every result reported in this repository and in the IEEE report.

## Data

1. **Short history.** The weather repository spans 2024-05-16 → 2026-10-12 (≈ 2.4 years). Each
   chronological split therefore covers different seasons: validation is January–June 2026 and
   test is June–October 2026 (northern summer). Inter-annual variability is barely sampled.
2. **Capital-city sampling.** 185 capital-city sites are retained; they are not representative of
   rural areas, secondary cities or within-city variation. The provider occasionally substitutes
   a nearby town for a capital; such fragments were excluded rather than stitched together.
3. **Single daily snapshot at a fixed UTC instant.** Each site has one observation per day, taken
   at a local hour that depends on longitude (≈ 04–06 h in the Americas, ≈ 17 h in East Asia).
   Heat, humidity and VPD are therefore measured at different phases of the diurnal cycle, which
   biases heat-stress labels by region (tropical American capitals have almost no heat-driven
   High/Critical days).
4. **Air-quality source.** Pollutant values are instantaneous provider estimates, not regulatory
   24-h/8-h means, and show abrupt regime shifts (around 2024-09 and 2025-10) that probably
   reflect changes in the provider's model. Comparisons with WHO guideline levels are approximate.
5. **Missing scientific modalities.** Because the project was restricted to four Kaggle datasets,
   no observed SPEI, soil moisture (GWETTOP/GWETROOT) or vegetation index (NDVI/EVI) is available
   for the capital cities. The "drought, soil and vegetation" branch of the proposal is realised
   as an **atmospheric dryness proxy** (VPD and precipitation-occurrence history from the snapshot
   precipitation field). No synthetic substitutes were generated; ablations for soil and
   vegetation are **not evaluated**.
6. **Geographic mismatch of the drought dataset.** The US drought data cover US counties only.
   They are used only for an auxiliary benchmark and for transferring a learned function; the
   transferred features suffer from domain shift (daily totals vs snapshot precipitation,
   US climates vs global capitals).
7. **Berkeley Earth** ends in 2013 and is monthly; it provides climatological context only.

## Target definition

8. **Project-defined labels.** The four classes are a researcher-designed compound score built
   from published heat-index categories, WHO air-quality guideline levels and a researcher-defined
   dryness band. They are not an established habitability standard and say nothing directly
   about health outcomes or whether a place is habitable.
9. **Circularity.** Labels are deterministic functions of observed weather/air-quality variables,
   and similar variables are model inputs. The task is honest *forecasting* (inputs end the day
   before the target), but the model learns to anticipate a rule, not an independent outcome.
   Same-day reconstruction would be trivial and is not reported as a result.
10. **No external validation.** EM-DAT (or any outcome dataset) was not available, so the
    classes were checked only for internal robustness, temporal stability and regional
    plausibility.

## Modelling and evaluation

11. **Persistence dominates.** Day-to-day label persistence is high (≈ 76 % of consecutive days),
    so a large part of any model's accuracy is explained by persistence; improvements over it
    are the relevant measure of skill.
12. **Single chronological test period** (≈ 4 months). Rolling-origin evaluation over multiple
    years was not possible with 2.4 years of data. Seasonal composition differs between splits.
13. **Spatial generalisation** was tested with one random 20 % held-out-site split (3 seeds),
    not with climate-zone stratification.
14. **Bounded tuning.** Hyper-parameter search used 24 (MAF-Net) / 10 / 10 Optuna trials with a
    capped number of epochs; better configurations may exist for every model, including baselines.
15. **Statistics.** Confidence intervals and paired differences come from a site-block bootstrap
    on a single test period; they are not corrected for multiple comparisons and do not account
    for temporal autocorrelation across sites.
16. **Explainability.** Attention weights, permutation importance and integrated gradients
    describe model behaviour; they are not causal explanations, and correlated features share
    importance.
