# Compound Habitability-Risk Label — Technical Definition

> **Status:** fixed on 2026-10-25 *before* any model was trained or selected. Thresholds were
> chosen from published guidance or stated a priori; none was adjusted using model performance.
> Code: [`src/climatesense/labeling/compound_risk.py`](../src/climatesense/labeling/compound_risk.py),
> parameters: [`configs/preprocessing.yaml`](../configs/preprocessing.yaml) (`labels:` block),
> analysis: [`scripts/analyze_labels.py`](../scripts/analyze_labels.py) →
> `results/analysis/label_sensitivity.json`.

The four classes — **Low (0), Moderate (1), High (2), Critical (3)** — are a *project-defined*
ordinal scale of concurrent environmental stress at a site on a day. They are **not** an
internationally standardised habitability rating and do not state that a city is or will become
uninhabitable. They are computed only from the **observed** (never imputed or forward-filled)
values of a site-day; days without an observation receive no label.

## 1. Inputs

| Symbol | Variable (World Weather Repository column) | Unit |
|---|---|---|
| T | `temperature_celsius` | °C |
| RH | `humidity` | % |
| PM2.5, PM10, O3, NO2 | `air_quality_PM2.5`, `air_quality_PM10`, `air_quality_Ozone`, `air_quality_Nitrogen_dioxide` | µg/m³ (provider) |
| P | `precip_mm` (snapshot value, used only as "precipitation recorded: yes/no") | mm |

Sentinel values (−9999) and physically impossible values (e.g. negative PM10) are set to missing
(`VALID_RANGE` in `src/climatesense/data/harmonize.py`).

## 2. Hazard scores (each 0–3)

### 2.1 Heat stress H — published thresholds

Heat index HI (US NWS algorithm: Rothfusz regression with the NWS low/high-humidity adjustments;
the simple Steadman formula is used where `(HI_simple + T)/2 < 80 °F`, exactly as in the NWS
procedure, so the regression is never extrapolated to cool conditions):

| HI (°C) | NWS category | score |
|---|---|---|
| < 27 | — | 0 |
| 27–32 | Caution | 1 |
| 32–41 | Extreme caution | 2 |
| ≥ 41 | Danger / Extreme danger | 3 |

Wet-bulb temperature Tw (Stull, 2011 empirical formula, valid for RH 5–99 %, −20 to 50 °C):
Tw ≥ 28 °C → at least 2; Tw ≥ 31 °C → 3 (≈ the empirically observed limit of compensable heat
stress for young adults; cf. Raymond et al., 2020, who discuss the 35 °C theoretical limit).

**H = max(HI score, Tw score)**

### 2.2 Air-quality stress A — WHO 2021 guideline levels

| Pollutant | score 1 if > | score 2 if ≥ | score 3 if ≥ | Basis |
|---|---|---|---|---|
| PM2.5 | 15 (AQG 24 h) | 37.5 (IT-3) | 75 (IT-1) | WHO 2021 |
| PM10 | 45 (AQG 24 h) | 100 (IT-2) | 150 (IT-1) | WHO 2021 |
| O3 | 100 (AQG 8 h) | 160 (IT-1) | 240 (EU alert threshold, 1 h) | WHO 2021 / EU 2008/50/EC |
| NO2 | 25 (AQG 24 h) | 50 (IT-2) | 120 (IT-1) | WHO 2021 |

(`band()` uses `≥ edge` for every boundary.) **A = max over available pollutants**; missing if all
four are missing. *Caveat:* the provider supplies instantaneous concentrations, whereas WHO
guidelines refer to 24-h / 8-h means; the comparison is therefore an approximation. The WHO-based
score agrees exactly with the provider's own US-EPA index category mapping on 51 % of site-days
(Spearman ρ = 0.63), see §5.

### 2.3 Hydro-climatic dryness D — researcher-defined proxy

No observed SPEI, soil-moisture or vegetation product is part of the permitted datasets (the US
drought dataset covers US counties only and cannot be aligned with capital cities). The dryness
hazard is therefore a **proxy** built from observed atmospheric variables:

* Vapour-pressure deficit VPD = e_s(T)·(1 − RH/100), with e_s(T) = 0.6108·exp(17.27T/(T+237.3)) kPa.
  Bands (researcher-defined, informed by plant-physiology literature on VPD stress, e.g.
  Grossiord et al., 2020): < 1.0 → 0; 1.0–2.0 → 1; 2.0–3.0 → 2; ≥ 3.0 → 3.
* Dry-spell modifier: +1 (capped at 3) if **no precipitation was recorded on any observed day of
  the trailing 30 days** and at least 20 of those days were observed.

**D = min(VPD band + dry-spell, 3)**. This is *not* SPEI, soil moisture or NDVI and is reported
as a reduced-modality substitute throughout.

## 3. Compound aggregation

With S = H + A + D (missing A or D contribute 0), m = max(H, A, D) and n₂ = #{hazards ≥ 2}:

```
Critical (3) : S ≥ 6            or (m = 3 and n₂ ≥ 2)
High     (2) : S ≥ 4  or m = 3  or n₂ ≥ 2
Moderate (1) : S ≥ 2  or m ≥ 2
Low      (0) : otherwise
```

The class reflects both the **number** and the **severity** of concurrent stresses, as required by
the problem-identification document. The label is undefined if H is missing (no temperature or
humidity). Label coverage (share of the three hazards available) is stored per site-day in
`label_coverage`; it is 1.0 on all labelled days of the retained sites.

## 4. Resulting distribution (observed site-days, 185 sites)

| Class | Low | Moderate | High | Critical |
|---|---|---|---|---|
| All (160 496 days) | 52.4 % | 25.8 % | 12.1 % | 9.8 % |
| Train (2024-05-16 → 2026-01-21) | 47.6 % | 27.9 % | 13.6 % | 10.9 % |
| Validation (2026-01-22 → 2026-06-02) | 65.0 % | 20.7 % | 7.5 % | 6.8 % |
| Test (2026-06-03 → 2026-10-12) | 61.8 % | 21.3 % | 9.3 % | 7.6 % |

Among High/Critical days, heat ≥ 2 occurs on 49 %, air ≥ 2 on 56 % and dryness ≥ 2 on 70 %.

## 5. Validity checks (`results/analysis/label_sensitivity.json`)

| Variant | Quadratic-weighted κ vs. base | Exact agreement |
|---|---|---|
| Class boundaries −1 (S thresholds 1/3/5) | 0.82 | 0.63 |
| Class boundaries +1 (S thresholds 3/5/7) | 0.94 | 0.89 |
| Heat-index bands −1 °C | 0.98 | 0.96 |
| Heat-index bands +1 °C | 0.99 | 0.97 |
| VPD bands × 1.25 | 0.96 | 0.93 |
| Air score from provider US-EPA index | 0.88 | 0.79 |
| Without dryness | 0.77 | 0.75 |
| Without air quality | 0.79 | 0.75 |

* **Robustness:** small threshold perturbations change few labels (κ ≥ 0.94); removing a whole
  modality changes the label substantially (κ ≈ 0.77–0.79), i.e. every hazard contributes.
* **Temporal persistence:** the next-day label equals today's label on 75.6 % of consecutive
  observed day pairs — any forecaster must be compared against this persistence rule.
* **Regional plausibility:** highest High/Critical shares: Riyadh (97 %), New Delhi (95 %),
  Khartoum (95 %), Jakarta (93 %), Kuwait City (91 %), Doha, Kuala Lumpur, Muscat, Manama, Dhaka;
  lowest (0 %): Buenos Aires, Suva, Quito, Copenhagen, Oslo, Dublin, Wellington, …
* **Observation-hour bias (important):** the provider snapshot is taken at nearly the same UTC
  instant for all cities, so the *local* hour of the observation depends on longitude (median
  ≈ 04–06 h in the Americas, ≈ 10 h in Europe/Africa, ≈ 12 h in western Asia, ≈ 17 h in East
  Asia/Oceania). Heat (and VPD) are therefore measured near the daily minimum in the Americas and
  near the maximum in Asia. Tropical American capitals (e.g. Panama City, Managua) consequently
  have almost no heat-driven High/Critical days. This is a measurement-design bias of the source
  data; the label was not altered post hoc, the local observation hour is provided to the models
  as an input feature, and the bias is reported as a limitation.
* **External event validation (EM-DAT):** not performed — EM-DAT is not among the permitted
  datasets, so external validity of the classes remains untested.

## 6. Use in the forecasting task

The model input for a sample ending on day *t* contains only days *t−13 … t* (and the hazard
scores computed from those days). The target is the class observed on day *t+h* (h = 1 primary;
3 and 7 secondary). Because the label is a deterministic function of next-day weather and air
quality, the task is genuine short-range forecasting of a rule-defined state, not reconstruction
of today's label; persistence is the reference that skill must beat.

## References

* NOAA/NWS, *The Heat Index Equation* (Rothfusz 1990, NWS Technical Attachment SR 90-23).
* R. Stull, "Wet-bulb temperature from relative humidity and air temperature," *J. Appl. Meteor. Climatol.*, 50(11), 2267–2269, 2011, doi:10.1175/JAMC-D-11-0143.1.
* C. Raymond, T. Matthews, R. M. Horton, *Science Advances* 6(19), eaaw1838, 2020, doi:10.1126/sciadv.aaw1838.
* WHO, *WHO Global Air Quality Guidelines*, Geneva, 2021.
* C. Grossiord et al., "Plant responses to rising vapor pressure deficit," *New Phytologist* 226(6), 1550–1566, 2020, doi:10.1111/nph.16485.
