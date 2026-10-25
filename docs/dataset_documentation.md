# Dataset Documentation

All data come from the **four Kaggle datasets specified for the project** (the user restricted the
project to these sources). They are downloaded with `kagglehub` by
[`scripts/download_datasets.py`](../scripts/download_datasets.py); provenance (handle, version,
file list, byte sizes, download time) is in `data/manifests/download_manifest.json`, schema
profiles in `data/manifests/dataset_quality.json`, cleaning/join statistics in
`data/manifests/data_audit.json`. Raw files are **not** committed to Git (see `.gitignore`).

## A. World Weather Repository — main spatio-temporal anchor

| Item | Value |
|---|---|
| Kaggle handle | `nelgiriyewithana/global-weather-repository` |
| Version used | **1105** (the latest version, 1106, was published as an empty 22-byte archive on the download date; the script records skipped empty versions and falls back automatically) |
| File | `GlobalWeatherRepository.csv` (45.0 MB) |
| Records | 170 733 rows × 41 columns |
| Temporal coverage | 2024-05-16 → 2026-10-12 (local time stamps, one snapshot per location per day) |
| Spatial coverage | 211 raw country strings, 268 location names (capital cities, plus provider substitutions and a few multilingual one-off records) |
| Variables | temperature, feels-like, humidity, wind, gust, pressure, precipitation, cloud, visibility, UV index, CO, O3, NO2, SO2, PM2.5, PM10, US-EPA index, GB-DEFRA index, astronomy fields |
| Licence | as stated on the Kaggle dataset page (not redistributed here) |

**Quality issues found and handled**

* `-9999` sentinels in CO, SO2 and negative PM10 (−1848.15) → set to missing; 21 values in total
  violated physically plausible ranges (`VALID_RANGE`).
* 1 exact duplicate row; 416 extra same-day records were averaged into one site-day.
* Location instability: the provider sometimes substitutes a nearby place name (e.g. Rangoon /
  Yangon) or switches to a different town for months (e.g. Costa Rica: San Ignacio → San José →
  San Andrés). A **site** is therefore defined as *(country string, lat, lon rounded to 0.1°)*:
  cosmetic renames at identical coordinates are merged, genuinely different places are not.
* 279 sites were found; **185 sites with ≥ 500 observed days are retained** (160 497 site-days).
  The other 94 are short fragments, one-off multilingual records (e.g. "Bélgica", "火鸡") or
  substituted towns; they are listed in `data/manifests/site_selection.csv`.
* The site-day grid has 161 908 rows, of which 1 412 are unobserved calendar days.
* **Observation-time bias:** snapshots are taken at roughly the same UTC instant, so the local
  observation hour ranges from ≈ 04 h (Americas) to ≈ 17 h (East Asia). This biases heat-related
  variables by longitude (see `docs/label_definition.md` §5).
* **Air-quality regime shifts:** cross-site median PM2.5 shows abrupt level changes around
  2024-09 and 2025-10 (`results/figures/eda/temporal_trends.png`), consistent with changes in the
  provider's air-quality model. This non-stationarity affects the chronological evaluation.
* Air-quality values are instantaneous model/provider estimates, not regulatory 24-h means.

## B. World Cities — geographic harmonisation and population context

| Item | Value |
|---|---|
| Kaggle handle | `juanmah/world-cities` (version 12; SimpleMaps World Cities Basic) |
| File | `worldcities.csv`, 50 250 rows × 11 columns |
| Used fields | `city_ascii`, `country`, `iso3`, `capital`, `lat`, `lng`, `population` |

Join: nearest World Cities entry by **haversine distance** within 50 km (coordinates decide; names
are only used as a check, which prevents false joins between same-named cities in different
countries). Result: 182 / 185 sites matched (median distance 1.3 km); 3 unmatched (provider
coordinates far from any listed city: "National" (Bolivia), "Bras" (Brazil), Tarawa (Kiribati));
40 matches had a second candidate within 2 km (flagged `wc_ambiguous`, the nearest is used);
9 country-name disagreements are naming variants only (e.g. Swaziland/Eswatini, Burma/Myanmar,
Macedonia/North Macedonia). Population is used **only as optional exposure context**
(log10 population), never as a hazard.

## C. Berkeley Earth Surface Temperature — long-term thermal context

| Item | Value |
|---|---|
| Kaggle handle | `berkeleyearth/climate-change-earth-surface-temperature-data` (version 2) |
| File used | `GlobalLandTemperaturesByCity.csv`, 8 599 212 rows (monthly, 1743-11 → 2013-09); the by-country, by-state, by-major-city and global files were profiled only |
| Derived | per Berkeley city: 1961–1990 monthly normals (≥ 80 % complete), annual normal, seasonal amplitude, 2000–2012 warming relative to 1961–1990 |

3 510 Berkeley cities have valid normals. Each site is matched to the nearest Berkeley series
within 150 km (156 / 185 matched, median 73 km — Berkeley city series are gridded, hence the
distance). Derived site features: `temp_anom_bk` (snapshot temperature minus the calendar-month
normal) and three static context features. Berkeley data end in 2013 and are **never treated as
contemporaneous daily observations**; the snapshot-vs-monthly-mean difference also contains the
diurnal/observation-hour bias described above.

## D. US Drought & Meteorological Data — auxiliary benchmark only

| Item | Value |
|---|---|
| Kaggle handle | `cdminix/us-drought-meteorological-data` (version 5) |
| Files | `train_timeseries.csv` (19 300 680 rows, 2000–2016), `validation_timeseries.csv` (2 268 840, 2017–2018), `test_timeseries.csv` (2 271 948, 2019–2020), `soil_data.csv` (3 109 counties × 32) |
| Content | daily NASA POWER weather for 3 108 US counties (FIPS) + weekly US Drought Monitor score (0–5) |

**This dataset is never joined to capital-city records** (US counties ≠ global capitals). It is
used for (i) an auxiliary drought-classification benchmark on 400 randomly selected counties
(None / D0–D1 / D2 / D3–D4 from the preceding 30 days of weather, using the dataset's own
chronological splits) and (ii) a transfer-learning experiment: an encoder trained on the five
variables shared with the weather repository (T, RH, specific humidity, precipitation, wind) is
applied to each site's own past 30 days, and its drought-class probabilities are tested as
additional dryness-branch features. Results: `experiments/drought_aux/drought_aux_results.json`.

## Modalities actually available

| Modality in the proposal | Status in this study |
|---|---|
| Meteorology / heat-humidity | **Observed** (World Weather Repository) |
| Air quality (PM2.5, PM10, O3, NO2, CO, SO2) | **Observed** (provider estimates) |
| Drought (SPEI) | **Not available** — replaced by an observed atmospheric dryness proxy (VPD, precipitation-occurrence history); plus transferred US-drought representation |
| Soil moisture (GWETTOP/GWETROOT) | **Not available** for capital cities |
| Vegetation (NDVI/EVI) | **Not available** |
| Long-term climatology | Berkeley Earth (monthly normals to 2013) |
| Geography & population | World Cities |

NASA POWER, SPEIbase, MODIS/AppEEARS, OpenAQ and EM-DAT were part of the original proposal but
were **not used** because the project was restricted to the four Kaggle datasets above. No
synthetic NDVI, SPEI or soil moisture was generated. The implemented model is therefore a
**reduced-modality** realisation of the proposed framework; its dryness branch accepts real SPEI,
soil-moisture or NDVI features without architectural change.
