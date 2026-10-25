"""Generate and execute the eleven project notebooks (nbformat + nbclient).

Presentation notebooks load stored artefacts (checkpoints, logs, predictions) instead of
retraining; every number shown is computed when the notebook executes.
    python scripts/build_notebooks.py            # build + execute all
    python scripts/build_notebooks.py --no-exec  # build only
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"

HEADER = '''import sys, json, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib
import matplotlib.pyplot as plt
from IPython.display import Image, display, Markdown
pd.set_option("display.max_columns", 60); pd.set_option("display.width", 220)
from climatesense.utils import read_json
import platform, torch, sklearn
print("Executed:", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), "| Python", platform.python_version(),
      "| torch", torch.__version__, "| sklearn", sklearn.__version__, "| pandas", pd.__version__)
def show(p, width=850):
    display(Image(filename=str(ROOT / p), width=width))'''


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


def nb01():
    return [
        md("""# 01 — Dataset Acquisition
**ClimateSense AI** · Debshata Choudhury (A2305223104) · Generative AI, Amity University

**Objective.** Download the four Kaggle datasets that the project is restricted to, record their
provenance, and inspect the actual files and schemas before any modelling.

| Name | Kaggle handle | Role |
|---|---|---|
| World Weather Repository | `nelgiriyewithana/global-weather-repository` | daily capital-city weather + air quality (anchor) |
| US Drought & Meteorological | `cdminix/us-drought-meteorological-data` | auxiliary drought benchmark / transfer learning |
| World Cities | `juanmah/world-cities` | coordinates, ISO codes, population |
| Berkeley Earth | `berkeleyearth/climate-change-earth-surface-temperature-data` | long-term temperature climatology |

Downloads are performed by `scripts/download_datasets.py` (kagglehub, cached, retried). The Kaggle
token is read from a local, git-ignored `.env`; it is never printed or committed."""),
        code(HEADER),
        md("## 1. Provenance manifest\nThe manifest records handle, resolved version, files and byte sizes."),
        code('''man = read_json(ROOT / "data/manifests/download_manifest.json")
print("downloaded at (UTC):", man["downloaded_at_utc"], "| kagglehub", man["kagglehub_version"])
rows = []
for k, v in man["datasets"].items():
    rows.append(dict(dataset=k, handle=v["handle"], version=v.get("version"), status=v["status"],
                     skipped_empty_versions=v.get("skipped_empty_versions"), n_files=len(v.get("files", [])),
                     total_MB=round(sum(f["bytes"] for f in v.get("files", [])) / 1e6, 1)))
pd.DataFrame(rows)'''),
        md("""The latest version of the weather repository (1106) was published as an empty archive on the
download date; the script detected this and fell back to version 1105, which is recorded above."""),
        code('''for k, v in man["datasets"].items():
    print(k); [print("   ", f["file"], f"{f['bytes']/1e6:.1f} MB") for f in v["files"]]'''),
        md("## 2. Actual schemas (first rows) — no column names are assumed"),
        code('''w = pd.read_csv(ROOT / "data/raw/weather/GlobalWeatherRepository.csv", nrows=3)
print(w.shape[1], "columns"); w.T'''),
        code('''display(pd.read_csv(ROOT / "data/raw/world_cities/worldcities.csv", nrows=3))
display(pd.read_csv(ROOT / "data/raw/historical_temperature/GlobalLandTemperaturesByCity.csv", nrows=3))
display(pd.read_csv(ROOT / "data/raw/drought_us/train_timeseries/train_timeseries.csv", nrows=3))'''),
        md("## 3. Size, coverage and quality summary\nProduced by `scripts/inspect_datasets.py` (full-file scans)."),
        code('''q = read_json(ROOT / "data/manifests/dataset_quality.json")
summ = {
 "weather": dict(rows=q["weather"]["rows"], cols=len(q["weather"]["columns"]), start=q["weather"]["date_min"], end=q["weather"]["date_max"],
                 dup_ids=q["weather"]["duplicate_ids"], countries_raw=q["weather"]["n_countries_raw"]),
 "world_cities": dict(rows=q["world_cities"]["rows"], cols=len(q["world_cities"]["columns"]), dup_ids=q["world_cities"]["duplicate_ids"]),
 "berkeley_by_city": dict(rows=q["berkeley_by_city"]["rows"], cols=len(q["berkeley_by_city"]["columns"]), start=q["berkeley_by_city"]["date_min"],
                          end=q["berkeley_by_city"]["date_max"], cities=q["berkeley_by_city"]["n_cities"]),
}
for p, v in q["drought_us_timeseries"]["splits"].items():
    summ[f"drought_us_{p}"] = dict(rows=v["rows"], start=v["date_min"], end=v["date_max"], counties=v["counties"])
pd.DataFrame(summ).T'''),
        code('''print("Weather -9999 sentinels:", q["weather"]["sentinel_minus9999_counts"])
pd.Series(q["berkeley_by_city"]["missing_pct"]).rename("missing %").to_frame()'''),
        md("""## 4. Scientific sources not used
NASA POWER, CSIC SPEIbase, MODIS MOD13A3/AppEEARS, OpenAQ and EM-DAT were named in the research
proposal, but the project was explicitly restricted to the four Kaggle datasets. Consequently no
observed SPEI, soil-moisture or NDVI is available for the capital cities, and **no synthetic
substitute was generated**. The dryness branch uses an observed atmospheric proxy instead (see
notebook 04), and the model is reported as a reduced-modality realisation."""),
    ]


def nb02():
    return [
        md("""# 02 — Exploratory Data Analysis
Quality audit and exploratory statistics of the harmonised site-day panel (185 capital-city
sites). Extreme values are **kept** unless physically impossible — extremes are the events the
system must detect."""),
        code(HEADER),
        code('''panel = pd.read_parquet(ROOT / "data/processed/panel.parquet")
obs = panel[panel.observed == 1]
print("site-days:", len(obs), "| sites:", obs.site_id.nunique(), "| dates:", obs.date.min().date(), "→", obs.date.max().date())
cols = ["temp_c_obs","rh_obs","wind_kph_obs","pressure_hpa_obs","precip_mm_obs","uv_index_obs","pm25_obs","pm10_obs","o3_obs","no2_obs","co_obs","so2_obs"]
obs[cols].describe(percentiles=[.01,.5,.99]).T.round(2)'''),
        md("### Missingness (site-day grid, including unobserved calendar days)"),
        code('''(panel[cols].isna().mean()*100).round(3).rename("% missing").to_frame().T'''),
        code('show("results/figures/eda/missingness_heatmap.png")'),
        md("### Thermal and pollution distributions"),
        code('show("results/figures/eda/thermal_distributions.png"); show("results/figures/eda/pollution_distributions.png")'),
        code('''who = {"pm25_obs":15, "pm10_obs":45, "o3_obs":100, "no2_obs":25}
pd.Series({k: (obs[k] > v).mean()*100 for k, v in who.items()}).round(1).rename("% site-days above WHO AQG").to_frame()'''),
        md("### Correlations\nSpearman correlations are used because pollutant distributions are heavily skewed."),
        code('show("results/figures/eda/correlation_heatmap.png", 650)'),
        md("### Geography and time"),
        code('show("results/figures/eda/site_map_high_critical_share.png"); show("results/figures/eda/temporal_trends.png")'),
        md("""**Observation.** The cross-site median PM2.5 changes level abruptly around 2024-09 and
2025-10. Such step changes are typical of a change in the provider's air-quality model rather
than of the atmosphere, so they are a source of non-stationarity between training and test
periods."""),
        md("### Local observation hour by longitude (sampling bias)"),
        code('''s = pd.read_csv(ROOT / "data/processed/sites.csv").set_index("site_id")
o = obs.assign(lon=obs.site_id.map(s.lon))
o.groupby(pd.cut(o.lon, [-180,-90,-30,30,90,180]), observed=True).obs_hour_obs.describe()[["mean","25%","50%","75%"]].round(1)'''),
        md("""The snapshot is taken at nearly the same UTC instant everywhere, so the local hour runs from
≈04 h in the Americas to ≈17 h in East Asia. Heat-related variables are therefore sampled at
different phases of the diurnal cycle — a key limitation carried through to the labels."""),
        md("### Seasonality, dryness proxy and Berkeley Earth context"),
        code('show("results/figures/eda/seasonal_class_distribution.png"); show("results/figures/eda/dryness_indicators.png"); show("results/figures/eda/berkeley_warming.png", 550)'),
        md("### Auxiliary US drought data"),
        code('show("results/figures/eda/us_drought_score_distribution.png", 500)'),
    ]


def nb03():
    return [
        md("""# 03 — Data Cleaning and Integration
This notebook re-runs the harmonisation functions from `src/climatesense/data/harmonize.py` to
show each step (it does not overwrite the stored panel).

**Steps.** (1) schema normalisation and sentinel removal; (2) site definition
`(country, round(lat,0.1°), round(lon,0.1°))`; (3) same-day aggregation; (4) site selection
(≥ 500 observed days); (5) continuous daily grid; (6) spatial joins to World Cities and Berkeley
Earth by haversine distance:

$$ d = 2R\\,\\arcsin\\sqrt{\\sin^2\\tfrac{\\Delta\\varphi}{2} + \\cos\\varphi_1\\cos\\varphi_2\\sin^2\\tfrac{\\Delta\\lambda}{2}},\\quad R = 6371\\ \\text{km}.$$"""),
        code(HEADER),
        code('''from climatesense.data import harmonize as H
raw = H.load_weather_raw()
day, audit = H.clean_weather(raw)
print(json.dumps(audit, indent=1))'''),
        code('''kept, stats = H.select_sites(day, 500)
print("sites found:", len(stats), "| kept (>=500 observed days):", stats.kept.sum(), "| site-days kept:", len(kept))
stats.sort_values("n_days").head(8)'''),
        md("Examples of excluded fragments — provider substitutions and one-off multilingual records:"),
        code('''stats[~stats.kept].sort_values("n_days", ascending=False).head(10)[["country","location","n_days","first","last"]]'''),
        md("### Spatial joins (coordinates first, names only as a check)"),
        code('''sites = pd.read_csv(ROOT / "data/processed/sites.csv")
a = read_json(ROOT / "data/manifests/data_audit.json")
display(pd.DataFrame([a["world_cities_join"], a["berkeley_join"]], index=["World Cities", "Berkeley Earth"]))
sites.loc[sites.wc_matched & ~sites.wc_country_agrees, ["country","location","wc_city","wc_country","wc_distance_km"]]'''),
        md("Name disagreements are naming variants (Swaziland/Eswatini, Myanmar/Burma …), confirming the joins."),
        code('''sites.loc[~sites.wc_matched, ["country","location","lat","lon","wc_distance_km"]]'''),
        md("""### Temporal alignment and leakage-safe preprocessing
* The grid is daily and continuous per site; unobserved days have `observed = 0`.
* Model inputs are forward-filled for **at most 3 days** (past information only); labels use the
  raw `_obs` columns and are never imputed.
* Berkeley monthly normals (1961–1990) are static climatology — no look-ahead is possible.
* Standardisation statistics are fitted on **training-period rows only** (`PanelArrays`).
* Chronological split with identical calendar cut-offs for all sites; a sample is used only if its
  first input day *and* its target day fall inside the same split."""),
        code('''panel = pd.read_parquet(ROOT / "data/processed/panel.parquet")
sp = read_json(ROOT / "data/manifests/splits.json"); print(sp)
from climatesense.training.trainer import get_arrays
arr = get_arrays()
for s in ["train","val","test"]:
    ix = arr.sample_index(14, 1, s)
    print(f"{s:5s} samples={len(ix):6d}  first input={arr.date_ns[ix-13].min()}  last target={arr.date_ns[ix+1].max()}")'''),
        md("The gap between consecutive splits (≥ 14 days) is the implicit embargo created by the window rule."),
    ]


def nb04():
    return [
        md("""# 04 — Feature Engineering and Compound-Risk Labels
Full definition: `docs/label_definition.md`.

**Physical features** (from observed T, RH):
* Dew point (Magnus): $T_d = \\frac{b\\,\\gamma}{a-\\gamma},\\ \\gamma=\\ln(RH/100)+\\frac{aT}{b+T},\\ a=17.27,\\ b=237.3$
* Saturation vapour pressure $e_s(T)=0.6108\\,e^{17.27T/(T+237.3)}$ kPa; **VPD** $=e_s(T)(1-RH/100)$
* **Heat index** — NWS Rothfusz regression (with adjustments) where $(HI_{simple}+T)/2\\ge 80^\\circ$F, Steadman simple formula otherwise
* **Wet-bulb temperature** — Stull (2011) empirical formula
* Temperature anomaly vs Berkeley 1961–1990 calendar-month normal
* Past-only precipitation persistence: 7/30-day sums, wet-day fractions, days since recorded precipitation

**Hazard scores** $H, A, D \\in \\{0,1,2,3\\}$ (NWS heat-index categories + wet-bulb; WHO 2021 air-quality
guideline/interim targets; VPD bands + 30-day dry-spell). **Compound class** with
$S = H + A + D$, $m=\\max$, $n_2=\\#\\{\\ge2\\}$:

| Class | Rule |
|---|---|
| Critical | $S\\ge6$ or ($m=3$ and $n_2\\ge2$) |
| High | $S\\ge4$ or $m=3$ or $n_2\\ge2$ |
| Moderate | $S\\ge2$ or $m\\ge2$ |
| Low | otherwise |"""),
        code(HEADER),
        code('''from climatesense.features.engineering import heat_index_c, wet_bulb_stull_c, vpd_kpa, CLIMATE_FEATURES, AIR_FEATURES, DRY_FEATURES, CONTEXT_FEATURES
# sanity checks against published reference values
print("HI(90F, 60%) =", round(heat_index_c((90-32)*5/9, 60)*9/5+32, 1), "F  (NWS table: 100 F)")
print("Tw(20C, 50%) =", round(float(wet_bulb_stull_c(20.0, 50.0)), 2), "C  (Stull 2011: 13.7 C)")
print("VPD(30C, 40%) =", round(float(vpd_kpa(30.0, 40.0)), 2), "kPa")
pd.DataFrame({"climate": pd.Series(CLIMATE_FEATURES), "air": pd.Series(AIR_FEATURES), "dryness": pd.Series(DRY_FEATURES), "context": pd.Series(CONTEXT_FEATURES)}).fillna("")'''),
        md("### Label distribution and hazard composition"),
        code('''a = read_json(ROOT / "data/manifests/data_audit.json")
display(pd.DataFrame(a["label_distribution_by_split"]).T[["Low","Moderate","High","Critical"]].assign(total=lambda d: d.sum(1)))
show("results/figures/labels/hazard_scores.png"); show("results/figures/eda/class_distribution_by_split.png", 600)'''),
        md("### Sensitivity and validity analysis (computed before any model training)"),
        code('''ls = read_json(ROOT / "results/analysis/label_sensitivity.json")
pd.DataFrame({k: dict(kappa_w=v["quadratic_kappa_vs_base"], agreement=v["agreement_with_base"], **v["distribution"]) for k, v in ls["variants"].items()}).T.round(3)'''),
        code('''show("results/figures/labels/label_sensitivity.png")
print("next-day label persistence:", round(ls["next_day_label_persistence"], 3))
print("WHO-based air score vs provider US-EPA index:", ls["air_score_vs_us_epa_index"])
print("drivers of High/Critical:", {k: round(v, 3) for k, v in ls["drivers_high_critical"].items()})
pd.DataFrame(ls["observation_hour_bias_by_longitude"]).T.round(3)'''),
        code('''pd.DataFrame(ls["sites_highest_hc_share"])[["location","country","share"]].round(3).head(10)'''),
        md("""### Validity concerns
1. **Circularity.** Labels are deterministic functions of observed weather and air quality. The
   task is therefore forecasting a rule-defined state one or more days ahead — not discovery of
   an independent ground truth. Today's label persists to tomorrow on ≈ 76 % of days, so the
   persistence forecast is the reference to beat.
2. **Observation-hour bias** (table above) distorts heat stress by longitude.
3. **No external validation** (EM-DAT not permitted) — classes are internally consistent and
   regionally plausible but not validated against health or disaster outcomes.
4. **Proxy dryness**: VPD and precipitation-occurrence history replace SPEI/soil moisture/NDVI."""),
    ]


def nb05():
    return [
        md("""# 05 — Baseline Models
Classical baselines use **aggregated 14-day windows**: for each of the 39 daily features the last
value, mean, max, min, standard deviation and trend (last − first), plus 8 static context features
(242 features). Naive references:

* **Persistence** — tomorrow's class = today's observed class (strong because labels are persistent)
* **Majority class** and **site-month climatology** (most frequent training class for the site and month)

Hyper-parameters (fixed a priori, `scripts/train_baselines.py`): logistic regression (balanced, C=1);
random forest (300 trees, min leaf 5, balanced subsample); HistGradientBoosting (400 iterations,
lr 0.06, 31 leaves, L2 = 1, balanced, early stopping on an internal 10 % split); MLP 256-128
(α = 1e-3, early stopping). All selection uses the **validation** split; test results appear only
in notebook 09. To retrain set `RETRAIN = True`."""),
        code(HEADER),
        code('''RETRAIN = False
if RETRAIN:
    import subprocess; subprocess.run([sys.executable, str(ROOT / "scripts/train_baselines.py")], check=True)
rows = []
for f in sorted((ROOT / "experiments/baseline").glob("baseline-*_h1.json")):
    r = read_json(f); m = r["val_metrics"]
    rows.append(dict(model=r["model"], val_accuracy=m["accuracy"], val_balanced_accuracy=m["balanced_accuracy"], val_macro_f1=m["macro_f1"],
                     val_weighted_f1=m["weighted_f1"], val_roc_auc=m["roc_auc_macro_ovr"], train_time_s=r["train_time_s"], n_params=r["n_params"]))
s = read_json(ROOT / "experiments/baseline/summary_L14_h1.json")
for k in ["persistence_L14_h1", "majority_L14_h1", "climatology_L14_h1"]:
    rows.append(dict(model=k.split("_")[0], val_accuracy=s[k]["val_accuracy"], val_balanced_accuracy=s[k]["val_balanced_accuracy"], val_macro_f1=s[k]["val_macro_f1"]))
pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False).round(4)'''),
        md("### Validation classification report of the strongest classical baseline"),
        code('''from climatesense.labeling.compound_risk import CLASS_NAMES
d = np.load(ROOT / "results/predictions/baseline-hist_gradient_boosting_L14_h1_val.npz")
from sklearn.metrics import classification_report
print(classification_report(d["y"], d["proba"].argmax(1), target_names=CLASS_NAMES, digits=3))'''),
        md("""**Reading.** Gradient boosting and the random forest clearly beat the naive baselines, but the
margin over persistence is modest: most of the predictable signal is that compound-risk states
persist from one day to the next."""),
    ]


def nb06():
    return [
        md("""# 06 — Recurrent and Transformer Sequence Baselines
A sample is a tensor $X\\in\\mathbb{R}^{L\\times F}$ with $L = 14$ days and $F = 39$ standardised
daily features (+3 missingness flags), plus a static context vector $c\\in\\mathbb{R}^{8}$. The
single-stream baselines encode all features jointly:

$$h_t = \\mathrm{GRU}(x_t, h_{t-1}),\\qquad \\hat y = \\mathrm{softmax}(W_2\\,\\mathrm{GELU}(W_1[h_L; c]))$$

(LSTM and vanilla RNN analogously; the Transformer encoder adds sinusoidal positions, pre-norm
self-attention layers and uses [last step; mean over time]). Training: class-weighted
cross-entropy (or focal loss when tuned), AdamW, ReduceLROnPlateau on validation macro-F1,
gradient clipping 1.0, early stopping (patience 6), fixed seeds."""),
        code(HEADER),
        code('''from climatesense.training.trainer import get_arrays, build_model_on_cols, DEFAULTS
arr = get_arrays()
idx = arr.sample_index(14, 1, "train")
X, C, y = arr.windows(idx[:4], 14, 1)
print("window tensor:", X.shape, "| context:", C.shape, "| targets:", y)
for kind in ["rnn", "lstm", "gru", "transformer"]:
    m, cols = build_model_on_cols({**DEFAULTS, "model": kind}, arr)
    print(f"{kind:12s} params = {sum(p.numel() for p in m.parameters()):,}")'''),
        md("### Initial runs (default hyper-parameters, seed 42) — validation"),
        code('''rows = []
for f in sorted((ROOT / "experiments/multimodal").glob("multimodal-*.json")):
    r = read_json(f); c = r["config"]
    rows.append(dict(run=r["run_id"], model=c["model"] + ("" if c["model"] != "mafnet" else f" ({c['fusion']})"), params=r["n_params"],
                     best_epoch=r["best_epoch"], val_macro_f1=r["val_metrics"]["macro_f1"], val_bal_acc=r["val_metrics"]["balanced_accuracy"],
                     val_acc=r["val_metrics"]["accuracy"], train_time_s=round(r["train_time_s"], 1)))
pd.DataFrame(rows).round(4)'''),
        md("### Five-seed results for the tuned/selected sequence baselines (validation)"),
        code('''fs = read_json(ROOT / "experiments/final_val_summary.json")
pd.DataFrame(fs).T[["val_macro_f1_mean", "val_macro_f1_std"]].astype(float).round(4)'''),
        code('''# learning curves of the best-validation GRU seed
ids = fs["gru"]["run_ids"]; recs = [read_json(ROOT / "experiments/final" / f"{r}.json") for r in ids]
r = max(recs, key=lambda r: r["val_metrics"]["macro_f1"]); h = pd.DataFrame(r["history"])
fig, ax = plt.subplots(1, 2, figsize=(9, 3))
ax[0].plot(h.epoch, h.train_nll, label="train"); ax[0].plot(h.epoch, h.val_nll, label="val"); ax[0].set_title("NLL"); ax[0].legend()
ax[1].plot(h.epoch, h.train_macro_f1, label="train"); ax[1].plot(h.epoch, h.val_macro_f1, label="val"); ax[1].set_title("macro-F1"); ax[1].legend()
plt.tight_layout(); plt.show(); print(r["config"])'''),
    ]


def nb07():
    return [
        md("""# 07 — ClimateSense Multimodal Attention Fusion Network (MAF-Net)
**Branches.** Climate (A), air quality (B) and dryness proxy (C) sequences are encoded by separate
temporal encoders $E_m$; static context (D) by an MLP. Each representation is projected to a
common space and tagged with a learned modality embedding:

$$z_m = \\mathrm{LN}(W_m E_m(X_{:,\\mathcal F_m})) + e_m,\\qquad m\\in\\{A,B,C,D\\}.$$

**Masked modality attention.** Optional self-attention among the tokens, then a learned query $q$
pools them with multi-head cross-attention; modalities missing for the whole window are removed
through the key-padding mask $\\mathcal M$ (never replaced by fictitious values):

$$\\alpha = \\mathrm{softmax}\\Big(\\frac{qW_Q (ZW_K)^\\top}{\\sqrt{d}} + \\mathcal M\\Big),\\qquad f=\\alpha ZW_V W_O .$$

**Gated residual.** $g=\\sigma(W_g[f;\\bar z])$, $u = g\\odot f + (1-g)\\odot\\bar z$ with $\\bar z$ the masked
mean of tokens; head: LN → Linear → GELU → Dropout → Linear(4) → softmax.

**Modality dropout** (enhancement) randomly masks whole modalities during training so the model
does not over-rely on one branch and degrades gracefully when a modality is missing.
The `fusion="concat"` variant replaces attention with concatenation (ablation)."""),
        code(HEADER),
        code('show("results/figures/architecture_mafnet.png", 900)'),
        code('''from climatesense.training.trainer import get_arrays, build_model_on_cols, DEFAULTS, feature_sets
sel = read_json(ROOT / "experiments/tuning/selected_configs.json")["mafnet"]
cfg = {**DEFAULTS, **sel["params"], "model": "mafnet"}
print("selected hyper-parameters (validation macro-F1 = %.4f):" % sel["val_macro_f1"]); print(json.dumps(sel["params"], indent=1))
arr = get_arrays()
model, cols = build_model_on_cols(cfg, arr)
fs = feature_sets(cfg, arr)
print({m: len(v) for m, v in fs.items()}, "| total trainable parameters:", f"{sum(p.numel() for p in model.parameters()):,}")
print(model)'''),
        md("### Tensor shapes through the network (forward hooks on a real batch)"),
        code('''import torch
from climatesense.training.trainer import WindowBatcher
b = WindowBatcher(arr, cfg, cols); model.eval()
idx = torch.as_tensor(arr.sample_index(cfg["lookback"], 1, "val")[:32])
x, c, y = b.get(idx.to(b.X.device)); model.to(b.X.device)
shapes = []
for name, mod in list(model.encoders.items()):
    mod.register_forward_hook(lambda m, i, o, n=name: shapes.append((f"encoder[{n}]", tuple(i[0].shape), tuple(o.shape))))
model.ctx_enc.register_forward_hook(lambda m, i, o: shapes.append(("context MLP", tuple(i[0].shape), tuple(o.shape))))
model.cross.register_forward_hook(lambda m, i, o: shapes.append(("cross-attention", tuple(i[1].shape), tuple(o[0].shape))))
model.head.register_forward_hook(lambda m, i, o: shapes.append(("classifier head", tuple(i[0].shape), tuple(o.shape))))
with torch.no_grad(): logits, w = model(x, c)
display(pd.DataFrame(shapes, columns=["module", "input shape", "output shape"]))
print("attention weights per sample:", tuple(w.shape), "rows sum to", w.sum(1)[:3].cpu().numpy())'''),
        md("### Selected run: training dynamics"),
        code('show("results/figures/training_curves.png")'),
        md("""The attention weights above are produced by an actual attention operation over modality
tokens; they indicate how the fusion module weighted the representations, **not** causal
importance (see notebook 10)."""),
    ]


def nb08():
    return [
        md("""# 08 — Hyper-parameter Optimisation
Optuna TPE sampler (seed 42) with median pruning, maximising **validation macro-F1** (look-back 14,
horizon 1, max 20 epochs, patience 5). Budget: 24 trials for MAF-Net, 10 each for the GRU and
Transformer baselines so that baselines are tuned too. Search space:

| Parameter | Values |
|---|---|
| hidden size | 32, 64, 128 |
| layers | 1, 2 |
| dropout | 0.1, 0.2, 0.3 |
| learning rate | 3e-3, 1e-3, 3e-4 |
| batch size | 128, 256 |
| weight decay | 0, 1e-4, 1e-3 |
| loss | CE, class-weighted CE, focal (γ = 2) |
| MAF-Net only: climate encoder | GRU, LSTM, Transformer |
| MAF-Net only: d_model / heads | 32, 64, 128 / 1, 2, 4 |
| MAF-Net only: modality dropout / token self-attention | 0, 0.1, 0.2 / on, off |

Sequence length (7, 14, 30) is studied separately (it changes the sample set), see notebook 10.
The test split is not used at any point of the search."""),
        code(HEADER),
        code('''for m in ["mafnet", "gru", "transformer"]:
    t = pd.DataFrame(read_json(ROOT / f"experiments/tuning/{m}_trials.json"))
    t = pd.concat([t.drop(columns="params"), pd.json_normalize(t.params)], axis=1)
    print(f"{m}: {len(t)} trials ({(t.state=='COMPLETE').sum()} complete, {(t.state=='PRUNED').sum()} pruned); best = {t.value.max():.4f}")
    display(t.sort_values("value", ascending=False).head(8).round(4))'''),
        code('show("results/figures/optuna_trials.png", 700)'),
        code('''sel = read_json(ROOT / "experiments/tuning/selected_configs.json")
pd.DataFrame({k: {**v["params"], "val_macro_f1": round(v["val_macro_f1"], 4)} for k, v in sel.items()}).T'''),
        md("""The selected configurations were then retrained for up to 40 epochs (patience 6) with five
seeds (MAF-Net, MAF-Net-concat, GRU) or three seeds (Transformer, LSTM, RNN)."""),
    ]


def nb09():
    return [
        md("""# 09 — Model Evaluation on the Held-out Test Period
Test period: 2026-06-03 → 2026-10-12 (never used for training, tuning or selection). The selected
MAF-Net checkpoint is the seed with the highest **validation** macro-F1. Confidence intervals use a
site-block bootstrap (1 000 resamples of whole sites) to respect autocorrelation within a site."""),
        code(HEADER),
        code('''mc = pd.read_csv(ROOT / "results/tables/model_comparison_h1.csv")
mc[["model","modalities","n_params","n_runs","val_macro_f1","accuracy","balanced_accuracy","macro_precision","macro_recall","macro_f1","macro_f1_std","weighted_f1","roc_auc_macro_ovr","pr_auc_macro","train_time_s"]].round(4)'''),
        code('show("results/figures/model_comparison.png", 750)'),
        md("### Selected MAF-Net: confusion matrices, ROC and PR curves"),
        code('''st = read_json(ROOT / "results/metrics/selected_model_test.json")
m = st["test_metrics"]; print("selected run:", st["selected_run"])
print({k: round(m[k], 4) for k in ["accuracy","balanced_accuracy","macro_precision","macro_recall","macro_f1","weighted_f1","roc_auc_macro_ovr","pr_auc_macro","log_loss","brier","quadratic_kappa"]})
print("95% CI macro-F1:", np.round(st["ci95_macro_f1_site_block_bootstrap"], 4), "| accuracy:", np.round(st["ci95_accuracy_site_block_bootstrap"], 4))
from climatesense.labeling.compound_risk import CLASS_NAMES
pd.DataFrame(m["per_class"]).T.rename(index=dict(zip(map(str, range(4)), CLASS_NAMES))).round(3)'''),
        code('show("results/figures/confusion_matrix_raw.png", 420); show("results/figures/confusion_matrix_normalized.png", 420); show("results/figures/confusion_matrix_persistence.png", 420)'),
        code('show("results/figures/roc_pr_curves.png"); show("results/figures/calibration.png", 420)'),
        code('''from sklearn.metrics import classification_report
d = np.load(ROOT / f"results/predictions/{st['selected_run']}_test.npz")
print(classification_report(d["y"], d["proba"].argmax(1), target_names=CLASS_NAMES, digits=3))'''),
        md("### Paired comparisons on identical test samples (site-block bootstrap of Δ macro-F1)"),
        code('''pd.DataFrame(st["paired_vs"]).T[["run","diff","ci95","p_diff_le_0"]]'''),
        md("""`diff` = macro-F1(MAF-Net) − macro-F1(other); `p_diff_le_0` is the bootstrap share of resamples
in which MAF-Net is not better. These are descriptive resampling statistics, not a formal
multiple-comparison-corrected significance test."""),
        md("### Forecast horizon and spatial generalisation"),
        code('''display(pd.read_csv(ROOT / "results/tables/horizons.csv")[["horizon","model","n_runs","accuracy","balanced_accuracy","macro_f1","macro_f1_std","roc_auc_macro_ovr"]].round(4))
show("results/figures/horizons.png", 520)
display(pd.read_csv(ROOT / "results/tables/spatial_holdout.csv").round(4))'''),
    ]


def nb10():
    return [
        md("""# 10 — Ablation Study and Explainability
Every ablation retrains the selected MAF-Net configuration with three seeds; metrics are reported
on validation (selection-relevant) and test (held-out)."""),
        code(HEADER),
        code('''ab = pd.read_csv(ROOT / "results/tables/ablation.csv")
ab.sort_values("val_macro_f1", ascending=False).round(4)'''),
        code('show("results/figures/ablation.png", 700)'),
        md("### Look-back length (validation; each length has its own sample set)"),
        code('''pd.read_csv(ROOT / "results/tables/lookback.csv")[["lookback","n_runs","val_macro_f1","val_macro_f1_std","macro_f1","macro_f1_std"]].round(4)'''),
        md("### Transferred drought representation (auxiliary US benchmark)"),
        code('''dr = read_json(ROOT / "experiments/drought_aux/drought_aux_results.json")
print("US drought benchmark (400 counties, 30-day weather → USDM class)")
print("  test macro-F1 = %.3f (majority = %.3f), balanced acc = %.3f" % (dr["test_metrics"]["macro_f1"], dr["test_majority_macro_f1"], dr["test_metrics"]["balanced_accuracy"]))
print("  transfer features on capital-city sites:", dr["transfer_feature_summary"])
show("results/figures/drought_aux_confusion.png", 380)'''),
        md("""### Explainability
* **Permutation importance** — drop in macro-F1 when a feature's whole 14-day trajectory is
  shuffled across samples (test subset of 8 000 windows).
* **Integrated gradients** (32 steps, zero baseline = training mean) for samples predicted High/Critical.
* **Attention weights** by predicted class. Attention is descriptive, not causal."""),
        code('''ex = read_json(ROOT / "results/analysis/explainability.json")
display(pd.DataFrame(ex["permutation_importance_groups"]).T.round(4))
show("results/figures/explain/permutation_importance_groups.png", 520)
show("results/figures/explain/permutation_importance_features.png", 600)
show("results/figures/explain/integrated_gradients_high_critical.png", 600)
show("results/figures/explain/attention_by_class.png", 600)'''),
        md("### Case studies: real held-out samples predicted High or Critical"),
        code('''cs = pd.json_normalize(ex["case_studies"])
cs[[c for c in cs.columns if not c.startswith("signals")]]'''),
        code('''pd.DataFrame([c["signals_last_input_day"] for c in ex["case_studies"]], index=[c["city"] for c in ex["case_studies"]])'''),
        code('show("results/figures/explain/risk_timelines.png")'),
        md("""Predictions are model outputs for a project-defined environmental-stress class; they are not
observed human-health outcomes."""),
    ]


def nb11():
    return [
        md("""# 11 — Final Results and Reproducible Inference
Loads the selected checkpoint, reruns inference on the held-out test period, verifies that the
stored metrics are reproduced exactly, and demonstrates inference for one city."""),
        code(HEADER),
        code('''from climatesense.training.trainer import load_trained, predict
from climatesense.evaluation.metrics import compute_metrics
from climatesense.labeling.compound_risk import CLASS_NAMES
st = read_json(ROOT / "results/metrics/selected_model_test.json"); rid = st["selected_run"]
rec = read_json(ROOT / "experiments" / rid.split("-")[0] / f"{rid}.json")
model, batcher, arr = load_trained(rec)
idx = arr.sample_index(rec["config"]["lookback"], rec["config"]["horizon"], "test")
P, Y, A = predict(model, batcher, idx)
m = compute_metrics(Y, P)
print("checkpoint:", rec["checkpoint"], "| params:", rec["n_params"])
print("recomputed test macro-F1 = %.4f | stored = %.4f" % (m["macro_f1"], st["test_metrics"]["macro_f1"]))
assert abs(m["macro_f1"] - st["test_metrics"]["macro_f1"]) < 1e-6'''),
        md("### Inference example: latest test-period forecasts for one site"),
        code('''p = arr.panel
site = p.loc[p.location_name.eq("New Delhi"), "site_id"].iloc[0] if p.location_name.eq("New Delhi").any() else p.site_id.iloc[0]
code_ = list(arr.site_index).index(site)
mask = arr.site_codes[idx] == code_
out = pd.DataFrame({"forecast_for": p.date.values[idx[mask] + 1], "observed_class": [CLASS_NAMES[int(v)] for v in Y[mask]],
                    "predicted_class": [CLASS_NAMES[int(v)] for v in P[mask].argmax(1)]})
for k in range(4): out[f"P({CLASS_NAMES[k]})"] = P[mask][:, k].round(3)
print(site); out.tail(10)'''),
        md("### Summary of findings"),
        code('''mc = pd.read_csv(ROOT / "results/tables/model_comparison_h1.csv").set_index("model")
ab = pd.read_csv(ROOT / "results/tables/ablation.csv").set_index("variant")
def g(name, col="macro_f1"): return mc.loc[name, col]
lines = [
 f"* Selected MAF-Net (seed with best validation F1): test macro-F1 {st['test_metrics']['macro_f1']:.3f} (95% CI {st['ci95_macro_f1_site_block_bootstrap'][0]:.3f}–{st['ci95_macro_f1_site_block_bootstrap'][1]:.3f}), accuracy {st['test_metrics']['accuracy']:.3f}, balanced accuracy {st['test_metrics']['balanced_accuracy']:.3f}, macro ROC-AUC {st['test_metrics']['roc_auc_macro_ovr']:.3f}.",
 f"* Five-seed MAF-Net mean test macro-F1 {g('MAF-Net (proposed, attention fusion)'):.3f} ± {g('MAF-Net (proposed, attention fusion)', 'macro_f1_std'):.3f}; concat fusion {g('MAF-Net w/o attention (concat fusion)'):.3f}; tuned GRU {g('GRU (tuned)'):.3f}; gradient boosting {g('Gradient boosting (HistGB)'):.3f}; persistence {g(\\"Persistence (today's class)\\"):.3f}.",
 f"* Ablation (test macro-F1, 3 seeds): climate-only {ab.loc['climate_only','macro_f1']:.3f}, air-only {ab.loc['air_only','macro_f1']:.3f}, dryness-only {ab.loc['dryness_only','macro_f1']:.3f}, all modalities with attention {ab.loc['all_attention','macro_f1']:.3f}.",
]
display(Markdown("\\n".join(lines)))'''),
        md("""### Limitations (see `docs/limitations.md`)
Two-and-a-half years of capital-city snapshots; observation-hour bias by longitude; provider
air-quality regime shifts; dryness is a proxy (no SPEI, soil moisture or NDVI); labels are
project-defined and not externally validated; results do not establish human-health impacts."""),
    ]


BUILDERS = {
    "01_dataset_acquisition": nb01, "02_exploratory_data_analysis": nb02, "03_data_cleaning_and_integration": nb03,
    "04_feature_engineering_and_labels": nb04, "05_baseline_models": nb05, "06_lstm_gru_experiments": nb06,
    "07_attention_fusion_model": nb07, "08_hyperparameter_optimization": nb08, "09_model_evaluation": nb09,
    "10_ablation_and_explainability": nb10, "11_final_results_and_inference": nb11,
}


def main():
    execute = "--no-exec" not in sys.argv
    only = [a for a in sys.argv[1:] if not a.startswith("--")]
    status = {}
    for name, fn in BUILDERS.items():
        if only and not any(o in name for o in only):
            continue
        nb = nbf.v4.new_notebook()
        nb.cells = fn()
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        path = NB / f"{name}.ipynb"
        if execute:
            t0 = time.time()
            try:
                NotebookClient(nb, timeout=1200, kernel_name="python3",
                               resources={"metadata": {"path": str(NB)}}).execute()
                status[name] = f"ok ({time.time() - t0:.0f}s)"
            except Exception as exc:  # keep the partially executed notebook for debugging
                status[name] = f"FAILED: {str(exc).splitlines()[0][:200]}"
        nbf.write(nb, path)
        print(name, status.get(name, "built"), flush=True)
    return status


if __name__ == "__main__":
    main()
