# ClimateSense AI pipeline.  On Windows use `python` from the activated .venv.
PY ?= python

.PHONY: setup data prepare eda baselines neural tune seeds ablation horizons lookback spatial drought evaluate explain figures notebooks report test all

setup:
	$(PY) scripts/setup_environment.py
data:
	$(PY) scripts/download_datasets.py
prepare:
	$(PY) scripts/prepare_data.py --force
eda:
	$(PY) scripts/inspect_datasets.py && $(PY) scripts/analyze_labels.py
baselines:
	$(PY) scripts/train_baselines.py --horizon 1
	$(PY) scripts/train_baselines.py --horizon 3 --only hist_gradient_boosting logreg
	$(PY) scripts/train_baselines.py --horizon 7 --only hist_gradient_boosting logreg
drought:
	$(PY) scripts/train_drought_aux.py
neural:
	$(PY) scripts/run_experiments.py --stage neural_baselines
tune:
	$(PY) scripts/run_experiments.py --stage tune
seeds:
	$(PY) scripts/run_experiments.py --stage seeds
ablation:
	$(PY) scripts/run_experiments.py --stage ablation
horizons:
	$(PY) scripts/run_experiments.py --stage horizons
lookback:
	$(PY) scripts/run_experiments.py --stage lookback
spatial:
	$(PY) scripts/run_experiments.py --stage spatial
evaluate:
	$(PY) scripts/evaluate_models.py
explain:
	$(PY) scripts/explain_model.py
figures:
	$(PY) scripts/generate_figures.py
notebooks:
	$(PY) scripts/build_notebooks.py
report:
	$(PY) scripts/generate_report.py
test:
	$(PY) -m pytest

all: data prepare eda baselines drought neural tune seeds ablation horizons lookback spatial evaluate explain figures notebooks report test
