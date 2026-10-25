"""Check that every required deliverable exists and is non-trivial; scan for secrets and large files.

Writes docs/deliverable_check.json and exits non-zero if a required item is missing.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "README.md", "LICENSE", ".gitignore", ".env.example", "pyproject.toml", "requirements.txt",
    "requirements-dev.txt", "environment.yml", "Makefile", "pytest.ini",
    "configs/data.yaml", "configs/preprocessing.yaml", "configs/training.yaml", "configs/model.yaml", "configs/experiments.yaml",
    "docs/methodology.md", "docs/dataset_documentation.md", "docs/label_definition.md", "docs/experiment_log.md",
    "docs/model_architecture.md", "docs/reproducibility.md", "docs/limitations.md", "docs/final_execution_status.md",
    "data/manifests/download_manifest.json", "data/manifests/data_audit.json", "data/manifests/dataset_quality.json",
    "results/metrics/test_metrics.json", "results/metrics/selected_model_test.json",
    "results/tables/model_comparison_h1.csv", "results/tables/ablation.csv", "results/tables/horizons.csv",
    "results/figures/confusion_matrix_raw.png", "results/figures/confusion_matrix_normalized.png",
    "results/figures/roc_pr_curves.png", "results/figures/training_curves.png", "results/figures/model_comparison.png",
    "results/figures/ablation.png", "results/figures/calibration.png", "results/figures/architecture_mafnet.png",
    "results/analysis/explainability.json", "results/analysis/label_sensitivity.json",
    "reports/final/ClimateSenseAI_IEEE_Final_Report.docx",
] + [f"notebooks/{n}.ipynb" for n in [
    "01_dataset_acquisition", "02_exploratory_data_analysis", "03_data_cleaning_and_integration",
    "04_feature_engineering_and_labels", "05_baseline_models", "06_lstm_gru_experiments", "07_attention_fusion_model",
    "08_hyperparameter_optimization", "09_model_evaluation", "10_ablation_and_explainability",
    "11_final_results_and_inference"]]

SECRET_PATTERNS = [re.compile(p) for p in [r"KGAT_[0-9a-f]{20,}", r"ghp_[A-Za-z0-9]{30,}", r"AKIA[0-9A-Z]{16}",
                                           r"-----BEGIN (RSA |EC )?PRIVATE KEY-----"]]


def notebook_status(path: Path) -> dict:
    nb = json.loads(path.read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    errors = [o for c in code for o in c.get("outputs", []) if o.get("output_type") == "error"]
    executed = all(c.get("execution_count") for c in code)
    return {"cells": len(nb["cells"]), "code_cells": len(code), "executed": executed, "errors": len(errors)}


def main():
    res = {"missing": [], "empty": [], "notebooks": {}, "secrets_in_tracked_files": [], "large_tracked_files": []}
    for r in REQUIRED:
        p = ROOT / r
        if not p.exists():
            res["missing"].append(r)
        elif p.stat().st_size < 50 and r != ".env.example":
            res["empty"].append(r)
        elif r.endswith(".ipynb"):
            res["notebooks"][r] = notebook_status(p)
    tracked = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=ROOT, capture_output=True, text=True).stdout.split()
    for f in tracked:
        p = ROOT / f
        if not p.is_file():
            continue
        if p.stat().st_size > 20e6:
            res["large_tracked_files"].append([f, p.stat().st_size])
        if p.suffix in {".py", ".md", ".json", ".yaml", ".yml", ".txt", ".ipynb", ".toml", ".ini", ".cfg"} or p.name.startswith(".env"):
            text = p.read_text(encoding="utf-8", errors="ignore")
            for pat in SECRET_PATTERNS:
                if pat.search(text):
                    res["secrets_in_tracked_files"].append(f)
    res["ok"] = not (res["missing"] or res["empty"] or res["secrets_in_tracked_files"] or res["large_tracked_files"]
                     or any(v["errors"] or not v["executed"] for v in res["notebooks"].values()))
    (ROOT / "docs/deliverable_check.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
