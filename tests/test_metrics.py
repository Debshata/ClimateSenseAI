from pathlib import Path

import numpy as np
import pytest

from climatesense.evaluation.metrics import bootstrap_ci, compute_metrics, paired_bootstrap_diff

ROOT = Path(__file__).resolve().parents[1]


def test_perfect_predictions():
    y = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    m = compute_metrics(y, np.eye(4)[y])
    assert m["accuracy"] == 1 and m["macro_f1"] == 1 and m["roc_auc_macro_ovr"] == 1
    assert np.trace(np.array(m["confusion_matrix"])) == 8


def test_known_values():
    y = np.array([0, 0, 1, 1])
    P = np.eye(4)[[0, 1, 1, 1]]
    m = compute_metrics(y, P)
    assert m["accuracy"] == 0.75
    assert m["per_class"]["0"]["recall"] == 0.5 and m["per_class"]["1"]["precision"] == pytest.approx(2 / 3)


def test_roc_undefined_when_class_absent():
    y = np.array([0, 1, 2, 0, 1, 2])
    P = np.full((6, 4), 0.25)
    m = compute_metrics(y, P)
    assert m["roc_auc_macro_ovr"] is None
    assert m["per_class"]["3"]["roc_auc"] is None


def test_bootstrap_helpers():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 400)
    P = np.eye(4)[np.where(rng.random(400) < 0.8, y, rng.integers(0, 4, 400))]
    lo, hi = bootstrap_ci(y, P, "macro_f1", 200, groups=rng.integers(0, 20, 400))
    assert lo < compute_metrics(y, P)["macro_f1"] < hi
    d = paired_bootstrap_diff(y, P, P, 100)
    assert d["diff"] == 0


@pytest.mark.skipif(not (ROOT / "results/metrics/selected_model_test.json").exists(), reason="no trained model yet")
@pytest.mark.skipif(not (ROOT / "data/processed/panel.parquet").exists(), reason="processed data not available (run prepare_data.py)")
def test_selected_checkpoint_loads_and_reproduces_test_predictions():
    """Smoke test on real data: reload the selected checkpoint and reproduce stored test probabilities."""
    from climatesense.training.trainer import load_trained, predict
    from climatesense.utils import read_json

    sel = read_json(ROOT / "results/metrics/selected_model_test.json")["selected_run"]
    rec = read_json(ROOT / "models/best/model_record.json")  # exported, Git-tracked checkpoint
    assert rec["run_id"] == sel
    stored = np.load(ROOT / "models/best/predictions_test.npz")
    model, batcher, _ = load_trained(rec)
    idx = stored["idx"][:2000]
    P, Y, _ = predict(model, batcher, idx)
    np.testing.assert_allclose(P, stored["proba"][:2000], atol=1e-4)
    np.testing.assert_array_equal(Y, stored["y"][:2000])
