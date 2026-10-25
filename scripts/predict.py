"""Inference with the exported best MAF-Net checkpoint (models/best/).

    python scripts/predict.py --site "New Delhi" --date 2026-09-30
    python scripts/predict.py --site Riyadh                     # latest available date

The forecast for date D uses only the 14 observed days ending on D-1 (horizon 1).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch

from climatesense.labeling.compound_risk import CLASS_NAMES
from climatesense.training.trainer import load_trained
from climatesense.utils import read_json


def export_best():
    """Copy the selected checkpoint + record to models/best (called after evaluation)."""
    import shutil

    st = read_json(ROOT / "results/metrics/selected_model_test.json")
    rid = st["selected_run"]
    rec = read_json(ROOT / "experiments" / rid.split("-")[0] / f"{rid}.json")
    dst = ROOT / "models/best"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / rec["checkpoint"], dst / "mafnet_best.pt")
    for split in ("val", "test"):  # frozen predictions/probabilities of the selected model
        shutil.copy2(ROOT / "results/predictions" / f"{rid}_{split}.npz", dst / f"predictions_{split}.npz")
    import json

    rec = {**rec, "checkpoint": "models/best/mafnet_best.pt", "source_checkpoint": rec["checkpoint"]}
    (dst / "model_record.json").write_text(json.dumps(rec, indent=2, default=str))
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=False, default="New Delhi", help="location name (as in the weather data)")
    ap.add_argument("--date", default=None, help="target date YYYY-MM-DD (default: latest possible)")
    ap.add_argument("--export", action="store_true", help="export the selected checkpoint to models/best first")
    a = ap.parse_args()
    if a.export or not (ROOT / "models/best/model_record.json").exists():
        export_best()
    rec = read_json(ROOT / "models/best/model_record.json")
    model, batcher, arr = load_trained(rec)
    cfg = rec["config"]
    p = arr.panel
    cand = p[p.location_name.str.lower() == a.site.lower()]
    if cand.empty:
        raise SystemExit(f"unknown site {a.site!r}; examples: {sorted(p.location_name.unique())[:10]}")
    site = cand.site_id.mode().iloc[0]
    rows = np.where(arr.site_codes == list(arr.site_index).index(site))[0]
    if a.date:
        target = np.where(p.date.values[rows] == np.datetime64(a.date))[0]
        if not len(target):
            raise SystemExit("date outside the site's record")
        t = rows[target[0]] - cfg["horizon"]
    else:
        t = rows[-1]
    if t - cfg["lookback"] + 1 < rows[0]:
        raise SystemExit("not enough history before that date")
    with torch.no_grad():
        x, c, _ = batcher.get(torch.as_tensor([t], device=batcher.X.device))
        logits, att = model(x, c)
        prob = torch.softmax(logits, -1)[0].cpu().numpy()
    tgt_date = p.date.iloc[t] + pd.Timedelta(days=cfg["horizon"])
    print(f"site: {p.location_name.iloc[t]}, {p.country.iloc[t]}  | inputs: {p.date.iloc[t - cfg['lookback'] + 1].date()} .. {p.date.iloc[t].date()}")
    print(f"forecast for {tgt_date.date()}: {CLASS_NAMES[int(prob.argmax())]}")
    print("probabilities:", {k: round(float(v), 3) for k, v in zip(CLASS_NAMES, prob)})
    if att is not None:
        print("modality attention:", np.round(att[0].cpu().numpy(), 3).tolist(), "(climate, air, dryness, context)")
    if t + cfg["horizon"] < len(arr.y) and arr.site_codes[t + cfg["horizon"]] == arr.site_codes[t] and not np.isnan(arr.y[t + cfg["horizon"]]):
        print("observed class on that day:", CLASS_NAMES[int(arr.y[t + cfg["horizon"]])])


if __name__ == "__main__":
    main()
