"""Build the harmonised, labelled site-day panel.

Outputs
  data/processed/panel.parquet          site-day grid with features, hazard scores, labels
  data/processed/sites.csv              site table with World Cities / Berkeley Earth joins
  data/manifests/data_audit.json        cleaning, join and label statistics
  data/manifests/splits.json            chronological split boundaries
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from climatesense.data import harmonize as H
from climatesense.data.windows import split_dates
from climatesense.features.engineering import add_daily_features
from climatesense.labeling.compound_risk import CLASS_NAMES, label_panel
from climatesense.utils import atomic_write_json, load_yaml


def main(force: bool = False) -> None:
    out_panel = ROOT / "data/processed/panel.parquet"
    if out_panel.exists() and not force and "--force" not in sys.argv:
        print("panel exists; use --force to rebuild")
        return
    dcfg, pcfg = load_yaml("data"), load_yaml("preprocessing")
    hcfg = dcfg["harmonisation"]
    audit = {}

    raw = H.load_weather_raw()
    day, audit["weather_cleaning"] = H.clean_weather(raw, hcfg["site_coord_round_deg"])
    day, site_stats = H.select_sites(day, hcfg["min_observed_days"])
    audit["sites_total"] = int(len(site_stats))
    audit["sites_kept"] = int(site_stats.kept.sum())
    audit["site_days_kept"] = int(len(day))
    site_stats.to_csv(ROOT / "data/manifests/site_selection.csv", index=False)

    sites = site_stats[site_stats.kept].drop(columns="kept")
    wc = H.load_world_cities()
    sites = H.join_world_cities(sites, wc, hcfg["worldcities_max_km"])
    bk = H.berkeley_climatology(normal=tuple(hcfg["berkeley_normal_period"]),
                                recent=tuple(hcfg["berkeley_recent_period"]))
    sites = H.join_berkeley(sites, bk, hcfg["berkeley_max_km"])
    audit["world_cities_join"] = {
        "matched": int(sites.wc_matched.sum()), "unmatched": int((~sites.wc_matched).sum()),
        "ambiguous_lt2km": int(sites.wc_ambiguous.sum()),
        "country_name_disagreements": int((sites.wc_matched & ~sites.wc_country_agrees).sum()),
        "median_distance_km": float(sites.loc[sites.wc_matched, "wc_distance_km"].median()),
    }
    audit["berkeley_join"] = {
        "berkeley_cities_with_normals": int(len(bk)),
        "matched": int(sites.bk_matched.sum()), "unmatched": int((~sites.bk_matched).sum()),
        "median_distance_km": float(sites.loc[sites.bk_matched, "bk_distance_km"].median()),
    }
    # population is only context: missing population -> NaN (imputed by scaler), flagged in audit
    sites.to_csv(ROOT / "data/processed/sites.csv", index=False)

    grid = H.to_daily_grid(day)
    audit["grid_rows"] = int(len(grid))
    audit["grid_unobserved_days"] = int((grid.observed == 0).sum())
    feat = add_daily_features(grid, sites, hcfg["ffill_max_days"])
    lab = label_panel(feat, pcfg["labels"])
    sp = split_dates(lab.date, pcfg["split"]["train_frac"], pcfg["split"]["val_frac"])
    lab["split_by_date"] = np.select(
        [lab.date <= sp["train"][1], lab.date <= sp["val"][1]], ["train", "val"], "test")
    audit["label_distribution_all"] = {CLASS_NAMES[int(k)]: int(v) for k, v in lab.risk_class.value_counts().sort_index().items()}
    audit["label_distribution_by_split"] = {
        s: {CLASS_NAMES[int(k)]: int(v) for k, v in g.risk_class.value_counts().sort_index().items()}
        for s, g in lab.groupby("split_by_date")}
    audit["hazard_score_distribution"] = {
        c: {str(int(k)): int(v) for k, v in lab[c].value_counts().sort_index().items()}
        for c in ["heat_score_obs", "air_score_obs", "dry_score_obs"]}
    audit["label_coverage_mean"] = float(lab.loc[lab.observed == 1, "label_coverage"].mean())
    audit["date_range"] = [str(lab.date.min().date()), str(lab.date.max().date())]

    lab.to_parquet(out_panel, index=False)
    atomic_write_json({k: [str(a.date()), str(b.date())] for k, (a, b) in sp.items()},
                      ROOT / "data/manifests/splits.json")
    atomic_write_json(audit, ROOT / "data/manifests/data_audit.json")
    print(pd.Series(audit["label_distribution_all"]))
    print(audit["label_distribution_by_split"])
    print({k: v for k, v in audit.items() if "join" in k or "site" in k})


if __name__ == "__main__":
    main()
