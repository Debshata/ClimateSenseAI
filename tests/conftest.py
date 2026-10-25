import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LABEL_CFG = {
    "heat_index_c": [27.0, 32.0, 41.0], "wet_bulb_c": {"level2": 28.0, "level3": 31.0},
    "pm25": [15.0, 37.5, 75.0], "pm10": [45.0, 100.0, 150.0], "o3": [100.0, 160.0, 240.0],
    "no2": [25.0, 50.0, 120.0], "vpd_kpa": [1.0, 2.0, 3.0], "dry_spell_days": 30, "dry_spell_min_obs": 20,
    "compound": {"critical_sum": 6, "high_sum": 4, "moderate_sum": 2},
}


@pytest.fixture
def label_cfg():
    return LABEL_CFG


def make_raw(n_days=60, sites=(("Testland", "Alpha", 10.0, 20.0), ("Otherland", "Beta", -30.0, 150.0)), seed=0):
    """Small synthetic raw frame with the World Weather Repository schema."""
    rng = np.random.default_rng(seed)
    rows = []
    for country, loc, lat, lon in sites:
        for d in pd.date_range("2025-01-01", periods=n_days):
            rows.append({
                "country": country, "location_name": loc, "latitude": lat, "longitude": lon,
                "last_updated": f"{d.date()} 12:00", "temperature_celsius": 25 + rng.normal(0, 5),
                "feels_like_celsius": 26 + rng.normal(0, 5), "humidity": int(rng.integers(20, 95)),
                "wind_kph": 10.0, "gust_kph": 15.0, "pressure_mb": 1012.0, "precip_mm": float(rng.random() < 0.3),
                "cloud": 50, "visibility_km": 10.0, "uv_index": 5.0,
                "air_quality_Carbon_Monoxide": 300.0, "air_quality_Ozone": 60.0,
                "air_quality_Nitrogen_dioxide": 10.0, "air_quality_Sulphur_dioxide": 2.0,
                "air_quality_PM2.5": float(rng.gamma(2, 10)), "air_quality_PM10": 30.0,
                "air_quality_us-epa-index": 1, "air_quality_gb-defra-index": 1,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def raw():
    return make_raw()


@pytest.fixture
def labelled_panel(raw, label_cfg):
    from climatesense.data import harmonize as H
    from climatesense.features.engineering import add_daily_features
    from climatesense.labeling.compound_risk import label_panel

    day, _ = H.clean_weather(raw)
    day, stats = H.select_sites(day, 10)
    sites = stats[stats.kept].copy()
    sites["population"] = 1e6
    for c in ["bk_normal_annual", "bk_seasonal_amplitude", "bk_warming_recent_minus_normal"]:
        sites[c] = 1.0
    grid = H.to_daily_grid(day)
    return label_panel(add_daily_features(grid, sites), label_cfg), sites
