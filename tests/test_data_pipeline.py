import numpy as np
import pandas as pd

from climatesense.data import harmonize as H


def test_schema_normalisation_and_date_parsing(raw):
    day, audit = H.clean_weather(raw)
    for c in ["temp_c", "rh", "pm25", "pm10", "o3", "no2", "site_id", "date"]:
        assert c in day.columns
    assert pd.api.types.is_datetime64_any_dtype(day["date"])
    assert (day["date"].dt.hour == 0).all()  # normalised to the local calendar day
    assert audit["raw_rows"] == len(raw)


def test_sentinels_and_implausible_values_become_nan(raw):
    raw = raw.copy()
    raw.loc[0, "air_quality_Carbon_Monoxide"] = -9999.0
    raw.loc[1, "air_quality_PM10"] = -1848.15
    raw.loc[2, "humidity"] = 140
    day, audit = H.clean_weather(raw)
    assert audit["invalid_values_set_nan"]["co"] >= 1
    assert audit["invalid_values_set_nan"]["pm10"] >= 1
    assert audit["invalid_values_set_nan"]["rh"] >= 1
    assert day["co"].min() >= 0 and day["pm10"].min() >= 0


def test_duplicate_removal_and_same_day_aggregation(raw):
    dup = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)  # exact duplicate
    later = raw.iloc[[1]].copy()
    later["last_updated"] = later["last_updated"].str.replace("12:00", "18:00")
    later["temperature_celsius"] += 10
    day, audit = H.clean_weather(pd.concat([dup, later], ignore_index=True))
    assert audit["exact_duplicate_rows"] == 1
    assert audit["same_day_records_averaged"] == 1
    assert not day.duplicated(["site_id", "date"]).any()


def test_daily_grid_is_continuous_and_flags_unobserved(raw):
    raw = raw.drop(index=[5, 6, 7]).reset_index(drop=True)  # 3-day gap at the first site
    day, _ = H.clean_weather(raw)
    grid = H.to_daily_grid(day)
    for _, g in grid.groupby("site_id"):
        assert (g["date"].diff().dropna() == pd.Timedelta(days=1)).all()
    assert grid["observed"].sum() == len(day)
    assert (grid["observed"] == 0).sum() == 3


def test_site_selection_threshold(raw):
    day, _ = H.clean_weather(raw)
    kept, stats = H.select_sites(day, 61)
    assert len(kept) == 0 and not stats.kept.any()
    kept, stats = H.select_sites(day, 60)
    assert stats.kept.all()
