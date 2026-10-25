import numpy as np
import pandas as pd

from climatesense.data import harmonize as H
from climatesense.data.windows import PanelArrays, Scaler, split_dates
from climatesense.features.engineering import add_daily_features
from climatesense.labeling.compound_risk import label_panel
from conftest import make_raw


def _panel(raw, cfg):
    day, _ = H.clean_weather(raw)
    day, stats = H.select_sites(day, 10)
    sites = stats[stats.kept].copy()
    sites["population"] = 1e6
    for c in ["bk_normal_annual", "bk_seasonal_amplitude", "bk_warming_recent_minus_normal"]:
        sites[c] = 1.0
    return label_panel(add_daily_features(H.to_daily_grid(day), sites), cfg)


def test_rolling_features_do_not_use_future(label_cfg):
    raw = make_raw(n_days=80)
    base = _panel(raw, label_cfg)
    future = raw.copy()
    cut = pd.Timestamp("2025-02-15")
    mask = pd.to_datetime(future.last_updated) >= cut
    future.loc[mask, "precip_mm"] = 50.0
    future.loc[mask, "temperature_celsius"] = 45.0
    alt = _panel(future, label_cfg)
    cols = ["precip_sum_7d", "precip_sum_30d", "wet_frac_7d", "wet_frac_30d", "days_since_wet",
            "dry_score", "heat_score", "temp_c", "risk_class"]
    b = base[base.date < cut][cols].reset_index(drop=True)
    a = alt[alt.date < cut][cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_forward_fill_limited_and_labels_never_imputed(label_cfg):
    raw = make_raw(n_days=60)
    raw = raw.drop(index=list(range(10, 16))).reset_index(drop=True)  # 6-day gap at site 1
    p = _panel(raw, label_cfg)
    g = p[p.site_id.str.startswith("Testland")].reset_index(drop=True)
    gap = g[g.observed == 0]
    assert len(gap) == 6
    assert gap.risk_class.isna().all()            # unobserved days are never labelled
    assert gap.temp_c.notna().sum() == 3          # ffill at most 3 days


def _arrays(label_cfg, n_days=200):
    p = _panel(make_raw(n_days=n_days), label_cfg)
    return PanelArrays(p, split_dates(p.date)), p


def test_chronological_split_has_no_overlap(label_cfg):
    arr, p = _arrays(label_cfg)
    L, h = 14, 1
    ranges = {}
    for s in ["train", "val", "test"]:
        idx = arr.sample_index(L, h, s)
        assert len(idx) > 0
        first_in = arr.date_ns[idx - L + 1]
        tgt = arr.date_ns[idx + h]
        lo, hi = (np.datetime64(d.date(), "D") for d in arr.splits[s])
        assert (first_in >= lo).all() and (tgt <= hi).all()
        ranges[s] = (first_in.min(), tgt.max())
        assert (arr.site_codes[idx - L + 1] == arr.site_codes[idx + h]).all()  # windows never cross sites
    assert ranges["train"][1] < ranges["val"][0] and ranges["val"][1] < ranges["test"][0]


def test_scaler_fit_on_training_rows_only(label_cfg):
    arr, p = _arrays(label_cfg)
    p2 = p.copy()
    p2.loc[p2.date > arr.splits["train"][1], "temp_c"] += 100  # perturb val/test only
    arr2 = PanelArrays(p2, arr.splits)
    np.testing.assert_allclose(arr.scaler.mean, arr2.scaler.mean)
    np.testing.assert_allclose(arr.scaler.std, arr2.scaler.std)


def test_scaler_roundtrip():
    df = pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [0.0, 0.0, 0.0]})
    s = Scaler.fit(df, ["a", "b"])
    s2 = Scaler.from_dict(s.to_dict())
    np.testing.assert_allclose(s.transform(df), s2.transform(df))
    assert not np.isnan(s.transform(df)).any()
