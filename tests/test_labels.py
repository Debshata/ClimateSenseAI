import numpy as np

from climatesense.features.engineering import dewpoint_c, heat_index_c, vpd_kpa, wet_bulb_stull_c
from climatesense.labeling.compound_risk import air_score, band, compound_class


def test_heat_index_matches_nws_table():
    # NWS table: 90 F & 60 % RH -> 100 F ; 80 F & 40 % -> 80 F
    hi = heat_index_c((90 - 32) * 5 / 9, 60) * 9 / 5 + 32
    assert abs(hi - 100) < 1.5
    hi2 = heat_index_c((80 - 32) * 5 / 9, 40) * 9 / 5 + 32
    assert abs(hi2 - 80) < 1.5


def test_heat_index_cold_conditions_use_simple_formula():
    hi = heat_index_c(5.0, 90)
    assert 0 < hi < 10  # no spurious regression output far below its valid range


def test_wet_bulb_stull_reference():
    # Stull (2011): T=20 C, RH=50 % -> Tw ~= 13.7 C
    assert abs(wet_bulb_stull_c(20.0, 50.0) - 13.7) < 0.2


def test_dewpoint_and_vpd_physical_limits():
    assert abs(dewpoint_c(25.0, 100.0) - 25.0) < 0.05
    assert vpd_kpa(25.0, 100.0) == 0
    assert vpd_kpa(35.0, 10.0) > vpd_kpa(25.0, 10.0)


def test_band_edges_and_nan():
    out = band([0, 15, 37.5, 74.9, 75, np.nan], [15, 37.5, 75])
    np.testing.assert_array_equal(out[:5], [0, 1, 2, 2, 3])
    assert np.isnan(out[5])


def test_air_score_takes_worst_pollutant(label_cfg):
    s = air_score(np.array([5.0, 5.0]), np.array([20.0, 200.0]), np.array([50.0, 50.0]), np.array([np.nan, 5.0]), label_cfg)
    np.testing.assert_array_equal(s, [0, 3])
    assert np.isnan(air_score(*(np.array([np.nan]),) * 4, label_cfg))[0]


def test_compound_class_rules(label_cfg):
    H = np.array([0, 1, 2, 3, 2, 3, 0, 1, np.nan])
    A = np.array([0, 0, 0, 0, 2, 2, 1, 1, 0])
    D = np.array([0, 1, 0, 0, 0, 0, 0, 1, 0])
    y = compound_class(H, A, D, label_cfg)
    # S: 0,2,2,3,4,5,1,3 ; m: 0,1,2,3,2,3,1,1 ; n2: 0,0,1,1,2,2,0,0
    np.testing.assert_array_equal(y[:8], [0, 1, 1, 2, 2, 3, 0, 1])
    assert np.isnan(y[8])  # undefined without heat information


def test_label_values_and_reproducibility(labelled_panel):
    p, _ = labelled_panel
    y = p.risk_class.dropna()
    assert set(np.unique(y)).issubset({0, 1, 2, 3})
    from climatesense.labeling.compound_risk import label_panel
    from conftest import LABEL_CFG
    p2 = label_panel(p.drop(columns=["risk_class"]), LABEL_CFG)
    np.testing.assert_array_equal(p.risk_class.values, p2.risk_class.values)
