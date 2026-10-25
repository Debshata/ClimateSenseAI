import numpy as np
import pandas as pd

from climatesense.data import harmonize as H
from climatesense.utils import haversine_km


def test_haversine_known_distance():
    # London -> Paris is ~344 km
    assert abs(haversine_km(51.5074, -0.1278, 48.8566, 2.3522) - 343.5) < 3


def test_world_cities_join_uses_coordinates_and_threshold():
    sites = pd.DataFrame({"site_id": ["a", "b"], "country": ["Georgia", "Georgia"],
                          "lat": [41.72, 33.75], "lon": [44.78, -84.39]})
    wc = pd.DataFrame({"city_ascii": ["Tbilisi", "Atlanta", "Far"], "country": ["Georgia", "United States", "X"],
                       "iso3": ["GEO", "USA", "XXX"], "capital": ["primary", "admin", ""],
                       "population": [1e6, 5e5, 1.0], "lat": [41.7225, 33.7628, 0.0], "lng": [44.7925, -84.422, 0.0]})
    res = H.join_world_cities(sites, wc, max_km=50)
    # same-named country does not cause a false join: coordinates decide
    assert list(res.wc_city) == ["Tbilisi", "Atlanta"]
    assert res.wc_matched.all()
    assert bool(res.loc[1, "wc_country_agrees"]) is False  # flagged for review
    res2 = H.join_world_cities(sites.assign(lat=[0.0, 60.0], lon=[60.0, 60.0]), wc, max_km=50)
    assert not res2.wc_matched.any() and res2.population.isna().all()


def test_berkeley_coordinate_parsing():
    s = pd.Series(["52.24N", "33.12S", "5.26E", "70.10W"])
    np.testing.assert_allclose(H._parse_coord(s).values, [52.24, -33.12, 5.26, -70.10])


def test_site_id_merges_renamed_location_with_same_coordinates(raw):
    r = raw.copy()
    r.loc[r.index[:10], "location_name"] = "Alpha (renamed)"
    day, _ = H.clean_weather(r)
    assert day.site_id.nunique() == 2
