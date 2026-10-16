import math

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import cosine as scipy_cosine_distance

from src.nasa_power import (aggregate_daily_to_monthly, load_nasa_monthly,
                            monthly_matrix)
from src.rainfall import (DROUGHT, MONTHS, WATERLOGGING, CropRule, Region,
                          default_crops)
from src.seasons import classify_regime, detect_rainy_seasons, peak_count_sweep
from src.similarity import (cosine_similarity, cosine_similarity_matrix,
                            euclidean_distance_matrix, pearson_matrix)

KAMPALA = [120, 140, 180, 200, 220, 180, 90, 70, 60, 100, 110, 130]
GULU = [8, 25, 75, 160, 190, 145, 170, 215, 175, 150, 60, 15]
MBARARA = [70, 85, 120, 140, 90, 25, 20, 55, 100, 125, 120, 90]


# ------------------------------------------------------------------ Region
def test_region_statistics_by_hand():
    r = Region("Mbarara", MBARARA)
    assert r.annual_total() == 1040
    assert r.monthly_mean() == pytest.approx(1040 / 12)
    assert r.wettest_month() == ("Apr", 140.0)
    assert r.driest_month() == ("Jul", 20.0)
    assert r.coefficient_of_variation() == pytest.approx(np.std(MBARARA) / np.mean(MBARARA))
    assert len(r) == 12 and "Mbarara" in repr(r)


@pytest.mark.parametrize("bad", [[1] * 11, [1] * 13, [-1] + [1] * 11, [np.nan] + [1] * 11])
def test_region_rejects_invalid_rainfall(bad):
    with pytest.raises(ValueError):
        Region("X", bad)


def test_region_rejects_empty_name_and_zero_rainfall_cv():
    with pytest.raises(ValueError):
        Region(" ", [1] * 12)
    with pytest.raises(ValueError):
        Region("Dry", [0] * 12).coefficient_of_variation()


# ------------------------------------------------------------------ CropRule
def test_monthly_band_is_seasonal_band_divided_by_cycle():
    maize = default_crops()[0]
    assert (maize.monthly_min_mm, maize.monthly_max_mm) == (125, 200)
    beans = default_crops()[1]
    assert beans.monthly_min_mm == 100 and beans.monthly_max_mm == pytest.approx(500 / 3)
    assert len(default_crops()) >= 3 and all(c.source for c in default_crops())


def test_classify_boundaries():
    maize = default_crops()[0]
    assert maize.classify(124.99) == DROUGHT
    assert maize.classify(125) == "Good for maize"
    assert maize.classify(200) == "Good for maize"
    assert maize.classify(200.01) == WATERLOGGING
    with pytest.raises(ValueError):
        maize.classify(-1)


def test_classify_region_gives_twelve_labels():
    labels = default_crops()[0].classify_region(Region("Kampala", KAMPALA))
    assert len(labels) == 12 and labels[0] == DROUGHT and labels[4] == WATERLOGGING


@pytest.mark.parametrize("kwargs", [dict(seasonal_min_mm=500, seasonal_max_mm=400),
                                    dict(seasonal_min_mm=-1, seasonal_max_mm=5),
                                    dict(seasonal_min_mm=1, seasonal_max_mm=5, cycle_months=0),
                                    dict(seasonal_min_mm=1, seasonal_max_mm=5, cycle_months=13)])
def test_croprule_validation(kwargs):
    base = dict(name="X", seasonal_min_mm=1, seasonal_max_mm=2, cycle_months=3, source="s")
    base.update(kwargs)
    with pytest.raises(ValueError):
        CropRule(**base)


def test_cycle_total_wraps_around_the_year():
    beans, mb = default_crops()[1], Region("Mbarara", MBARARA)
    assert beans.cycle_total(mb, 2) == 120 + 140 + 90            # Mar-May
    assert beans.cycle_total(mb, 11) == 90 + 70 + 85              # Dec, Jan, Feb
    with pytest.raises(ValueError):
        beans.cycle_total(mb, 12)


def test_planting_windows_mbarara_beans_and_maize():
    mb = Region("Mbarara", MBARARA)
    crops = {c.name: c for c in default_crops()}
    ok = [w["plant"] for w in crops["Beans"].planting_windows(mb) if w["status"] == "adequate"]
    assert ok == ["Feb", "Mar", "Sep", "Oct"]
    assert all(w["status"] == "deficit" for w in crops["Maize"].planting_windows(mb))  # no rain-fed maize
    kam = crops["Maize"].planting_windows(Region("Kampala", KAMPALA))
    assert {w["plant"]: w["status"] for w in kam}["Mar"] == "adequate"


# ------------------------------------------------------------------ similarity
def test_cosine_is_scale_invariant_and_ignores_size():
    a = np.array(KAMPALA, float)
    assert cosine_similarity(a, 3 * a) == pytest.approx(1.0)       # 3x wetter, "identical"
    assert np.linalg.norm(a - 3 * a) > 900                          # yet far apart in mm


def test_cosine_known_values_and_errors():
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        cosine_similarity([0, 0], [1, 2])
    with pytest.raises(ValueError):
        cosine_similarity([1, 2], [1, 2, 3])


def test_cosine_matches_scipy_for_every_pair():
    X = np.array([KAMPALA, GULU, MBARARA], float)
    S = cosine_similarity_matrix(X)
    for i in range(3):
        for j in range(3):
            assert S[i, j] == pytest.approx(1 - scipy_cosine_distance(X[i], X[j]))
    assert np.allclose(np.diag(S), 1)


def test_math_cos_is_not_a_vector_similarity():
    with pytest.raises(TypeError):
        math.cos([1, 2, 3])                                        # takes one number (an angle)


def test_pearson_and_euclidean_matrices():
    X = np.array([KAMPALA, GULU, MBARARA], float)
    P, E = pearson_matrix(X), euclidean_distance_matrix(X)
    assert P[0, 1] == pytest.approx(np.corrcoef(KAMPALA, GULU)[0, 1])
    assert np.allclose(E, E.T) and np.allclose(np.diag(E), 0)
    assert E[0, 1] == pytest.approx(np.linalg.norm(np.array(KAMPALA) - np.array(GULU)))
    assert pearson_matrix([KAMPALA, [2 * v + 5 for v in KAMPALA]])[0, 1] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        pearson_matrix([KAMPALA, [5] * 12])


# ------------------------------------------------------------------ seasons
def test_regimes_at_default_threshold():
    reg = {n: classify_regime(detect_rainy_seasons(v)) for n, v in
           {"Kampala": KAMPALA, "Gulu": GULU, "Mbarara": MBARARA}.items()}
    assert reg == {"Kampala": "unimodal", "Gulu": "unimodal", "Mbarara": "bimodal"}
    assert [MONTHS[i] for i in detect_rainy_seasons(MBARARA)] == ["Apr", "Oct"]


def test_low_threshold_creates_spurious_peaks():
    sweep = peak_count_sweep(KAMPALA, [0.02, 0.30])
    assert sweep == {0.02: 2, 0.30: 1}                             # Dec "peak" is only a 10 mm wiggle


def test_peak_at_year_boundary_is_found():
    v = [200, 100, 50, 20, 10, 10, 10, 20, 50, 100, 120, 180]      # Dec 180 and Jan 200 are one broad peak
    peaks = detect_rainy_seasons(v)
    assert peaks == [0]
    two = [100, 40, 20, 90, 20, 10, 10, 10, 20, 30, 40, 80]
    assert len(detect_rainy_seasons(two, 0.2)) == 2


def test_seasons_edge_cases():
    assert detect_rainy_seasons([50] * 12) == []
    assert classify_regime([]) == "no distinct season"
    assert classify_regime([1, 2, 3]) == "multimodal"
    for bad in ([1] * 11, [-1] + [1] * 11):
        with pytest.raises(ValueError):
            detect_rainy_seasons(bad)
    with pytest.raises(ValueError):
        detect_rainy_seasons(MBARARA, prominence_fraction=1.5)


# ------------------------------------------------------------------ NASA POWER helpers
def _daily(year, month, days, value):
    return {f"{year}{month:02d}{d:02d}": value for d in range(1, days + 1)}


def test_aggregate_daily_to_monthly_sums_and_flags_missing_days():
    daily = {**_daily(2020, 1, 31, 2.0), **_daily(2020, 2, 29, 1.0)}
    daily["20200215"] = -999                                         # missing day
    out = aggregate_daily_to_monthly("Gulu", daily).set_index("month")
    assert out.loc[1, "rainfall_mm"] == pytest.approx(62.0) and bool(out.loc[1, "complete"])
    assert out.loc[2, "rainfall_mm"] == pytest.approx(28.0) and not bool(out.loc[2, "complete"])
    assert out.loc[2, "n_days"] == 28
    with pytest.raises(ValueError):
        aggregate_daily_to_monthly("Gulu", {"20200101": -999})


def test_load_nasa_csv_validation_and_matrix(tmp_path):
    rows = [{"region": "A", "year": y, "month": m, "rainfall_mm": 10.0 + m} for y in (2020, 2021) for m in range(1, 13)]
    good = tmp_path / "ok.csv"
    pd.DataFrame(rows).to_csv(good, index=False)
    df = load_nasa_monthly(good)
    assert monthly_matrix(df, "A").shape == (2, 12)
    with pytest.raises(ValueError):
        monthly_matrix(df, "B")
    with pytest.raises(ValueError):
        monthly_matrix(df[df["month"] != 5], "A")                    # a missing month

    pd.DataFrame(rows).drop(columns="year").to_csv(tmp_path / "cols.csv", index=False)
    with pytest.raises(ValueError):
        load_nasa_monthly(tmp_path / "cols.csv")
    bad = pd.DataFrame(rows); bad.loc[0, "rainfall_mm"] = -5
    bad.to_csv(tmp_path / "neg.csv", index=False)
    with pytest.raises(ValueError):
        load_nasa_monthly(tmp_path / "neg.csv")
    (tmp_path / "empty.csv").write_text("region,year,month,rainfall_mm\n")
    with pytest.raises(ValueError):
        load_nasa_monthly(tmp_path / "empty.csv")
