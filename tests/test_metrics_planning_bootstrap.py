import numpy as np
import pytest

from src.bootstrap import bootstrap_cagr_interval
from src.data import load_districts
from src.forecasters import CAGRForecaster, LinearTrendForecaster
from src.metrics import compare_models, mae, mape, rmse, select_best
from src.planning import additional_classrooms, classrooms_needed


def test_metrics_hand_calculation():
    actual, pred = [100, 200], [110, 180]
    assert mae(actual, pred) == pytest.approx(15)
    assert rmse(actual, pred) == pytest.approx(np.sqrt(250))
    assert mape(actual, pred) == pytest.approx(10.0)  # (10% + 10%) / 2


def test_metrics_edge_cases():
    with pytest.raises(ValueError):
        mape([0, 1], [1, 1])
    with pytest.raises(ValueError):
        mae([], [])
    with pytest.raises(ValueError):
        rmse([1, 2], [1])


def test_compare_and_select_best():
    d = [x for x in load_districts() if x.name == "Kampala"][0]
    scores = compare_models(d, {"Linear": LinearTrendForecaster, "CAGR": CAGRForecaster})
    assert select_best(scores) == "CAGR"
    with pytest.raises(ValueError):
        select_best({})


def test_classrooms_rounding_and_validation():
    assert classrooms_needed(0) == 0
    assert classrooms_needed(1.0) == 4          # 180 pupils / 53 = 3.4 -> 4
    assert additional_classrooms(1800, 2254.76) == 1544
    with pytest.raises(ValueError):
        classrooms_needed(-1)
    with pytest.raises(ValueError):
        classrooms_needed(10, school_age_share=1.5)


def test_bootstrap_reproducible_and_ordered():
    d = load_districts()[0]
    lo1, hi1 = bootstrap_cagr_interval(d.years, d.populations, 5, seed=1)
    lo2, hi2 = bootstrap_cagr_interval(d.years, d.populations, 5, seed=1)
    np.testing.assert_array_equal(lo1, lo2)
    assert np.all(lo1 < hi1)
    point = CAGRForecaster().fit(d.years, d.populations).predict(5)
    assert np.all((lo1 < point) & (point < hi1))


def test_bootstrap_edge_cases():
    with pytest.raises(ValueError):
        bootstrap_cagr_interval([2015, 2016, 2017], [1, 2, 3], 2)   # too short
    d = load_districts()[0]
    with pytest.raises(ValueError):
        bootstrap_cagr_interval(d.years, d.populations, 5, level=1.2)
