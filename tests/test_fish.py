import numpy as np
import pytest

from src.fish import FishStock, fibonacci_sequence
from src.price import PriceModel
from src.risk import (RiskAssessor, annual_revenue, describe_revenue,
                      revenue_by_year, weekly_revenue)


# ------------------------------------------------------------------ Fibonacci
def test_fibonacci_sequence():
    assert fibonacci_sequence(8) == [0, 1, 1, 2, 3, 5, 8, 13]
    assert len(fibonacci_sequence(15)) == 15
    with pytest.raises(ValueError):
        fibonacci_sequence(0)


# ------------------------------------------------------------------ FishStock
def test_stock_converges_to_theoretical_equilibrium():
    fish = FishStock()
    stock, _ = fish.simulate(0.10, weeks=400)
    assert stock[-1] == pytest.approx(fish.equilibrium_stock(0.10), rel=1e-6)  # 7,500
    assert fish.equilibrium_stock(0.10) == pytest.approx(7500)


def test_no_harvest_converges_to_carrying_capacity():
    stock, catch = FishStock().simulate(0.0, weeks=400)
    assert stock[-1] == pytest.approx(10_000, rel=1e-6)
    assert catch.sum() == 0


def test_msy_and_equilibrium_catch_peak_at_r_over_2():
    fish = FishStock()
    assert fish.msy == 1000 and fish.msy_harvest_rate == 0.2 and fish.msy_stock == 5000
    assert fish.equilibrium_catch(0.2) == pytest.approx(fish.msy)
    for h in (0.1, 0.15, 0.25, 0.3):
        assert fish.equilibrium_catch(h) < fish.msy


def test_overharvesting_collapses_stock():
    fish = FishStock()
    stock, _ = fish.simulate(0.5, weeks=2000)      # h > r
    assert stock[-1] < 1.0 and np.all(stock >= 0)
    assert fish.equilibrium_stock(0.5) == 0.0


@pytest.mark.parametrize("h, status", [(0.05, "sustainable"), (0.2, "sustainable"),
                                       (0.3, "overfishing"), (0.4, "collapse"), (0.9, "collapse")])
def test_sustainability_status(h, status):
    assert FishStock().sustainability_status(h) == status


def test_closed_season_has_zero_catch_in_closed_weeks_each_year():
    _, catch = FishStock().simulate(0.1, weeks=104, closed_weeks=8)
    assert np.all(catch[:8] == 0) and np.all(catch[52:60] == 0)
    assert np.all(catch[8:52] > 0)


@pytest.mark.parametrize("kwargs", [dict(harvest_rate=-0.1), dict(harvest_rate=1.5),
                                    dict(harvest_rate=0.1, weeks=0),
                                    dict(harvest_rate=0.1, closed_weeks=60)])
def test_simulate_rejects_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        FishStock().simulate(**kwargs)


def test_fishstock_rejects_bad_parameters():
    for bad in (dict(r=0), dict(carrying_capacity=-1), dict(initial_stock=-5)):
        with pytest.raises(ValueError):
            FishStock(**bad)


# ------------------------------------------------------------------ PriceModel
def test_prices_start_correctly_and_stay_within_bounds():
    paths = PriceModel().simulate_paths(2000, 52)
    assert paths.shape == (2000, 52)
    assert np.all(paths[:, 0] == 12_000)
    assert paths.min() >= 9_000 and paths.max() <= 16_000


def test_prices_reproducible_for_same_seed_and_differ_across_seeds():
    a = PriceModel(seed=1).simulate_paths(10, 20)
    b = PriceModel(seed=1).simulate_paths(10, 20)
    c = PriceModel(seed=2).simulate_paths(10, 20)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_pricemodel_rejects_bad_arguments():
    for bad in (dict(start=20_000), dict(weekly_volatility=0), dict(lower=0, start=0)):
        with pytest.raises(ValueError):
            PriceModel(**bad)
    with pytest.raises(ValueError):
        PriceModel().simulate_paths(0, 10)


# ------------------------------------------------------------------ revenue
def test_revenue_by_hand():
    # 1 t at 10 UGX/kg = 10,000 ; 2 t at 20 UGX/kg = 40,000
    w = weekly_revenue([1, 2], [10, 20])
    np.testing.assert_allclose(w, [10_000, 40_000])
    assert annual_revenue([1, 2], [[10, 20]])[0] == 50_000


def test_revenue_by_year_shapes_and_errors():
    out = revenue_by_year(np.ones(104), np.full((3, 104), 10.0))
    assert out.shape == (3, 2) and out[0, 0] == 52 * 1000 * 10
    with pytest.raises(ValueError):
        revenue_by_year(np.ones(100), np.full((1, 100), 10.0))
    with pytest.raises(ValueError):
        weekly_revenue([1, 2, 3], [10, 20])            # length mismatch
    with pytest.raises(ValueError):
        weekly_revenue([-1, 2], [10, 20])              # negative catch


def test_variance_depends_on_units_but_cv_does_not():
    ugx = [8e9, 9e9, 10e9, 11e9]
    billions = [v / 1e9 for v in ugx]
    a, b = describe_revenue(ugx), describe_revenue(billions)
    assert a["variance"] == pytest.approx(b["variance"] * 1e18)     # units: squared
    assert a["cv"] == pytest.approx(b["cv"])
    with pytest.raises(ValueError):
        describe_revenue([0, 0, 0])
    with pytest.raises(ValueError):
        describe_revenue([5])


# ------------------------------------------------------------------ RiskAssessor
@pytest.mark.parametrize("cv, label", [(0.0, "Low"), (0.099, "Low"), (0.10, "Moderate"),
                                       (0.249, "Moderate"), (0.25, "High"), (1.0, "High")])
def test_risk_classification_boundaries(cv, label):
    assert RiskAssessor().classify(cv) == label


def test_risk_assessor_validation():
    with pytest.raises(ValueError):
        RiskAssessor(low=0.3, high=0.2)
    with pytest.raises(ValueError):
        RiskAssessor(var_level=0.7)
    with pytest.raises(ValueError):
        RiskAssessor().classify(-0.1)


def test_var_is_mean_minus_fifth_percentile():
    rev = np.arange(1, 101, dtype=float)            # 1..100
    rep = RiskAssessor().assess(rev)
    assert rep.percentile_5 == pytest.approx(np.percentile(rev, 5))
    assert rep.value_at_risk == pytest.approx(rep.mean - rep.percentile_5)
    assert rep.var_fraction == pytest.approx(rep.value_at_risk / rep.mean)


def test_closed_season_lowers_catch_when_not_overfishing():
    fish = FishStock()
    _, open_c = fish.simulate(0.10, 260)
    _, shut_c = fish.simulate(0.10, 260, closed_weeks=8)
    assert shut_c.sum() < open_c.sum()
