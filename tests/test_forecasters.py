import numpy as np
import pytest

from src.forecasters import (
    CAGRForecaster,
    FibonacciRatioForecaster,
    LinearTrendForecaster,
    fibonacci_ratios,
)

YEARS = np.arange(2015, 2020)


def test_linear_recovers_exact_line():
    pred = LinearTrendForecaster().fit(YEARS, 10 + 5 * (YEARS - 2015)).predict(2)
    np.testing.assert_allclose(pred, [35, 40])


def test_cagr_recovers_exact_exponential():
    vals = 100 * 1.05 ** np.arange(5)
    m = CAGRForecaster().fit(YEARS, vals)
    assert m.rate == pytest.approx(0.05)
    np.testing.assert_allclose(m.predict(2), vals[-1] * 1.05 ** np.array([1, 2]))


def test_fibonacci_ratios_sequence():
    np.testing.assert_allclose(fibonacci_ratios(4), [1, 2, 1.5, 5 / 3])


def test_fibonacci_modes():
    vals = [100, 100, 100, 100, 100]
    plain = FibonacciRatioForecaster().fit(YEARS, vals).predict(3)
    comp = FibonacciRatioForecaster(compound=True).fit(YEARS, vals).predict(3)
    np.testing.assert_allclose(plain, [100, 200, 150])
    np.testing.assert_allclose(comp, [100, 200, 300])


def test_future_years():
    m = LinearTrendForecaster().fit(YEARS, YEARS * 2.0)
    assert list(m.future_years(3)) == [2020, 2021, 2022]


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        CAGRForecaster().predict(3)


@pytest.mark.parametrize("h", [0, -1])
def test_bad_horizon_value(h):
    with pytest.raises(ValueError):
        LinearTrendForecaster().fit(YEARS, YEARS * 1.0).predict(h)


def test_bad_horizon_type():
    with pytest.raises(TypeError):
        LinearTrendForecaster().fit(YEARS, YEARS * 1.0).predict(2.5)


def test_cagr_rejects_zero_start():
    with pytest.raises(ValueError):
        CAGRForecaster().fit([2015, 2016], [0, 5])


def test_fit_rejects_negative_and_short_input():
    with pytest.raises(ValueError):
        LinearTrendForecaster().fit([2015, 2016], [1, -1])
    with pytest.raises(ValueError):
        LinearTrendForecaster().fit([2015], [1])
