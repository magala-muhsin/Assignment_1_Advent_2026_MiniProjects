import numpy as np
import pytest

from src.district import DistrictPopulation


def make():
    return DistrictPopulation("Gulu", [2015, 2016, 2017], [100, 110, 121])


def test_len_and_repr():
    d = make()
    assert len(d) == 3
    assert "Gulu" in repr(d) and "2015-2017" in repr(d)


def test_growth_and_cagr():
    d = make()
    np.testing.assert_allclose(d.growth_rates(), [0.10, 0.10])
    assert d.cagr() == pytest.approx(0.10)


def test_statistics_matches_numpy_ddof1():
    d = make()
    s, n = d.describe_statistics(), d.describe_numpy(ddof=1)
    assert s["variance"] == pytest.approx(n["variance"])
    assert d.describe_numpy(ddof=0)["variance"] < n["variance"]


def test_split():
    d = make()
    ty, tv, ey, ev = d.split(2016)
    assert list(ty) == [2015, 2016] and list(ey) == [2017] and ev[0] == 121


@pytest.mark.parametrize(
    "years, pops",
    [
        ([2015, 2016], [100]),            # unequal length
        ([2015, 2016], [100, -5]),        # negative
        ([], []),                         # empty
        ([2016, 2015], [1, 2]),           # not increasing
        ([2015.5, 2016.5], [1, 2]),       # non-integer years
        ([2015, 2016], [1, np.nan]),      # non-finite
    ],
)
def test_invalid_inputs_raise(years, pops):
    with pytest.raises(ValueError):
        DistrictPopulation("X", years, pops)


def test_cagr_zero_population_raises():
    with pytest.raises(ValueError):
        DistrictPopulation("X", [2015, 2016], [0, 5]).cagr()
