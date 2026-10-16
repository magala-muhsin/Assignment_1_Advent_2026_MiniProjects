import numpy as np
import pytest

from src.demand import generate_demand, load_demand_csv, prompt_nonnegative_number
from src.dispatch import (energy_costs, resolve_dispatch, sensitivity_analysis,
                          source_statistics)
from src.microgrid import HybridMicroGrid, MicroGrid


# ------------------------------------------------------------------ MicroGrid
def test_determinant_and_condition_number():
    g = MicroGrid()
    assert g.determinant == pytest.approx(-5.0)
    assert g.condition_number == pytest.approx(5.8284, abs=1e-3)
    assert g.is_well_posed()


def test_solve_day_known_answer():
    x, y = MicroGrid().solve_day(100, 80)       # 3*12+2*32=100, 4*12+32=80
    assert (x, y) == (pytest.approx(12.0), pytest.approx(32.0))


@pytest.mark.parametrize("bad", [(-1, 5), (np.nan, 5), (np.inf, 5)])
def test_invalid_demands_rejected(bad):
    with pytest.raises(ValueError):
        MicroGrid().solve_day(*bad)


def test_wrong_number_of_demands_rejected():
    with pytest.raises(ValueError):
        MicroGrid().solve_day(1, 2, 3)


def test_loop_and_batch_agree():
    B = generate_demand(seed=1)[["D1_kWh", "D2_kWh"]].to_numpy().T
    g = MicroGrid()
    np.testing.assert_allclose(g.solve_loop(B), g.solve_batch(B))
    assert g.solve_batch(B).shape == (2, 30)


def test_non_square_matrix_rejected():
    with pytest.raises(ValueError):
        MicroGrid([[1, 2, 3], [4, 5, 6]], source_names=("a", "b"))


# ------------------------------------------------------------------ Hybrid
def test_hybrid_solves_three_by_three():
    sol = HybridMicroGrid().solve_day(54, 62, 23)   # x=14, y=6, z=3
    np.testing.assert_allclose(sol, [14, 6, 3])
    assert isinstance(HybridMicroGrid(), MicroGrid)


def test_dependent_third_equation_is_singular():
    g = HybridMicroGrid(third_row=(7, 3, 0))        # row3 = row1 + row2
    assert g.rank == 2 and not g.is_well_posed()
    with pytest.raises(np.linalg.LinAlgError):
        g.solve_day(100, 80, 180)


# ------------------------------------------------------------------ Data / input
def test_generate_demand_reproducible_and_nonnegative():
    a, b = generate_demand(seed=7), generate_demand(seed=7)
    assert a.equals(b) and len(a) == 30
    assert (a[["D1_kWh", "D2_kWh"]] >= 0).all().all()
    with pytest.raises(ValueError):
        generate_demand(n_days=0)


def test_prompt_rejects_bad_input_then_accepts():
    answers = iter(["", "abc", "-4", "nan", "inf", "57.5"])
    msgs = []
    value = prompt_nonnegative_number("D1: ", input_fn=lambda _: next(answers), print_fn=msgs.append)
    assert value == 57.5 and len(msgs) == 5


def test_load_csv_roundtrip_and_errors(tmp_path):
    df = generate_demand(n_days=5)
    path = tmp_path / "d.csv"
    df.to_csv(path, index=False)
    _, B = load_demand_csv(path)
    assert B.shape == (2, 5)

    (tmp_path / "bad.csv").write_text("D1_kWh,D2_kWh\n1,-2\n")
    with pytest.raises(ValueError):
        load_demand_csv(tmp_path / "bad.csv")
    (tmp_path / "empty.csv").write_text("D1_kWh,D2_kWh\n")
    with pytest.raises(ValueError):
        load_demand_csv(tmp_path / "empty.csv")
    (tmp_path / "cols.csv").write_text("a,b\n1,2\n")
    with pytest.raises(ValueError):
        load_demand_csv(tmp_path / "cols.csv")


# ------------------------------------------------------------------ Dispatch
def test_infeasible_day_flagged_and_repaired():
    # D1=60, D2=90 gives x=24, y=-6 (infeasible). NNLS must return y=0.
    B = np.array([[100.0, 60.0], [80.0, 90.0]])
    r = resolve_dispatch(MicroGrid(), B)
    assert list(r.infeasible) == [False, True]
    assert np.all(r.dispatch >= 0)
    np.testing.assert_allclose(r.dispatch[:, 1], [21.6, 0.0], atol=1e-6)
    assert r.residual_norm[0] == pytest.approx(0, abs=1e-9)
    assert r.residual_norm[1] == pytest.approx(6.0)


def test_nnls_residual_not_worse_than_clipping():
    B = generate_demand()[["D1_kWh", "D2_kWh"]].to_numpy().T
    g = MicroGrid()
    assert resolve_dispatch(g, B, "nnls").residual_norm.sum() <= resolve_dispatch(g, B, "clip").residual_norm.sum() + 1e-9
    with pytest.raises(ValueError):
        resolve_dispatch(g, B, "magic")


def test_statistics_and_cost_by_hand():
    s = source_statistics([10, 20, 30])
    assert s["mean"] == 20 and s["variance"] == 100 and s["cv_percent"] == pytest.approx(50)
    with pytest.raises(ValueError):
        source_statistics([0, 0, 0])
    c = energy_costs([10, 20], [1, 2])               # 10*150+1*450, 20*150+2*450
    np.testing.assert_allclose(c["daily"], [1950, 3900])
    assert c["monthly_total"] == 5850 and c["battery_total"] == 1350


def test_sensitivity_amplification_bounded_by_condition_number():
    B = generate_demand()[["D1_kWh", "D2_kWh"]].to_numpy().T
    res = sensitivity_analysis(MicroGrid(), B, n_draws=200, seed=3)
    assert 0 < res.amplification_max <= res.condition_number + 1e-9
    assert np.all(res.mean_move > 0) and res.draws.shape == (200, 2, 30)
    with pytest.raises(ValueError):
        sensitivity_analysis(MicroGrid(), B, pct=1.5)
