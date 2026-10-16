"""Domain classes for Mini-Project 2: the Kasese health-centre micro-grid."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import solve


class MicroGrid:
    """Daily energy-dispatch system ``A @ sources = demands``.

    The default system is::

        3x + 2y = D1   (daytime load, kWh)
        4x +  y = D2   (critical-equipment load, kWh)

    with ``x`` = solar kWh and ``y`` = battery kWh.

    Parameters
    ----------
    coefficients:
        Square coefficient matrix (one row per load equation).
    source_names:
        Names of the unknowns, one per column.

    Raises
    ------
    ValueError
        If the matrix is not a finite square 2-D array, or the number of
        source names does not match its size.
    """

    COND_LIMIT: float = 1e12  # above this a solve is not trustworthy

    def __init__(
        self,
        coefficients: ArrayLike = ((3, 2), (4, 1)),
        source_names: Sequence[str] = ("solar", "battery"),
    ) -> None:
        A = np.asarray(coefficients, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("The coefficient matrix must be square (2-D).")
        if not np.all(np.isfinite(A)):
            raise ValueError("Coefficients must be finite numbers.")
        if len(source_names) != A.shape[0]:
            raise ValueError("Need one source name per unknown.")
        self.A: NDArray[np.float64] = A
        self.source_names: tuple[str, ...] = tuple(source_names)

    # ---------------------------------------------------------------- dunders
    def __repr__(self) -> str:
        return f"{type(self).__name__}(sources={self.source_names}, A={self.A.tolist()})"

    # ------------------------------------------------------------ diagnostics
    @property
    def n_sources(self) -> int:
        """Number of unknown energy sources."""
        return self.A.shape[0]

    @property
    def determinant(self) -> float:
        """Determinant of A (zero => singular, no unique solution)."""
        return float(np.linalg.det(self.A))

    @property
    def condition_number(self) -> float:
        """2-norm condition number: worst-case amplification of relative input error."""
        return float(np.linalg.cond(self.A))

    @property
    def rank(self) -> int:
        """Matrix rank (< n_sources means linearly dependent equations)."""
        return int(np.linalg.matrix_rank(self.A))

    def is_well_posed(self) -> bool:
        """True if A has full rank and a finite, moderate condition number."""
        cond = self.condition_number
        return self.rank == self.n_sources and bool(np.isfinite(cond)) and cond < self.COND_LIMIT

    def diagnostics(self) -> dict[str, float | int | bool]:
        """Determinant, condition number, rank and the well-posed verdict."""
        return {
            "determinant": self.determinant,
            "condition_number": self.condition_number,
            "rank": self.rank,
            "well_posed": self.is_well_posed(),
        }

    # ---------------------------------------------------------------- solving
    def _validate(self, demands: ArrayLike) -> NDArray[np.float64]:
        b = np.asarray(demands, dtype=float)
        if b.ndim not in (1, 2) or b.shape[0] != self.n_sources:
            raise ValueError(
                f"Demands must have {self.n_sources} rows (one per load equation)."
            )
        if b.size == 0 or (b.ndim == 2 and b.shape[1] == 0):
            raise ValueError("Demands must not be empty.")
        if not np.all(np.isfinite(b)):
            raise ValueError("Demands must be finite numbers.")
        if np.any(b < 0):
            raise ValueError("Demands cannot be negative.")
        if not self.is_well_posed():
            raise np.linalg.LinAlgError(
                "System is singular (linearly dependent equations) or too "
                "ill-conditioned to solve reliably."
            )
        return b

    def solve(self, demands: ArrayLike) -> NDArray[np.float64]:
        """Solve for one day (1-D demands) or many days (2-D, one column per day)."""
        return solve(self.A, self._validate(demands))

    def solve_day(self, *demands: float) -> NDArray[np.float64]:
        """Solve one day, e.g. ``grid.solve_day(d1, d2)``; returns the source vector."""
        return self.solve(list(demands))

    def solve_loop(self, demands: ArrayLike) -> NDArray[np.float64]:
        """Solve every column of ``demands`` (n x days) one at a time in a Python loop."""
        B = self._validate(demands)
        if B.ndim != 2:
            raise ValueError("solve_loop expects a 2-D (n x days) array.")
        return np.column_stack([solve(self.A, B[:, j]) for j in range(B.shape[1])])

    def solve_batch(self, demands: ArrayLike) -> NDArray[np.float64]:
        """Solve all days in ONE call by passing an n x days right-hand side."""
        B = self._validate(demands)
        if B.ndim != 2:
            raise ValueError("solve_batch expects a 2-D (n x days) array.")
        return solve(self.A, B)


class HybridMicroGrid(MicroGrid):
    """Micro-grid with a diesel generator ``z`` and a third balance equation.

    ::

        3x + 2y + 0z = D1
        4x +  y + 0z = D2
        a*x + b*y + c*z = D3     (default row: x + y + z = D3, total generation)

    Parameters
    ----------
    third_row:
        Coefficients of the third equation. A row that is a combination of the
        first two (with no ``z`` term) makes the system singular.
    """

    def __init__(self, third_row: Sequence[float] = (1, 1, 1)) -> None:
        if len(third_row) != 3:
            raise ValueError("third_row needs exactly three coefficients.")
        super().__init__(
            [[3, 2, 0], [4, 1, 0], list(third_row)],
            source_names=("solar", "battery", "diesel"),
        )
