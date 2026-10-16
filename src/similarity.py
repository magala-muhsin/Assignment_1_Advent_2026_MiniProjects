"""Similarity and distance measures between rainfall profiles."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _vector(a: ArrayLike) -> NDArray[np.float64]:
    v = np.asarray(a, dtype=float)
    if v.ndim != 1 or v.size == 0:
        raise ValueError("Inputs must be non-empty 1-D vectors.")
    if not np.all(np.isfinite(v)):
        raise ValueError("Inputs must be finite.")
    return v


def cosine_similarity(a: ArrayLike, b: ArrayLike) -> float:
    """cos(theta) between two vectors: ``a . b / (|a| |b|)``.

    This is the angle between the vectors, **not** ``math.cos`` of a number.
    Ranges from -1 to 1 (0 to 1 for non-negative data such as rainfall).
    """
    x, y = _vector(a), _vector(b)
    if x.shape != y.shape:
        raise ValueError("Vectors must have the same length.")
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if nx == 0 or ny == 0:
        raise ValueError("Cosine similarity is undefined for a zero vector.")
    return float(np.dot(x, y) / (nx * ny))


def _check_matrix(X: ArrayLike) -> NDArray[np.float64]:
    M = np.asarray(X, dtype=float)
    if M.ndim != 2 or M.shape[0] < 1 or M.shape[1] < 2:
        raise ValueError("Need a 2-D array: one row per region, at least 2 columns.")
    if not np.all(np.isfinite(M)):
        raise ValueError("Values must be finite.")
    return M


def cosine_similarity_matrix(X: ArrayLike) -> NDArray[np.float64]:
    """Pairwise cosine similarity between the rows of ``X``."""
    M = _check_matrix(X)
    norms = np.linalg.norm(M, axis=1)
    if np.any(norms == 0):
        raise ValueError("Cosine similarity is undefined for a zero row.")
    return (M @ M.T) / np.outer(norms, norms)


def pearson_matrix(X: ArrayLike) -> NDArray[np.float64]:
    """Pairwise Pearson correlation between the rows of ``X`` (shape of the profile)."""
    M = _check_matrix(X)
    if np.any(M.std(axis=1) == 0):
        raise ValueError("Pearson correlation is undefined for a constant row.")
    return np.corrcoef(M)


def euclidean_distance_matrix(X: ArrayLike) -> NDArray[np.float64]:
    """Pairwise Euclidean distance between the rows of ``X`` (in the units of X)."""
    M = _check_matrix(X)
    diff = M[:, None, :] - M[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))
