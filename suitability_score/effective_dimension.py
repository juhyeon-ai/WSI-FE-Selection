# Effective_dimension.py
# PCA explained-variance effective dimension (higher-is-better)
#
# Definition:
#   Given patch embeddings Z ∈ R^{K×d}, the effective dimension k is defined as
#   the smallest integer k such that the cumulative explained variance of the
#   top-k eigenvalues of the covariance matrix reaches a target ratio (e.g., 0.99).
#
# Requirements: numpy only.

from __future__ import annotations
import numpy as np

from .config import DEFAULT_CONFIG, metric_kwargs

_SHARED = DEFAULT_CONFIG["shared"]
_DEFAULTS = metric_kwargs(DEFAULT_CONFIG, "effective_dimension")


def pca_effective_dim(
    Z: np.ndarray,
    explained_var: float = _DEFAULTS["explained_var"],
    center: bool = _DEFAULTS["center"],
    eps: float = _SHARED["eps"],
) -> int:
    """
    PCA-based effective dimension (integer).

    Steps:
      1) (optional) Center embeddings across patches.
      2) Compute covariance: C = (X^T X) / K.
      3) Eigen-decompose C (symmetric PSD).
      4) Return minimal k achieving target explained variance.

    Args:
        Z: (K, d) patch embeddings
        explained_var: target ratio in (0,1], e.g., 0.99
        center: subtract mean across patches
        eps: numerical stability

    Returns:
        k (int) in [0, d]
    """
    Z = np.asarray(Z)
    if Z.ndim != 2:
        raise ValueError(f"Z must be 2D, got shape={Z.shape}")

    K, d = Z.shape
    if K <= 1 or d <= 0:
        return 0

    if not (0.0 < explained_var <= 1.0):
        raise ValueError(f"explained_var must be in (0,1], got {explained_var}")

    X = Z.astype(np.float64, copy=False)

    if center:
        X = X - X.mean(axis=0, keepdims=True)

    # Covariance matrix (d × d)
    C = (X.T @ X) / float(K)

    # Eigenvalues (descending)
    evals = np.linalg.eigvalsh(C)[::-1]
    evals = np.maximum(evals, 0.0)

    total = float(np.sum(evals))
    if total <= eps:
        return 0

    cumulative_ratio = np.cumsum(evals) / total
    k = int(np.searchsorted(cumulative_ratio, explained_var, side="left") + 1)

    return min(k, d)