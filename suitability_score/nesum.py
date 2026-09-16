# NESum.py
# NESum (He et al. 2022-style) with optional Gaussian projection
#
# Definition:
#   Given patch embeddings Z ∈ R^{K×d} for a slide,
#   1) Optional Gaussian projection to fixed dimension (data-independent)
#   2) Patch-wise L2 normalization (once, after projection)
#   3) Center across patches
#   4) C = (1/K) X^T X
#   5) NESum(Z) = trace(C) / lambda_1(C)
#
# Requirements: numpy only.

from __future__ import annotations
import numpy as np

from .config import DEFAULT_CONFIG, metric_kwargs

_SHARED = DEFAULT_CONFIG["shared"]
_DEFAULTS = metric_kwargs(DEFAULT_CONFIG, "nesum")


# =========================
# L2 Normalization (patch-wise)
# =========================
def l2_normalize_rows(X: np.ndarray, eps: float = _SHARED["eps"]) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return X / norms


# =========================
# Gaussian Projection (data-independent)
# =========================
def gaussian_random_projection(
    X: np.ndarray,
    out_dim: int = _SHARED["reduce_dim"],
    seed: int = _SHARED["seed"],
    eps: float = _SHARED["eps"],
) -> np.ndarray:
    """
    Random Gaussian projection:
        R ~ N(0, 1), then Xp = (X @ R) / sqrt(out_dim)

    Args:
        X: (K, d)
        out_dim: projected dimension
        seed: random seed
        eps: numerical stability

    Returns:
        (K, out_dim)
    """
    K, d = X.shape
    if out_dim >= d:
        return X.astype(np.float64, copy=False)

    rng = np.random.default_rng(seed)
    R = rng.normal(loc=0.0, scale=1.0, size=(d, out_dim)).astype(np.float64)
    Xp = X @ R
    Xp = Xp / (np.sqrt(out_dim) + eps)
    return Xp


# =========================
# NESum Score
# =========================
def nesum_score(
    Z: np.ndarray,
    eps: float = _SHARED["eps"],
    center: bool = _DEFAULTS["center"],
    # preprocessing
    reduce_dim: int | None = _SHARED["reduce_dim"],
    reduce_seed: int = _SHARED["seed"],
) -> float:
    """
    Compute NESum for one slide.

    Args:
        Z: (K, d) patch embeddings
        eps: numerical stability
        center: if True, subtract mean across patches before covariance
        reduce_dim: if not None and < d, apply Gaussian projection to this dim
        reduce_seed: seed for Gaussian projection

    Returns:
        NESum value (float, >=0)
    """
    Z = np.asarray(Z)
    if Z.ndim != 2:
        raise ValueError(f"Z must be 2D, got shape={Z.shape}")

    K, d = Z.shape
    if K <= 1 or d <= 1:
        return 0.0

    X = Z.astype(np.float64, copy=False)

    # 1) Optional Gaussian projection
    if reduce_dim is not None and reduce_dim > 0 and X.shape[1] > reduce_dim:
        X = gaussian_random_projection(X, out_dim=int(reduce_dim), seed=int(reduce_seed), eps=eps)

    # 2) Patch-wise L2 normalization (ONLY once, after projection)
    X = l2_normalize_rows(X, eps=eps)

    # 3) Optional centering
    if center:
        X = X - X.mean(axis=0, keepdims=True)

    # 4) trace(C) = (1/K) * ||X||_F^2
    fro2 = float(np.sum(X * X))
    if fro2 <= eps:
        return 0.0
    traceC = fro2 / float(K)

    # 5) Largest eigenvalue of C = (1/K) X^T X
    C = (X.T @ X) / float(K)  # (d, d)
    evals = np.linalg.eigvalsh(C)  # ascending
    lam1 = float(evals[-1])

    if not np.isfinite(lam1) or lam1 <= eps:
        return 0.0

    val = traceC / lam1
    if not np.isfinite(val):
        return 0.0
    return float(max(val, 0.0))