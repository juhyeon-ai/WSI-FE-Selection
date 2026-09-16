# Self_cluster.py
# Self-cluster score (Tsitsulin et al.)
#
# Goal:
#   Unsupervised representation compactness/diversity score for patch embeddings.
#   (Labels and SAM clusters are NOT used, but accepted to keep a unified interface.)
#
# Unified API:
#   self_cluster_score_slides(patch_embeddings, sam_clusters, labels, ...) -> float
#
# Notes:
#   - Per WSI: ALWAYS row-wise L2 normalize patch embeddings.
#   - Optional dimension unification per WSI: {"none","pca","gaussian"}
#   - Self-Cluster (Frobenius-norm squared formulation):
#       W in R^{n x d} (row-wise unit norm)
#       Q2 = || W W^T ||_F^2 = || W^T W ||_F^2
#       E[Q2] ≈ n + n(n-1)/d  (isotropic random unit vectors)
#       score = (Q2 - n - n(n-1)/d) / (n^2 - n - n(n-1)/d)  clipped to [0,1]
#
# Requirements: numpy

from __future__ import annotations
from typing import Literal
import numpy as np

from .config import DEFAULT_CONFIG, metric_kwargs

_SHARED = DEFAULT_CONFIG["shared"]
_DEFAULTS = metric_kwargs(DEFAULT_CONFIG, "self_cluster")


# =========================
# Preprocessing (per WSI)
# =========================
def l2_normalize_rows(X: np.ndarray, eps: float = _SHARED["eps"]) -> np.ndarray:
    X = X.astype(np.float64, copy=False)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (n + eps)


def gaussian_random_projection(
    X: np.ndarray,
    out_dim: int = _SHARED["reduce_dim"],
    seed: int = _SHARED["seed"],
    eps: float = _SHARED["eps"],
) -> np.ndarray:
    """
    X: (n, d) -> (n, out_dim), Gaussian random projection.
    NOTE: If d <= out_dim, returns X as-is (no up-projection).
    """
    n, d = X.shape
    if out_dim <= 0:
        raise ValueError(f"out_dim must be > 0, got {out_dim}")
    if d <= out_dim:
        return X.astype(np.float64, copy=False)

    rng = np.random.default_rng(seed)
    R = rng.normal(loc=0.0, scale=1.0, size=(d, out_dim)).astype(np.float64)
    Xp = X @ R
    Xp = Xp / (np.sqrt(out_dim) + eps)
    return Xp


def pca_project(
    X: np.ndarray,
    out_dim: int = _SHARED["reduce_dim"],
) -> np.ndarray:
    """
    PCA per WSI (fit on X) using SVD:
      Xc = U S V^T  -> PC scores = U[:, :m] * S[:m]
    If rank < out_dim, pads with zeros.
    NOTE: If min(n, d) <= out_dim, returns centered X (no up-projection).
    """
    n, d = X.shape
    if out_dim <= 0:
        raise ValueError(f"out_dim must be > 0, got {out_dim}")

    X = X.astype(np.float64, copy=False)
    Xc = X - X.mean(axis=0, keepdims=True)

    r = min(n, d)
    if r <= out_dim:
        return Xc

    try:
        U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    except np.linalg.LinAlgError:
        return Xc

    m = min(out_dim, U.shape[1])
    Xp = U[:, :m] * S[:m]

    if m < out_dim:
        pad = np.zeros((n, out_dim - m), dtype=Xp.dtype)
        Xp = np.concatenate([Xp, pad], axis=1)

    return Xp


def preprocess_wsi(
    Z: np.ndarray,
    eps: float = _SHARED["eps"],
    reduce_method: Literal["none", "pca", "gaussian"] = _DEFAULTS["reduce_method"],
    reduce_dim: int = _SHARED["reduce_dim"],
    reduce_seed: int = _SHARED["seed"],
) -> np.ndarray:
    """
    Per-WSI preprocessing:
      1) row-wise L2 normalization (ALWAYS)
      2) optional dimension unification
    """
    if Z.ndim != 2:
        raise ValueError(f"Z must be 2D, got shape={Z.shape}")

    # ALWAYS normalize
    W = l2_normalize_rows(Z, eps=eps)

    rm = str(reduce_method).lower().strip()
    if rm == "none":
        return W
    if rm == "pca":
        # PCA on normalized embeddings
        return pca_project(W, out_dim=reduce_dim)
    if rm == "gaussian":
        return gaussian_random_projection(W, out_dim=reduce_dim, seed=reduce_seed, eps=eps)

    raise ValueError("reduce_method must be one of: ['none','pca','gaussian']")


# =========================
# Self-Cluster (single WSI)
# =========================
def self_cluster_score_one(
    Z: np.ndarray,
    eps: float = _SHARED["eps"],
    reduce_method: Literal["none", "pca", "gaussian"] = _DEFAULTS["reduce_method"],
    reduce_dim: int = _SHARED["reduce_dim"],
    reduce_seed: int = _SHARED["seed"],
) -> float:
    """
    Self-Cluster score for one WSI patch embedding matrix Z (n, d).

    Returns:
      score in [0,1]
    """
    W = preprocess_wsi(
        Z,
        eps=eps,
        reduce_method=reduce_method,
        reduce_dim=reduce_dim,
        reduce_seed=reduce_seed,
    )

    n, d = W.shape
    if n <= 1 or d <= 1:
        return 0.0

    # Efficient: ||W W^T||_F^2 == ||W^T W||_F^2
    A = W.T @ W  # (d, d)
    Q2 = float(np.sum(A * A))  # Frobenius norm squared (no sqrt)

    exp_term = (n * (n - 1)) / float(d)
    den = (n * n) - n - exp_term
    if abs(den) <= eps:
        return 0.0

    num = Q2 - n - exp_term
    sc = num / den
    if np.isnan(sc) or np.isinf(sc):
        return 0.0

    return float(np.clip(sc, 0.0, 1.0))


# =========================
# Unified API (LP / SAM_LP style)
# =========================
def self_cluster_score_slides(
    patch_embeddings: list[np.ndarray],
    sam_clusters: list[np.ndarray],  # accepted for API alignment; NOT used
    labels: np.ndarray,             # accepted for API alignment; NOT used
    *,
    eps: float = _SHARED["eps"],
    reduce_method: Literal["none", "pca", "gaussian"] = _DEFAULTS["reduce_method"],
    reduce_dim: int = _SHARED["reduce_dim"],
    reduce_seed: int = _SHARED["seed"],
    aggregate: Literal["mean", "median"] = _DEFAULTS["aggregate"],
) -> float:
    """
    Unified interface:
      - patch_embeddings: list of (n_i, d)
      - sam_clusters:     list of (n_i,)  (unused)
      - labels:           (N,)           (unused)

    Returns:
      aggregate(SelfCluster(Z_i)) across slides.
    """
    if len(patch_embeddings) != len(sam_clusters):
        raise ValueError("patch_embeddings and sam_clusters must have the same length")

    y = np.asarray(labels).reshape(-1)
    if y.shape[0] != len(patch_embeddings):
        raise ValueError("labels length must match number of slides (even though labels are not used)")

    vals = []
    for Z in patch_embeddings:
        vals.append(
            self_cluster_score_one(
                Z,
                eps=eps,
                reduce_method=reduce_method,
                reduce_dim=reduce_dim,
                reduce_seed=reduce_seed,
            )
        )

    if len(vals) == 0:
        return 0.0

    v = np.asarray(vals, dtype=np.float64)
    if aggregate == "median":
        return float(np.median(v))
    if aggregate == "mean":
        return float(np.mean(v))
    raise ValueError("aggregate must be one of: ['mean','median']")