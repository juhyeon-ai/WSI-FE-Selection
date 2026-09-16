# Linear_probing.py
# Linear Probing Score with Gaussian Projection (single L2 normalization)
#
# Definition:
#   1) Slide embedding = mean pooling of patch embeddings
#   2) Optional Gaussian random projection
#   3) L2-normalize slide embeddings
#   4) Train multinomial logistic regression
#   5) Return average log-likelihood (train == eval)
#
# Requirements: numpy, scikit-learn

from __future__ import annotations
import numpy as np

from .config import DEFAULT_CONFIG, metric_kwargs

_SHARED = DEFAULT_CONFIG["shared"]
_DEFAULTS = metric_kwargs(DEFAULT_CONFIG, "linear_probing")


# =========================
# L2 Normalization
# =========================
def l2_normalize_rows(X: np.ndarray, eps: float = _SHARED["eps"]) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return X / norms


# =========================
# Gaussian Projection
# =========================
_GAUSS_CACHE = {}  # (seed, in_dim, out_dim) -> projection matrix


def gaussian_projection(
    X: np.ndarray,
    out_dim: int = _SHARED["reduce_dim"],
    seed: int = _SHARED["seed"],
) -> np.ndarray:
    """
    Random Gaussian projection:
        R ~ N(0, 1/out_dim)

    Args:
        X: (N, d)
        out_dim: projected dimension
        seed: random seed

    Returns:
        (N, out_dim)
    """
    N, in_dim = X.shape
    if out_dim >= in_dim:
        return X

    key = (seed, in_dim, out_dim)
    R = _GAUSS_CACHE.get(key)

    if R is None:
        rng = np.random.RandomState(seed)
        R = rng.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(out_dim),
            size=(in_dim, out_dim),
        ).astype(np.float64)
        _GAUSS_CACHE[key] = R

    return X @ R


# =========================
# Slide Embedding
# =========================
def mean_pool_slide_embeddings(
    patch_embeddings: list[np.ndarray]
) -> np.ndarray:
    slides = []
    for Z in patch_embeddings:
        if Z.ndim != 2:
            raise ValueError(f"Patch embedding must be 2D, got {Z.shape}")
        slides.append(Z.mean(axis=0))
    return np.stack(slides, axis=0)


# =========================
# Linear Probing Score
# =========================
def linear_probing_score(
    patch_embeddings: list[np.ndarray],
    labels: np.ndarray,
    C: float = _DEFAULTS["C"],
    max_iter: int = _DEFAULTS["max_iter"],
    seed: int = _SHARED["seed"],
    reduce_dim: int | None = _SHARED["reduce_dim"],
    reduce_seed: int = _SHARED["seed"],
    eps: float = _SHARED["eps"],
) -> float:
    """
    Linear Probing Score (average log-likelihood).

    Args:
        patch_embeddings: list of (K_i, d)
        labels: (N,) integer labels
        C: inverse regularization strength
        max_iter: solver iterations
        seed: logistic regression seed
        reduce_dim: if not None, apply Gaussian projection
        reduce_seed: seed for projection
        eps: numerical stability

    Returns:
        Average log-likelihood (float)
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception as e:
        raise ImportError(
            "scikit-learn is required for linear probing."
        ) from e

    # 1) Mean pooling
    X = mean_pool_slide_embeddings(patch_embeddings)

    # 2) Optional Gaussian projection
    if reduce_dim is not None:
        X = gaussian_projection(X, out_dim=reduce_dim, seed=reduce_seed)

    # 3) L2 normalize (ONLY once, after projection)
    X = l2_normalize_rows(X, eps=eps)

    y = np.asarray(labels).reshape(-1).astype(int)

    if len(np.unique(y)) < 2:
        return float("-inf")

    # 4) Logistic regression
    clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver="lbfgs",
        random_state=seed,
    )
    clf.fit(X, y)

    # 5) Log-likelihood
    prob = clf.predict_proba(X)
    cls = clf.classes_
    idx = np.searchsorted(cls, y)
    p = prob[np.arange(len(y)), idx]
    p = np.clip(p, 1e-12, 1.0)

    return float(np.mean(np.log(p)))