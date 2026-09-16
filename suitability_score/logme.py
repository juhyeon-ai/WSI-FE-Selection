# LogME.py
# LogME Score (classification) with Gaussian Projection + single L2 normalization
#
# Definition:
#   1) Slide embedding = mean pooling of patch embeddings
#   2) Optional Gaussian random projection to fixed dimension
#   3) L2-normalize slide embeddings (once, after projection)
#   4) Compute LogME score: LogME().fit(X, y)  (higher is better)
#
# Requirements: numpy, numba

from __future__ import annotations
import numpy as np

from .config import DEFAULT_CONFIG, metric_kwargs

_SHARED = DEFAULT_CONFIG["shared"]
_DEFAULTS = metric_kwargs(DEFAULT_CONFIG, "logme")
from numba import njit


# =========================
# Slide embedding
# =========================
def mean_pool_slide_embeddings(patch_embeddings: list[np.ndarray]) -> np.ndarray:
    """
    Args:
        patch_embeddings: list of arrays, each (K_i, d)

    Returns:
        X: (N, d) slide embeddings
    """
    slides = []
    for Z in patch_embeddings:
        if Z.ndim != 2:
            raise ValueError(f"Patch embedding must be 2D, got {Z.shape}")
        slides.append(Z.mean(axis=0))
    return np.stack(slides, axis=0).astype(np.float64, copy=False)


# =========================
# Gaussian projection
# =========================
_GAUSS_CACHE = {}  # (seed, in_dim, out_dim) -> projection matrix


def gaussian_projection(X: np.ndarray, out_dim: int = _SHARED["reduce_dim"], seed: int = _SHARED["seed"]) -> np.ndarray:
    """
    Random Gaussian projection:
        R ~ N(0, 1/out_dim)
        X_proj = X @ R

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

    key = (int(seed), int(in_dim), int(out_dim))
    R = _GAUSS_CACHE.get(key)
    if R is None:
        rng = np.random.RandomState(int(seed))
        R = rng.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(out_dim),
            size=(in_dim, out_dim),
        ).astype(np.float64)
        _GAUSS_CACHE[key] = R

    return X @ R


# =========================
# L2 normalization (single)
# =========================
def l2_normalize_rows(X: np.ndarray, eps: float = _SHARED["eps"]) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return X / norms


# ============================================================
# LogME core (kept intact, classification mode)
# ============================================================
@njit
def each_evidence(y_, f, fh, v, s, vh, N, D):
    epsilon = 1e-5
    alpha = 1.0
    beta = 1.0
    lam = alpha / beta
    tmp = (vh @ (f @ np.ascontiguousarray(y_)))
    for _ in range(11):
        gamma = (s / (s + lam)).sum()
        m = v @ (tmp * beta / (alpha + beta * s))
        alpha_de = (m * m).sum()
        alpha = gamma / (alpha_de + epsilon)
        beta_de = ((y_ - fh @ m) ** 2).sum()
        beta = (N - gamma) / (beta_de + epsilon)
        new_lam = alpha / beta
        if np.abs(new_lam - lam) / lam < 0.01:
            break
        lam = new_lam
    evidence = D / 2.0 * np.log(alpha) \
               + N / 2.0 * np.log(beta) \
               - 0.5 * np.sum(np.log(alpha + beta * s)) \
               - beta / 2.0 * (beta_de + epsilon) \
               - alpha / 2.0 * (alpha_de + epsilon) \
               - N / 2.0 * np.log(2 * np.pi)
    return evidence / N, alpha, beta, m


@njit
def truncated_svd(x):
    u, s, vh = np.linalg.svd(x.transpose() @ x)
    s = np.sqrt(s)
    u_times_sigma = x @ vh.transpose()
    k = np.sum((s > 1e-10) * 1)
    s = s.reshape(-1, 1)
    s = s[:k]
    vh = vh[:k]
    u = u_times_sigma[:, :k] / s.reshape(1, -1)
    return u, s, vh


class LogME(object):
    """
    LogME implementation for classification.
    Score: mean evidence over classes (higher is better).
    """
    def __init__(self, regression: bool = False):
        self.regression = regression
        self.fitted = False
        self.reset()

    def reset(self):
        self.num_dim = 0
        self.alphas = []
        self.betas = []
        self.ms = []

    def _fit_fixed_point(self, f: np.ndarray, y: np.ndarray):
        N, D = f.shape
        if N > D:
            u, s, vh = truncated_svd(f)
        else:
            u, s, vh = np.linalg.svd(f, full_matrices=False)

        s = s.reshape(-1, 1)
        sigma = (s ** 2)

        evidences = []
        self.num_dim = y.shape[1] if self.regression else int(y.max() + 1)
        for i in range(self.num_dim):
            y_ = y[:, i] if self.regression else (y == i).astype(np.float64)
            y_ = y_.reshape(-1, 1)
            x = u.T @ y_
            x2 = x ** 2
            res_x2 = (y_ ** 2).sum() - x2.sum()

            alpha, beta = 1.0, 1.0
            for _ in range(11):
                t = alpha / beta
                gamma = (sigma / (sigma + t)).sum()
                m2 = (sigma * x2 / ((t + sigma) ** 2)).sum()
                res2 = (x2 / ((1 + sigma / t) ** 2)).sum() + res_x2
                alpha = gamma / (m2 + 1e-5)
                beta = (N - gamma) / (res2 + 1e-5)
                t_ = alpha / beta
                if abs(t_ - t) / t <= 1e-3:
                    break

            evidence = D / 2.0 * np.log(alpha) \
                       + N / 2.0 * np.log(beta) \
                       - 0.5 * np.sum(np.log(alpha + beta * sigma)) \
                       - beta / 2.0 * res2 \
                       - alpha / 2.0 * m2 \
                       - N / 2.0 * np.log(2 * np.pi)
            evidence /= N
            m = 1.0 / (t + sigma) * s * x
            m = (vh.T @ m).reshape(-1)
            evidences.append(evidence)
            self.alphas.append(alpha)
            self.betas.append(beta)
            self.ms.append(m)

        self.ms = np.stack(self.ms)
        return np.mean(evidences)

    _fit = _fit_fixed_point

    def fit(self, f: np.ndarray, y: np.ndarray):
        if self.fitted:
            self.reset()
        else:
            self.fitted = True

        f = f.astype(np.float64, copy=False)

        if self.regression:
            y = y.astype(np.float64, copy=False)
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)

        return self._fit(f, y)


# =========================
# Optional: numba warmup
# =========================
_WARMED_UP = False


def warmup_logme_numba(verbose: bool = False) -> None:
    """
    Optional: pre-compile numba kernels to remove first-call overhead.
    """
    global _WARMED_UP
    if _WARMED_UP:
        return
    if verbose:
        print("[LogME] Numba warmup compile...")

    f_tmp = np.random.randn(20, 50).astype(np.float64)
    y_tmp = np.random.randint(0, 2, 50).astype(np.float64)
    each_evidence(
        y_tmp, f_tmp, f_tmp.T,
        np.eye(20, dtype=np.float64),
        np.ones(20, dtype=np.float64),
        np.eye(20, dtype=np.float64),
        50, 20
    )
    truncated_svd(np.random.randn(20, 10).astype(np.float64))
    _WARMED_UP = True

    if verbose:
        print("[LogME] Warmup done.")


# =========================
# Public API: LogME score
# =========================
def logme_score(
    patch_embeddings: list[np.ndarray],
    labels: np.ndarray,
    reduce_dim: int | None = _SHARED["reduce_dim"],
    reduce_seed: int = _SHARED["seed"],
    eps_norm: float = _SHARED["eps"],
    warmup_numba: bool = _DEFAULTS["warmup_numba"],
) -> float:
    """
    Compute LogME score (classification).

    Args:
        patch_embeddings: list of (K_i, d) arrays
        labels: (N,) integer labels in [0..C-1]
        reduce_dim: if not None, apply Gaussian projection to this dimension
        reduce_seed: seed for Gaussian projection
        eps_norm: numerical stability for L2 norm
        warmup_numba: if True, run warmup compilation

    Returns:
        LogME score (float), higher is better
    """
    if warmup_numba:
        warmup_logme_numba(verbose=False)

    # 1) Mean pooling
    X = mean_pool_slide_embeddings(patch_embeddings)

    # 2) Optional Gaussian projection
    if reduce_dim is not None:
        X = gaussian_projection(X, out_dim=int(reduce_dim), seed=int(reduce_seed))

    # 3) L2 normalize ONCE (after projection)
    X = l2_normalize_rows(X, eps=eps_norm)

    y = np.asarray(labels).reshape(-1).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")

    # 4) LogME
    lm = LogME(regression=False)
    return float(lm.fit(X, y))