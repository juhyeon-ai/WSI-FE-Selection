# SAM_LP.py
#
# Definition:
#   Given per-slide patch embeddings Z_i ∈ R^{n_i×d}, SAM cluster ids c_i ∈ Z^{n_i},
#   and slide labels y_i:
#
#   (0) Initialize slide embeddings by mean pooling patches.
#   (1) Fit a multinomial logistic regression classifier on slide embeddings (train==eval).
#   (2) For each slide, compute SAM-cluster centroids, score each centroid by log p(y_i | centroid),
#       and re-embed slide using selected centroids (topK or soft weighting).
#   (3) Refit classifier on re-embedded slides; repeat for em_iters steps.
#   (4) Return average log-likelihood on final slide embeddings (train==eval).
#
# Preprocessing (kept minimal, matching your LP style):
#   - Optional: Gaussian random projection to fixed dimension
#   - L2-normalize ONCE after projection (row-wise)
#
# Requirements: numpy, scikit-learn

from __future__ import annotations
import numpy as np

from .config import DEFAULT_CONFIG, metric_kwargs

_SHARED = DEFAULT_CONFIG["shared"]
_DEFAULTS = metric_kwargs(DEFAULT_CONFIG, "sam_lp")


# =========================
# L2 Normalization (row-wise)
# =========================
def l2_normalize_rows(X: np.ndarray, eps: float = _SHARED["eps"]) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return X / norms


# =========================
# Gaussian Projection (cached)
# =========================
_GAUSS_CACHE = {}  # (seed, in_dim, out_dim) -> R


def gaussian_projection(X: np.ndarray, out_dim: int = _SHARED["reduce_dim"], seed: int = _SHARED["seed"]) -> np.ndarray:
    """
    Random Gaussian projection:
        R ~ N(0, 1/out_dim),  X' = X R

    Args:
        X: (N, d)
        out_dim: projected dimension
        seed: random seed

    Returns:
        (N, out_dim) if out_dim < d else (N, d)
    """
    N, in_dim = X.shape
    if out_dim >= in_dim:
        return X

    key = (int(seed), int(in_dim), int(out_dim))
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
# Mean pooling (slide init)
# =========================
def mean_pool_slide_embeddings(patch_embeddings: list[np.ndarray]) -> np.ndarray:
    """
    Args:
        patch_embeddings: list of arrays Z_i with shape (n_i, d)

    Returns:
        slide_embs: (N, d)
    """
    slides = []
    for Z in patch_embeddings:
        if Z.ndim != 2:
            raise ValueError(f"Each Z must be 2D, got {Z.shape}")
        slides.append(Z.mean(axis=0))
    return np.stack(slides, axis=0).astype(np.float64, copy=False)


# =========================
# SAM cluster centroids (per slide)
# =========================
def sam_cluster_centroids(
    Z: np.ndarray,
    sam_cluster: np.ndarray,
    min_cluster_size: int = _DEFAULTS["min_cluster_size"],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-slide cluster centroids.

    Args:
        Z: (n, d) patch embeddings
        sam_cluster: (n,) cluster ids
        min_cluster_size: clusters smaller than this are ignored

    Returns:
        C: (K, d) centroids
        sizes: (K,) cluster sizes
    """
    if Z.ndim != 2:
        raise ValueError(f"Z must be 2D, got {Z.shape}")
    sam_cluster = np.asarray(sam_cluster).astype(int)
    if sam_cluster.ndim != 1 or sam_cluster.shape[0] != Z.shape[0]:
        raise ValueError("sam_cluster must be shape (n_patches,) aligned with Z")

    cents = []
    sizes = []

    for cid in np.unique(sam_cluster):
        idx = np.where(sam_cluster == int(cid))[0]
        if idx.size < int(min_cluster_size):
            continue
        cents.append(Z[idx].mean(axis=0))
        sizes.append(idx.size)

    if len(cents) == 0:
        # fallback: single centroid = whole-slide mean
        return Z.mean(axis=0, keepdims=True).astype(np.float64), np.array([Z.shape[0]], dtype=np.float64)

    return np.stack(cents, axis=0).astype(np.float64), np.asarray(sizes, dtype=np.float64)


# =========================
# EM step: re-embed slide by centroid selection
# =========================
def reembed_slide_from_centroids(
    C: np.ndarray,
    sizes: np.ndarray,
    y: int,
    clf,
    select_mode: str = _DEFAULTS["select_mode"],   # "topk" or "soft"
    topk: int = _DEFAULTS["topk"],
    beta: float = _DEFAULTS["beta"],
    weight_mode: str = _DEFAULTS["weight_mode"],   # "none" | "sqrt" | "linear"
    eps: float = _SHARED["eps"],
) -> np.ndarray:
    """
    Given centroids C (K,d), choose/weight them using classifier confidence for the true label y,
    then return a single slide embedding (d,).

    NOTE: C must already be in the same feature space as clf input
          (i.e., after optional projection + L2 norm pipeline).
    """
    K = C.shape[0]
    if K == 1:
        return C[0].copy()

    # p(y | centroid)
    prob = clf.predict_proba(C)
    cls = clf.classes_
    pos = int(np.searchsorted(cls, int(y)))
    pos = np.clip(pos, 0, prob.shape[1] - 1)
    py = np.clip(prob[:, pos], 1e-12, 1.0)
    logp = np.log(py)  # (K,)

    # size weights
    if weight_mode == "none":
        w_size = np.ones_like(sizes, dtype=np.float64)
    elif weight_mode == "linear":
        w_size = sizes.astype(np.float64)
    elif weight_mode == "sqrt":
        w_size = np.sqrt(sizes.astype(np.float64))
    else:
        raise ValueError("weight_mode must be one of ['none','sqrt','linear']")

    if select_mode == "topk":
        k = max(1, int(topk))
        idx = np.argsort(-logp)[:k]
        w = w_size[idx]
        v = (w[:, None] * C[idx]).sum(axis=0)
    elif select_mode == "soft":
        b = float(beta)
        a = np.exp(b * (logp - np.max(logp)))
        a = a / (a.sum() + eps)
        w = a * w_size
        v = (w[:, None] * C).sum(axis=0)
    else:
        raise ValueError("select_mode must be one of ['topk','soft']")

    return v.astype(np.float64, copy=False)


# =========================
# Logistic regression utilities
# =========================
def fit_logreg(X: np.ndarray, y: np.ndarray, C: float, max_iter: int, seed: int):
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception as e:
        raise ImportError("scikit-learn is required for SAM-LP (linear_head).") from e

    clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver="lbfgs",
        random_state=seed,
    )
    clf.fit(X, y)
    return clf


def avg_loglik(clf, X: np.ndarray, y: np.ndarray) -> float:
    prob = clf.predict_proba(X)
    cls = clf.classes_
    idx = np.searchsorted(cls, y)
    p = prob[np.arange(len(y)), idx]
    p = np.clip(p, 1e-12, 1.0)
    return float(np.mean(np.log(p)))


# =========================
# Public API: SAM-LP score
# =========================
def sam_lp_score(
    patch_embeddings: list[np.ndarray],
    sam_clusters: list[np.ndarray],
    labels: np.ndarray,
    *,
    # logistic regression
    C_lr: float = _DEFAULTS["C_lr"],
    max_iter: int = _DEFAULTS["max_iter"],
    seed: int = _SHARED["seed"],
    # preprocessing
    reduce_dim: int | None = _SHARED["reduce_dim"],
    reduce_seed: int = _SHARED["seed"],
    eps: float = _SHARED["eps"],
    # SAM centroid + EM
    em_iters: int = _DEFAULTS["em_iters"],
    min_cluster_size: int = _DEFAULTS["min_cluster_size"],
    select_mode: str = _DEFAULTS["select_mode"],   # "topk" | "soft"
    topk: int = _DEFAULTS["topk"],
    beta: float = _DEFAULTS["beta"],
    weight_mode: str = _DEFAULTS["weight_mode"],   # "none" | "sqrt" | "linear"
) -> float:
    """
    Returns:
        SAM-LP score = average log-likelihood (train==eval)
    """
    if len(patch_embeddings) != len(sam_clusters):
        raise ValueError("patch_embeddings and sam_clusters must have the same length")
    y = np.asarray(labels).reshape(-1).astype(int)
    N = len(y)
    if N != len(patch_embeddings):
        raise ValueError("labels length must match number of slides")

    if len(np.unique(y)) < 2:
        return float("-inf")

    # ---- (0) init slide embeddings by mean pooling ----
    X_slide = mean_pool_slide_embeddings(patch_embeddings)

    # ---- preprocessing: optional projection, then L2 norm ONCE ----
    if reduce_dim is not None:
        X_slide = gaussian_projection(X_slide, out_dim=int(reduce_dim), seed=int(reduce_seed))
    X_slide = l2_normalize_rows(X_slide, eps=eps)

    # ---- (1) fit initial classifier ----
    clf = fit_logreg(X_slide, y, C=C_lr, max_iter=max_iter, seed=seed)

    # Region centroids and their preprocessing do not depend on the classifier.
    prepared = []
    if int(em_iters) > 0:
        for Z, c in zip(patch_embeddings, sam_clusters):
            Cc, Ss = sam_cluster_centroids(Z, c, min_cluster_size=min_cluster_size)
            if reduce_dim is not None:
                Cc = gaussian_projection(Cc, out_dim=int(reduce_dim), seed=int(reduce_seed))
            prepared.append((l2_normalize_rows(Cc, eps=eps), Ss))

    # ---- (2)-(3) EM-like loop ----
    for _ in range(int(em_iters)):
        new_embs = []
        for i, (Cc, Ss) in enumerate(prepared):
            vi = reembed_slide_from_centroids(
                Cc, Ss, int(y[i]), clf,
                select_mode=select_mode,
                topk=topk,
                beta=beta,
                weight_mode=weight_mode,
                eps=eps,
            )
            new_embs.append(vi)

        X_slide = np.stack(new_embs, axis=0)

        # IMPORTANT: keep the same rule — L2 norm ONCE (here, we normalize the final slide matrix)
        X_slide = l2_normalize_rows(X_slide, eps=eps)

        clf = fit_logreg(X_slide, y, C=C_lr, max_iter=max_iter, seed=seed)

    # ---- (4) final score ----
    return avg_loglik(clf, X_slide, y)
