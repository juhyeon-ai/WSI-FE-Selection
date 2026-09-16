# SAM_cluster.py
# SAM-Cluster (SAM ratio score)
#
# Goal:
#   Compute an unsupervised structure-consistency score using SAM clusters.
#   (Labels are NOT used, but accepted to keep a unified interface.)
#
# Definition (score in [0,1]):
#   - Preprocess patch embeddings (optional Gaussian projection, then L2 normalize)
#   - Use SAM cluster ids as pseudo-structure groups
#   - Estimate within-cluster and between-cluster similarity^2 statistics by sampling patch pairs
#   - ratio = E[within sim^2] / E[between sim^2]
#   - score = 1 - 1/max(ratio, 1)
#
# Unified API:
#   sam_cluster_score(patch_embeddings, sam_clusters, labels, ...) -> float
#
# Requirements: numpy

from __future__ import annotations
import time
import numpy as np

from .config import DEFAULT_CONFIG, metric_kwargs

_SHARED = DEFAULT_CONFIG["shared"]
_DEFAULTS = metric_kwargs(DEFAULT_CONFIG, "sam_cluster")


# =========================
# Preprocessing
# =========================
def l2_normalize_rows(X: np.ndarray, eps: float = _SHARED["eps"]) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (n + eps)


def gaussian_random_projection(
    X: np.ndarray,
    out_dim: int = _SHARED["reduce_dim"],
    seed: int = _SHARED["seed"],
    eps: float = _SHARED["eps"],
    dtype: np.dtype = _DEFAULTS["dtype"],
) -> np.ndarray:
    """
    Random Gaussian projection:
        R ~ N(0, 1),  X' = (X R) / sqrt(out_dim)
    """
    n, d = X.shape
    if out_dim <= 0:
        raise ValueError(f"out_dim must be > 0, got {out_dim}")
    rng = np.random.default_rng(seed)
    R = rng.normal(0.0, 1.0, size=(d, out_dim)).astype(dtype, copy=False)
    X = X.astype(dtype, copy=False)
    return (X @ R) / (np.sqrt(out_dim) + eps)


def preprocess_patches(
    Z: np.ndarray,
    eps: float = _SHARED["eps"],
    reduce_method: str = _DEFAULTS["reduce_method"],  # "none" | "gaussian"
    reduce_dim: int = _SHARED["reduce_dim"],
    reduce_seed: int = _SHARED["seed"],
    dtype: np.dtype = _DEFAULTS["dtype"],
) -> np.ndarray:
    """
    Returns:
        W: (n, d') L2-normalized patch embeddings
    """
    if Z.ndim != 2:
        raise ValueError(f"Z must be 2D, got shape={Z.shape}")

    rm = reduce_method.lower().strip()
    if rm == "gaussian":
        W = gaussian_random_projection(Z, out_dim=reduce_dim, seed=reduce_seed, eps=eps, dtype=dtype)
    elif rm == "none":
        W = Z.astype(dtype, copy=False)
    else:
        raise ValueError("reduce_method must be one of: ['none','gaussian']")

    return l2_normalize_rows(W, eps=eps)


# =========================
# Cluster utilities
# =========================
def remap_cluster_ids(cluster_ids: np.ndarray) -> np.ndarray:
    """Remap arbitrary ids to contiguous {0,...,K-1}."""
    cluster_ids = np.asarray(cluster_ids).astype(np.int64).reshape(-1)
    _, inv = np.unique(cluster_ids, return_inverse=True)
    return inv.astype(np.int32, copy=False)


def build_cluster_slices(cid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sorting index and per-cluster slices.

    Returns:
        order: indices that sort cid
        starts: (K,) start offset for each cluster in cid_sorted
        counts: (K,) count for each cluster
    """
    cid = np.asarray(cid).astype(np.int32).reshape(-1)
    order = np.argsort(cid, kind="mergesort")
    cid_sorted = cid[order]
    K = int(cid_sorted[-1] + 1) if cid_sorted.size > 0 else 0
    counts = np.bincount(cid_sorted, minlength=K).astype(np.int32, copy=False)

    starts = np.empty(K, dtype=np.int64)
    if K > 0:
        starts[0] = 0
        if K > 1:
            starts[1:] = np.cumsum(counts[:-1], dtype=np.int64)
    return order, starts, counts


def downsample_patches_uniform(W: np.ndarray, cid: np.ndarray, max_patches: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    n = W.shape[0]
    if max_patches <= 0 or n <= max_patches:
        return W, cid
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_patches, replace=False)
    return W[idx], cid[idx]


# =========================
# Core ratio score (fast sampling)
# =========================
def _calc_ratio_score_fast(
    W: np.ndarray,
    cid: np.ndarray,
    *,
    n_pairs: int = _DEFAULTS["n_pairs"],
    rng_seed: int = _DEFAULTS["rng_seeds"][0],
    eps: float = _SHARED["eps"],
    cluster_sampling: str = _DEFAULTS["cluster_sampling"],     # "uniform"|"sqrt"|"proportional"
    topk_exclude: int = _DEFAULTS["ratio_topk_exclude"],
    sim_threshold: float = _DEFAULTS["ratio_sim_threshold"],
    oversample_factor: float = _DEFAULTS["oversample_factor"],
    max_seconds: float = _DEFAULTS["max_seconds"],
) -> float:
    n, d = W.shape
    if n < 2:
        return 0.0

    order, starts, counts = build_cluster_slices(cid)
    K = counts.shape[0]
    if K < 2:
        return 0.0

    sizes = counts.astype(np.float64)

    if cluster_sampling == "uniform":
        p = np.ones(K, dtype=np.float64)
    elif cluster_sampling == "sqrt":
        p = np.sqrt(np.maximum(sizes, 1.0))
    elif cluster_sampling == "proportional":
        p = np.maximum(sizes, 1.0)
    else:
        raise ValueError("cluster_sampling must be one of: ['uniform','sqrt','proportional']")
    p = p / (p.sum() + eps)

    rng = np.random.default_rng(rng_seed)

    # ---- centroid exclusion ----
    C_sum = np.zeros((K, d), dtype=np.float64)
    np.add.at(C_sum, cid.astype(np.int64, copy=False), W.astype(np.float64, copy=False))
    denom = np.maximum(counts.astype(np.float64), 1.0)[:, None]
    C = l2_normalize_rows(C_sum / denom, eps=eps)

    Sim_C = C @ C.T
    excluded = np.eye(K, dtype=bool)

    actual_k = min(int(topk_exclude), K - 2)
    if actual_k > 0:
        tmp = Sim_C.copy()
        np.fill_diagonal(tmp, -np.inf)
        top_idx = np.argpartition(tmp, -actual_k, axis=1)[:, -actual_k:]
        rows = np.arange(K)[:, None]
        excluded[rows, top_idx] = True

    if sim_threshold < 1.0:
        excluded |= (Sim_C > sim_threshold)

    if not np.any(~excluded):
        excluded = np.eye(K, dtype=bool)

    n_w = n_pairs // 2
    n_b = n_pairs - n_w
    if n_w < 20 or n_b < 20:
        return 0.0

    t0 = time.time()

    # ---- within ----
    a = rng.choice(K, size=n_w, p=p)
    ok = counts[a] >= 2
    if ok.sum() < 20:
        return 0.0
    a = a[ok]

    r1 = rng.integers(0, counts[a], size=a.size, endpoint=False)
    r2 = rng.integers(0, counts[a], size=a.size, endpoint=False)

    collide = (r1 == r2)
    guard = 0
    while np.any(collide) and guard < 5:
        r2[collide] = rng.integers(0, counts[a[collide]], size=int(collide.sum()), endpoint=False)
        collide = (r1 == r2)
        guard += 1

    pos_i = starts[a] + r1.astype(np.int64)
    pos_j = starts[a] + r2.astype(np.int64)
    i_idx = order[pos_i]
    j_idx = order[pos_j]

    dots_w = np.einsum("ij,ij->i", W[i_idx], W[j_idx])
    m_within = float(np.mean(dots_w * dots_w))

    # ---- between ----
    need = n_b
    dots_b_chunks = []

    while need > 0 and (time.time() - t0) < max_seconds:
        cand = int(max(1024, min(need * oversample_factor, 250000)))
        aa = rng.choice(K, size=cand, p=p)
        bb = rng.choice(K, size=cand, p=p)

        valid = ~excluded[aa, bb]
        if not np.any(valid):
            excluded = np.eye(K, dtype=bool)
            continue

        aa = aa[valid]
        bb = bb[valid]
        take = min(int(aa.size), int(need))
        aa = aa[:take]
        bb = bb[:take]

        ra = rng.integers(0, counts[aa], size=take, endpoint=False)
        rb = rng.integers(0, counts[bb], size=take, endpoint=False)

        pos_i = starts[aa] + ra.astype(np.int64)
        pos_j = starts[bb] + rb.astype(np.int64)

        i = order[pos_i]
        j = order[pos_j]

        dots_b = np.einsum("ij,ij->i", W[i], W[j])
        dots_b_chunks.append(dots_b * dots_b)
        need -= take

    if len(dots_b_chunks) == 0:
        return 0.0

    dots_b_all = np.concatenate(dots_b_chunks, axis=0)
    if dots_b_all.size < 20:
        return 0.0

    m_between = float(np.mean(dots_b_all))

    ratio = m_within / (m_between + eps)
    r = max(ratio, 1.0)
    score = 1.0 - 1.0 / r
    return float(np.clip(score, 0.0, 1.0))


# =========================
# Per-slide score (Z, cluster) -> float
# =========================
def sam_ratio_score(
    Z: np.ndarray,
    sam_cluster: np.ndarray,
    *,
    eps: float = _SHARED["eps"],
    reduce_method: str = _DEFAULTS["reduce_method"],   # "none" | "gaussian"
    reduce_dim: int = _SHARED["reduce_dim"],
    reduce_seed: int = _SHARED["seed"],
    # cluster filtering / patch downsampling
    min_cluster_ratio: float = _DEFAULTS["min_cluster_ratio"],
    max_patches: int = _DEFAULTS["max_patches"],
    # sampling
    n_pairs: int = _DEFAULTS["n_pairs"],
    rng_seeds: tuple[int, ...] = _DEFAULTS["rng_seeds"],
    cluster_sampling: str = _DEFAULTS["cluster_sampling"],  # "uniform"|"sqrt"|"proportional"
    ratio_topk_exclude: int = _DEFAULTS["ratio_topk_exclude"],
    ratio_sim_threshold: float = _DEFAULTS["ratio_sim_threshold"],
    oversample_factor: float = _DEFAULTS["oversample_factor"],
    max_seconds: float = _DEFAULTS["max_seconds"],
    dtype: np.dtype = _DEFAULTS["dtype"],
) -> float:
    if Z.ndim != 2:
        raise ValueError(f"Z must be 2D, got {Z.shape}")

    W = preprocess_patches(
        Z,
        eps=eps,
        reduce_method=reduce_method,
        reduce_dim=reduce_dim,
        reduce_seed=reduce_seed,
        dtype=dtype,
    )

    sam_cluster = np.asarray(sam_cluster).reshape(-1)
    if sam_cluster.shape[0] != W.shape[0]:
        raise ValueError(f"Mismatch: Z has {W.shape[0]} patches but sam_cluster has {sam_cluster.shape[0]}")

    cid = remap_cluster_ids(sam_cluster)

    # ---- remove tiny clusters ----
    if min_cluster_ratio > 0:
        total = cid.size
        min_size = max(2, int(total * float(min_cluster_ratio)))
        counts = np.bincount(cid)
        valid = counts >= min_size
        mask = valid[cid]
        if int(mask.sum()) < 50:
            return 0.0
        W = W[mask]
        cid = cid[mask]
        cid = remap_cluster_ids(cid)

    # ---- patch downsample ----
    W, cid = downsample_patches_uniform(W, cid, max_patches=int(max_patches), seed=int(reduce_seed))

    if W.shape[0] < 50 or int(np.max(cid, initial=-1)) < 1:
        return 0.0

    vals = []
    for s in rng_seeds:
        vals.append(
            _calc_ratio_score_fast(
                W, cid,
                n_pairs=int(n_pairs),
                rng_seed=int(s),
                eps=eps,
                cluster_sampling=cluster_sampling,
                topk_exclude=int(ratio_topk_exclude),
                sim_threshold=float(ratio_sim_threshold),
                oversample_factor=float(oversample_factor),
                max_seconds=float(max_seconds),
            )
        )
    return float(np.mean(vals)) if len(vals) > 0 else 0.0


# =========================
# Unified API (LP / SAM_LP style)
# =========================
def sam_cluster_score(
    patch_embeddings: list[np.ndarray],
    sam_clusters: list[np.ndarray],
    labels: np.ndarray,   # accepted for API alignment; NOT used
    *,
    eps: float = _SHARED["eps"],
    reduce_method: str = _DEFAULTS["reduce_method"],
    reduce_dim: int = _SHARED["reduce_dim"],
    reduce_seed: int = _SHARED["seed"],
    min_cluster_ratio: float = _DEFAULTS["min_cluster_ratio"],
    max_patches: int = _DEFAULTS["max_patches"],
    n_pairs: int = _DEFAULTS["n_pairs"],
    rng_seeds: tuple[int, ...] = _DEFAULTS["rng_seeds"],
    cluster_sampling: str = _DEFAULTS["cluster_sampling"],
    ratio_topk_exclude: int = _DEFAULTS["ratio_topk_exclude"],
    ratio_sim_threshold: float = _DEFAULTS["ratio_sim_threshold"],
    oversample_factor: float = _DEFAULTS["oversample_factor"],
    max_seconds: float = _DEFAULTS["max_seconds"],
    dtype: np.dtype = _DEFAULTS["dtype"],
) -> float:
    """
    Unified interface:
        - patch_embeddings: list of (n_i, d)
        - sam_clusters:     list of (n_i,)
        - labels:           (N,) but unused

    Returns:
        mean SAM ratio score across slides
    """
    if len(patch_embeddings) != len(sam_clusters):
        raise ValueError("patch_embeddings and sam_clusters must have the same length")

    y = np.asarray(labels).reshape(-1)
    if y.shape[0] != len(patch_embeddings):
        raise ValueError("labels length must match number of slides (even though labels are not used)")

    scores = []
    for Z, c in zip(patch_embeddings, sam_clusters):
        scores.append(
            sam_ratio_score(
                Z, c,
                eps=eps,
                reduce_method=reduce_method,
                reduce_dim=reduce_dim,
                reduce_seed=reduce_seed,
                min_cluster_ratio=min_cluster_ratio,
                max_patches=max_patches,
                n_pairs=n_pairs,
                rng_seeds=rng_seeds,
                cluster_sampling=cluster_sampling,
                ratio_topk_exclude=ratio_topk_exclude,
                ratio_sim_threshold=ratio_sim_threshold,
                oversample_factor=oversample_factor,
                max_seconds=max_seconds,
                dtype=dtype,
            )
        )
    return float(np.mean(scores)) if len(scores) > 0 else 0.0