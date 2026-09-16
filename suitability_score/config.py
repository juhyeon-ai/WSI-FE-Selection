"""One source of metric defaults for Python, the scoring CLI and preprocessing."""
from copy import deepcopy
import json
import math
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "suitability.json"
ALL_METRICS = ["nesum", "self_cluster", "sam_cluster", "linear_probing", "sam_lp", "effective_dimension", "logme"]


def _merge(base, override, prefix=""):
    if not isinstance(override, dict):
        raise ValueError(f"{prefix or 'config'} must be a JSON object")
    for key, value in override.items():
        name = f"{prefix}.{key}" if prefix else key
        if key not in base:
            raise ValueError(f"Unknown scoring setting: {name}")
        if isinstance(base[key], dict):
            _merge(base[key], value, name)
        else:
            base[key] = deepcopy(value)
    return base


def validate_config(config):
    def number(section, key, low, high=None, integer=False, low_open=False):
        value = config[section][key]
        valid = type(value) is int if integer else type(value) in (int, float)
        if not valid or not math.isfinite(value) or (value <= low if low_open else value < low) or (high is not None and value > high):
            raise ValueError(f"Invalid {section}.{key}: {value!r}")

    def choice(section, key, options):
        if config[section][key] not in options:
            raise ValueError(f"{section}.{key} must be one of {options}")

    metrics = config["metrics"]
    if not isinstance(metrics, list) or not metrics or any(not isinstance(m, str) or m not in ALL_METRICS for m in metrics) or len(set(metrics)) != len(metrics):
        raise ValueError("metrics must be a nonempty list of distinct supported metric names")
    number("shared", "reduce_dim", 2, integer=True)
    number("shared", "seed", 0, 2**32 - 1, integer=True)
    number("shared", "max_patches", 50, integer=True)
    number("shared", "eps", 0, low_open=True)
    number("linear_probe", "C", 0, low_open=True)
    number("linear_probe", "max_iter", 1, integer=True)
    choice("self_cluster", "reduce_method", ["none", "gaussian", "pca"])
    choice("self_cluster", "aggregate", ["mean", "median"])
    choice("sam_cluster", "reduce_method", ["none", "gaussian"])
    number("sam_cluster", "min_cluster_ratio", 0, 1)
    number("sam_cluster", "n_pairs", 40, integer=True)
    offsets = config["sam_cluster"]["rng_seed_offsets"]
    if not isinstance(offsets, list) or not offsets or any(type(x) is not int or x < 0 for x in offsets):
        raise ValueError("sam_cluster.rng_seed_offsets must be a nonempty list of nonnegative integers")
    choice("sam_cluster", "cluster_sampling", ["uniform", "sqrt", "proportional"])
    choice("sam_cluster", "dtype", ["float32", "float64"])
    number("sam_cluster", "ratio_topk_exclude", 0, integer=True)
    number("sam_cluster", "ratio_sim_threshold", -1, 1)
    number("sam_cluster", "oversample_factor", 0, low_open=True)
    if config["sam_cluster"]["max_seconds"] is not None:
        number("sam_cluster", "max_seconds", 0, low_open=True)
    number("sam_lp", "em_iters", 0, integer=True)
    number("sam_lp", "min_cluster_size", 1, integer=True)
    number("sam_lp", "topk", 1, integer=True)
    number("sam_lp", "beta", 0)
    choice("sam_lp", "select_mode", ["soft", "topk"])
    choice("sam_lp", "weight_mode", ["linear", "sqrt", "none"])
    number("effective_dimension", "explained_var", 0, 1, low_open=True)
    for section, key in [("nesum", "center"), ("effective_dimension", "center"), ("logme", "warmup_numba")]:
        if type(config[section][key]) is not bool:
            raise ValueError(f"{section}.{key} must be true or false")
    return config


def load_config(path=None, *, overrides=None):
    """Merge an optional partial config over the canonical defaults; reject typos."""
    config = json.loads(DEFAULT_CONFIG_PATH.read_text())
    if path is not None:
        if isinstance(path, dict):
            _merge(config, path)
        else:
            _merge(config, json.loads(Path(path).read_text()))
    if overrides is not None:
        _merge(config, overrides)
    return validate_config(config)


def metric_kwargs(config, metric):
    """Translate validated JSON settings into existing public function arguments."""
    shared = config["shared"]
    projection = dict(reduce_dim=shared["reduce_dim"], reduce_seed=shared["seed"])
    if metric == "effective_dimension":
        return dict(config[metric], eps=shared["eps"])
    if metric == "logme":
        return dict(config[metric], **projection, eps_norm=shared["eps"])
    if metric == "linear_probing":
        return dict(config["linear_probe"], **projection, seed=shared["seed"], eps=shared["eps"])
    settings = dict(config[metric], **projection, eps=shared["eps"])
    if metric == "sam_lp":
        settings.update(C_lr=config["linear_probe"]["C"], max_iter=config["linear_probe"]["max_iter"], seed=shared["seed"])
    elif metric == "sam_cluster":
        offsets = settings.pop("rng_seed_offsets")
        settings["rng_seeds"] = tuple(shared["seed"] + offset for offset in offsets)
        settings["max_patches"] = shared["max_patches"]
        if settings["max_seconds"] is None:
            settings["max_seconds"] = float("inf")
    return settings


# Loaded once per Python process. Restart after editing the canonical JSON if
# using direct function defaults in an existing notebook or interpreter.
DEFAULT_CONFIG = load_config()
