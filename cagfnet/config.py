"""Experiment configuration (defaults reproduce the paper's protocol)."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "output_dir": "results/default",
    "data": {
        # "nhanes": real NHANES 2015-2016 audiometry (downloaded automatically)
        # "synthetic": built-in pseudo-NHANES audiograms (offline smoke tests only)
        "source": "nhanes",
        "nhanes_dir": "data/nhanes",
        "cache_dir": "data/cache",
        "age_min": 20,
        "age_max": 69,
        "external_participant_fraction": 0.25,
        "class_probs": {"normal": 0.30, "etd": 0.25, "com": 0.25, "chol": 0.20},
        "severity_range": [0.1, 1.0],
        "n_train": 1200,
        "n_val": 400,
        "n_test": 400,
        "n_external": 600,
        "data_seed": 2027,
        "roi_size": 80,
    },
    "model": {
        "d_model": 64,
        "n_heads": 4,
        "ffn_hidden": 128,
        "ct_channels": [16, 32, 64, 64],
        "audio_channels": 32,
        "audio_tokens": 4,
        "dropout": 0.3,
        "modality_dropout": 0.2,
    },
    "train": {
        "epochs": 25,
        "batch_size": 64,
        "lr": 1.0e-3,
        "weight_decay": 1.0e-3,
        "l2_beta": 0.0,
        "seeds": [0, 1, 2],
        "num_threads": 8,
    },
    "eval": {
        "mc_samples": 30,
        "coverage": 0.8,
        "ece_bins": 15,
        "bootstrap": 1000,
        "noise_levels": [0, 100, 200, 300, 400, 500],
        "severity_bins": [0.1, 0.33, 0.55, 0.78, 1.0001],
    },
    "models": [
        "ct_only", "audio_only", "early_fusion", "cagf_net",
        "cagf_no_xattn", "cagf_no_gate", "cagf_no_moddrop",
    ],
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path, "r", encoding="utf-8") as f:
            cfg = _merge(cfg, yaml.safe_load(f) or {})
    if overrides:
        cfg = _merge(cfg, overrides)
    return cfg


def data_hash(cfg: dict) -> str:
    """Stable hash of the data section (used as dataset cache key)."""
    d = {k: v for k, v in cfg["data"].items() if k not in ("cache_dir", "nhanes_dir")}
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]
