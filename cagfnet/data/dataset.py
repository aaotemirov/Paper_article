"""Hybrid simulation dataset: real NHANES audiometric background + synthetic pathology + CT phantoms."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import data_hash
from .audiometry import audio_features, qc_flag, simulate_audiogram, simulate_tympanogram
from .nhanes import load_nhanes_ears, split_participants, synthetic_ears
from .phantom import CLASSES, encode_ct, make_phantom

log = logging.getLogger(__name__)
SPLITS = ("train", "val", "test", "external")


def _generate_split(n, pool, cfg, rng, external):
    d = cfg["data"]
    names = list(d["class_probs"])
    probs = np.array([d["class_probs"][k] for k in names], dtype=float)
    probs /= probs.sum()
    lo, hi = d["severity_range"]
    size = d["roi_size"]
    out = {k: [] for k in ("hu", "curves", "clinical", "y", "cls", "severity", "donor", "qc")}
    ac_cols = [f"ac_{k}" for k in range(7)]
    for _ in range(n):
        cls = names[rng.choice(len(names), p=probs)]
        sev = 0.0 if cls == "normal" else float(rng.uniform(lo, hi))
        donor = pool.iloc[rng.integers(len(pool))]
        ac, bc = simulate_audiogram(cls, sev, donor[ac_cols].to_numpy(np.float32), rng, external)
        tymp = simulate_tympanogram(cls, sev, donor[["p0", "ve", "w", "y"]].to_dict(), rng)
        curves, clinical = audio_features(ac, bc, tymp)
        out["hu"].append(make_phantom(cls, sev, rng, size=size, external=external))
        out["curves"].append(curves)
        out["clinical"].append(clinical)
        out["y"].append(int(cls == "chol"))
        out["cls"].append(CLASSES.index(cls))
        out["severity"].append(sev)
        out["donor"].append(int(donor["SEQN"]))
        out["qc"].append(qc_flag(ac, bc))
    return {k: np.asarray(v) for k, v in out.items()}


def build_hybrid_dataset(cfg: dict, use_cache: bool = True) -> dict[str, dict[str, np.ndarray]]:
    """Generate (or load from cache) train/val/test/external splits."""
    d = cfg["data"]
    cache = Path(d["cache_dir"]) / f"hybrid_{d['source']}_{data_hash(cfg)}.npz"
    if use_cache and cache.exists():
        z = np.load(cache)
        log.info("Loaded cached dataset %s", cache)
        return {s: {k.split("__", 1)[1]: z[k] for k in z.files if k.startswith(s + "__")} for s in SPLITS}

    if d["source"] == "nhanes":
        ears = load_nhanes_ears(d["nhanes_dir"], d["age_min"], d["age_max"])
    elif d["source"] == "synthetic":
        log.warning("Using SYNTHETIC audiometric background (smoke test only, not paper results)")
        ears = synthetic_ears(seed=d["data_seed"])
    else:
        raise ValueError(f"unknown data source {d['source']}")
    internal, external = split_participants(ears, d["external_participant_fraction"], d["data_seed"])
    log.info("Donor pools: internal %d ears, external %d ears", len(internal), len(external))

    rng = np.random.default_rng(d["data_seed"])
    data = {
        "train": _generate_split(d["n_train"], internal, cfg, rng, external=False),
        "val": _generate_split(d["n_val"], internal, cfg, rng, external=False),
        "test": _generate_split(d["n_test"], internal, cfg, rng, external=False),
        "external": _generate_split(d["n_external"], external, cfg, rng, external=True),
    }
    data["meta"] = {"n_ears_internal": np.array(len(internal)), "n_ears_external": np.array(len(external)),
                    "n_participants": np.array(ears["SEQN"].nunique())}
    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        flat = {f"{s}__{k}": v for s in SPLITS for k, v in data[s].items()}
        flat.update({f"meta__{k}": v for k, v in data["meta"].items()})
        np.savez_compressed(cache, **flat)
        log.info("Saved dataset cache %s", cache)
    return data


class HybridDataset(Dataset):
    """Torch dataset; optional extra CT noise (HU) is added before windowing."""

    def __init__(self, split: dict[str, np.ndarray], extra_ct_noise: float = 0.0, seed: int = 0):
        hu = split["hu"]
        if extra_ct_noise > 0:
            rng = np.random.default_rng(seed)
            hu = hu + rng.normal(0, extra_ct_noise, hu.shape).astype(np.float32)
        self.ct = torch.from_numpy(encode_ct(hu))
        self.curves = torch.from_numpy(split["curves"].astype(np.float32))
        self.clinical = torch.from_numpy(split["clinical"].astype(np.float32))
        self.y = torch.from_numpy(split["y"].astype(np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.ct[i], self.curves[i], self.clinical[i], self.y[i]
