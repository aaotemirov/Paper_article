"""NHANES 2015-2016 audiometry loader (AUX_I + DEMO_I).

Each usable ear contributes air-conduction thresholds at 0.5, 1, 2, 3, 4, 6, 8 kHz and
tympanometry (middle-ear pressure p0, equivalent volume Ve, static admittance Y, width w).
Ears are kept when the participant is 20-69 years old, all seven thresholds are valid,
otoscopy is normal (when the otoscopy variable is available) and the tympanogram is
Type A (0.3 <= Y <= 1.75 ml, p0 > -100 daPa). Codes 666 ("no response") and 888
("not measured") are treated as missing, never as dB values.
"""
from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/"
FILES = {"aux": "AUX_I.xpt", "demo": "DEMO_I.xpt"}
FREQS_KHZ = [0.5, 1, 2, 3, 4, 6, 8]
_AC_STEMS = ["AUXU500", "AUXU1K1", "AUXU2K", "AUXU3K", "AUXU4K", "AUXU6K", "AUXU8K"]
_TYMP = {"p0": "AUXTMEP", "ve": "AUXTPV", "y": "AUXTCOM", "w": "AUXTWID"}
INVALID_CODES = (666, 888)


def download_nhanes(dest: str | Path, force: bool = False) -> dict[str, Path]:
    """Download the two public-domain NHANES transport files (about 5 MB in total)."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out = {}
    for key, name in FILES.items():
        path = dest / name
        if force or not path.exists():
            log.info("Downloading %s ...", BASE_URL + name)
            req = urllib.request.Request(BASE_URL + name, headers={"User-Agent": "cagfnet-research-code"})
            with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
                f.write(r.read())
        out[key] = path
    return out


def _clean(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.where(~s.isin(INVALID_CODES))


def _find_otoscopy(cols: list[str], side: str) -> str | None:
    cands = [c for c in cols if "OTS" in c.upper() and c.upper().endswith(side)]
    return cands[0] if cands else None


def load_nhanes_ears(nhanes_dir: str | Path, age_min: int = 20, age_max: int = 69,
                     download: bool = True) -> pd.DataFrame:
    """Return one row per eligible ear: SEQN, side, age, ac_0..ac_6, p0, ve, y, w."""
    nhanes_dir = Path(nhanes_dir)
    paths = {k: nhanes_dir / v for k, v in FILES.items()}
    if not all(p.exists() for p in paths.values()):
        if not download:
            raise FileNotFoundError(f"NHANES files not found in {nhanes_dir}")
        paths = download_nhanes(nhanes_dir)

    aux = pd.read_sas(paths["aux"], format="xport")
    demo = pd.read_sas(paths["demo"], format="xport")[["SEQN", "RIDAGEYR"]]
    df = aux.merge(demo, on="SEQN", how="inner")
    df = df[(df["RIDAGEYR"] >= age_min) & (df["RIDAGEYR"] <= age_max)]

    rows = []
    for side in ("R", "L"):
        ac_cols = [f"{s}{side}" for s in _AC_STEMS]
        missing = [c for c in ac_cols if c not in df.columns]
        if missing:
            raise KeyError(f"NHANES columns not found: {missing}")
        part = pd.DataFrame({"SEQN": df["SEQN"].astype(int), "side": side, "age": df["RIDAGEYR"]})
        for k, c in enumerate(ac_cols):
            part[f"ac_{k}"] = _clean(df[c]).values
        for k, stem in _TYMP.items():
            col = f"{stem}{side}"
            part[k] = _clean(df[col]).values if col in df.columns else np.nan
        oto = _find_otoscopy(list(df.columns), side)
        part["oto_ok"] = (pd.to_numeric(df[oto], errors="coerce") == 1).values if oto else True
        rows.append(part)
    ears = pd.concat(rows, ignore_index=True)

    ac = [f"ac_{k}" for k in range(7)]
    keep = ears[ac].notna().all(axis=1) & ears["oto_ok"]
    keep &= ears["y"].between(0.3, 1.75) & (ears["p0"] > -100)
    keep &= ears["ve"].notna() & ears["w"].notna()
    ears = ears[keep].drop(columns="oto_ok").reset_index(drop=True)
    log.info("NHANES: %d eligible ears from %d participants", len(ears), ears["SEQN"].nunique())
    return ears


def synthetic_ears(n_participants: int = 3800, seed: int = 0) -> pd.DataFrame:
    """Offline stand-in for NHANES (age-dependent presbycusis model).

    Only for smoke tests and environments without internet access; results produced
    with it are NOT the results reported in the paper.
    """
    rng = np.random.default_rng(seed)
    freqs = np.array(FREQS_KHZ)
    rows = []
    for pid in range(n_participants):
        age = rng.integers(20, 70)
        slope = max(0.0, (age - 25) / 45.0)
        base = rng.normal(5, 6)
        for side in ("R", "L"):
            hf = slope * (8 + 22 * np.log2(freqs / 0.5) / 4) * rng.uniform(0.6, 1.4)
            ac = np.clip(np.round((base + hf + rng.normal(0, 4, 7)) / 5) * 5, -10, 110)
            rows.append({
                "SEQN": 100000 + pid, "side": side, "age": float(age),
                **{f"ac_{k}": float(ac[k]) for k in range(7)},
                "p0": float(rng.normal(-15, 25)), "ve": float(rng.uniform(0.8, 2.0)),
                "y": float(rng.uniform(0.35, 1.6)), "w": float(rng.uniform(60, 140)),
            })
    ears = pd.DataFrame(rows)
    ears = ears[ears["y"].between(0.3, 1.75) & (ears["p0"] > -100)].reset_index(drop=True)
    return ears


def split_participants(ears: pd.DataFrame, external_fraction: float, seed: int):
    """Participant-level split into internal and external donor pools (no overlap)."""
    rng = np.random.default_rng(seed)
    pids = np.array(sorted(ears["SEQN"].unique()))
    rng.shuffle(pids)
    n_ext = int(round(len(pids) * external_fraction))
    ext = set(pids[:n_ext].tolist())
    is_ext = ears["SEQN"].isin(ext)
    return ears[~is_ext].reset_index(drop=True), ears[is_ext].reset_index(drop=True)
