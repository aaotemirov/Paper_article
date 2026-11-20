"""Audiometric signal simulation and feature extraction.

The real NHANES air-conduction thresholds of a donor ear with a healthy middle ear define the
cochlear (sensorineural) state, i.e. the bone-conduction curve. A class-specific conductive
component is added on top: 15 + 20 s dB (ETD), 15 + 24 s dB (COM), 16 + 28 s dB (cholesteatoma,
larger at high frequencies). Tympanograms are drawn from class-specific mixtures (types C, B, Ad
and large equivalent volume); retraction and perforation occur in both COM and cholesteatoma,
so audiometry alone cannot separate these two classes.
"""
from __future__ import annotations

import numpy as np

FREQS_KHZ = np.array([0.5, 1, 2, 3, 4, 6, 8], dtype=np.float32)
N_BC = 5  # bone conduction is defined for 0.5-4 kHz only
LOW_IDX, HIGH_IDX = [0, 1, 2], [2, 3, 4]  # A_p: 0.5-2 kHz, A_y: 2-4 kHz

# relative frequency shape of the conductive component (mean ~ 1)
_SHAPE = {
    "etd": np.array([1.15, 1.10, 0.95, 0.90, 0.85, 0.80, 0.80]),
    "com": np.array([1.05, 1.00, 0.95, 0.95, 0.95, 0.90, 0.90]),
    "chol": np.array([0.80, 0.85, 1.00, 1.15, 1.25, 1.25, 1.25]),
}
_MAGNITUDE = {"etd": (15, 20), "com": (15, 24), "chol": (16, 28)}

# tympanogram pattern mixtures per class
_TYMP_MIX = {
    "etd": {"C": 0.6, "B": 0.4},
    "com": {"perforation": 0.4, "B": 0.3, "C": 0.3},
    "chol": {"C": 0.35, "perforation": 0.25, "B": 0.2, "Ad": 0.2},
}


def _round5(x):
    return np.clip(np.round(x / 5.0) * 5.0, -10, 120)


def simulate_audiogram(cls: str, severity: float, donor_ac: np.ndarray, rng: np.random.Generator,
                       external: bool = False):
    """Return (ac[7], bc[5]) thresholds in dB HL."""
    bc_full = donor_ac.astype(np.float32).copy()
    if cls == "normal":
        conductive = np.zeros(7, np.float32)
    else:
        a, b = _MAGNITUDE[cls]
        conductive = (a + b * severity) * _SHAPE[cls] + rng.normal(0, 3, 7)
        conductive = np.maximum(conductive, 0)
    ac = bc_full + conductive + rng.normal(0, 2.5, 7)
    bc = bc_full[:N_BC] + rng.normal(0, 2.5, N_BC)
    if external:                       # different audiometer: systematic +3 dB shift
        ac, bc = ac + 3.0, bc + 3.0
    return _round5(ac).astype(np.float32), _round5(bc).astype(np.float32)


def simulate_tympanogram(cls: str, severity: float, donor_tymp: dict, rng: np.random.Generator):
    """Return (p0 [daPa], Ve [ml], w [daPa], Y [ml])."""
    p0, ve, w, y = donor_tymp["p0"], donor_tymp["ve"], donor_tymp["w"], donor_tymp["y"]
    if cls != "normal":
        mix = _TYMP_MIX[cls]
        kind = rng.choice(list(mix), p=np.array(list(mix.values())))
        s = severity
        if kind == "C":            # retraction / negative pressure
            p0 = -rng.uniform(120, 200) - 150 * s
            y = y * rng.uniform(0.5, 0.9)
            w = w * rng.uniform(1.1, 1.6)
        elif kind == "B":          # effusion / mass: flat
            y = rng.uniform(0.05, 0.25)
            w = rng.uniform(250, 400)
            p0 = rng.uniform(-300, 0)
        elif kind == "perforation":  # large equivalent volume, flat
            ve = rng.uniform(2.2, 5.0)
            y = rng.uniform(0.05, 0.2)
            w = rng.uniform(250, 400)
        elif kind == "Ad":         # ossicular discontinuity: hypermobile
            y = rng.uniform(1.8, 3.0)
            w = w * rng.uniform(0.6, 0.9)
    return float(p0), float(ve), float(w), float(max(y, 0.05))


def audio_features(ac: np.ndarray, bc: np.ndarray, tymp):
    """Build the audio encoder inputs.

    curves  (3, 7): scaled AC, scaled BC (zero-padded), existence mask m in {0,1}^7
    clinical (11,): ABG at 5 frequencies, A_p, A_y (all /25), tympanometry tau (4)
    """
    p0, ve, w, y = tymp
    scale = lambda t: (t - 20.0) / 25.0  # noqa: E731
    mask = np.zeros(7, np.float32)
    mask[:N_BC] = 1.0
    bc_pad = np.zeros(7, np.float32)
    bc_pad[:N_BC] = scale(bc)
    curves = np.stack([scale(ac), bc_pad, mask]).astype(np.float32)

    abg = ac[:N_BC] - bc                                   # ABG(f_k) = T_AC(f_k) - T_BC(f_k)
    a_p, a_y = abg[LOW_IDX].mean(), abg[HIGH_IDX].mean()
    tau = np.array([p0 / 150.0, (ve - 1.5) / 1.2, (w - 150.0) / 120.0, np.log(y)], np.float32)
    clinical = np.concatenate([abg / 25.0, [a_p / 25.0, a_y / 25.0], tau]).astype(np.float32)
    return curves, clinical


def qc_flag(ac: np.ndarray, bc: np.ndarray) -> bool:
    """Quality-control check T_BC(f) <= T_AC(f) + 5 dB (violations are flagged, not corrected)."""
    return bool(np.all(bc <= ac[:N_BC] + 5))
