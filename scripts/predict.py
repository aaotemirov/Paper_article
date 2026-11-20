"""Run a trained CAGF-Net checkpoint on one ear.

    python scripts/predict.py --checkpoint results/paper/checkpoints/cagf_net_seed0.pt \
        --ct roi_hu.npy --ac 15 20 25 35 45 40 45 --bc 5 5 10 10 15 \
        --tymp -180 1.2 160 0.4 [--T 1.0 --c 0.0]

--ct      80x80 NumPy array in Hounsfield units (use cagfnet.data.phantom.dicom_to_hu for DICOM)
--ac      air-conduction thresholds (dB HL) at 0.5, 1, 2, 3, 4, 6, 8 kHz
--bc      bone-conduction thresholds (dB HL) at 0.5, 1, 2, 3, 4 kHz
--tymp    p0 (daPa), Ve (ml), width (daPa), Y (ml)
--T, --c  calibration parameters (see summary.json -> per_seed -> calibration)
Omit --ct or --ac/--bc/--tymp to run with a missing modality (learned surrogate tokens).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cagfnet.config import load_config  # noqa: E402
from cagfnet.data.audiometry import audio_features  # noqa: E402
from cagfnet.data.phantom import encode_ct  # noqa: E402
from cagfnet.models import build_model  # noqa: E402
from cagfnet.training import mc_dropout_logits, normalized_entropy  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--ct")
    ap.add_argument("--ac", type=float, nargs=7)
    ap.add_argument("--bc", type=float, nargs=5)
    ap.add_argument("--tymp", type=float, nargs=4)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--c", type=float, default=0.0)
    ap.add_argument("--threshold-u", type=float, default=0.5, help="refer to a physician if u > tau")
    a = ap.parse_args()

    cfg = load_config(a.config)
    model = build_model("cagf_net", cfg)
    model.load_state_dict(torch.load(a.checkpoint, map_location="cpu"))
    missing = None
    ct = torch.zeros(1, 2, cfg["data"]["roi_size"], cfg["data"]["roi_size"])
    if a.ct:
        ct = torch.from_numpy(encode_ct(np.load(a.ct).astype(np.float32)))[None]
    else:
        missing = "ct"
    if a.ac is not None and a.bc is not None and a.tymp is not None:
        curves, clin = audio_features(np.array(a.ac, np.float32), np.array(a.bc, np.float32), a.tymp)
    else:
        if missing == "ct":
            sys.exit("provide at least one modality (--ct or --ac/--bc/--tymp)")
        curves, clin = np.zeros((3, 7), np.float32), np.zeros(11, np.float32)
        missing = "audio"
    curves, clin = torch.from_numpy(curves)[None], torch.from_numpy(clin)[None]

    model.eval()
    with torch.no_grad():
        z = model(ct, curves, clin, missing=missing).item()
    p = 1 / (1 + np.exp(-(z / a.T + a.c)))
    if missing is None:
        ds = [(ct[0], curves[0], clin[0], torch.tensor(0.0))]
        mc = mc_dropout_logits(model, ds, 30)
        p_mc = (1 / (1 + np.exp(-(mc / a.T + a.c)))).mean()
        u = float(normalized_entropy(np.array([p_mc]))[0])
        print(f"P(cholesteatoma) = {p:.3f}   MC mean = {p_mc:.3f}   uncertainty u = {u:.3f}")
        print("decision:", "automatic" if u <= a.threshold_u else "refer to physician")
    else:
        print(f"P(cholesteatoma) = {p:.3f}   (missing modality: {missing})")
