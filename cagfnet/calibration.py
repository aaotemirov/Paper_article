"""Temperature scaling with an additional shift: p = sigma(z / T + c).

(T*, c*) = argmin_{T>0, c}  -sum_{i in V} [ y_i ln p_i + (1 - y_i) ln(1 - p_i) ]
The shift c compensates the positive logit bias introduced by the class weight omega.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def fit_temperature_shift(logits: np.ndarray, y: np.ndarray, max_iter: int = 200):
    z = torch.as_tensor(logits, dtype=torch.float64)
    t = torch.as_tensor(y, dtype=torch.float64)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    c = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t, c], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(z / log_t.exp() + c, t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp()), float(c.detach())


def apply_calibration(logits: np.ndarray, T: float, c: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(np.asarray(logits) / T + c)))


def calibrated_logit(logits: np.ndarray, T: float, c: float) -> np.ndarray:
    return np.asarray(logits) / T + c
