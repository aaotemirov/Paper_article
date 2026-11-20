"""Training loop, inference and Monte-Carlo dropout."""
from __future__ import annotations

import copy
import logging
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(model: nn.Module, train_ds, val_ds, cfg: dict, seed: int) -> nn.Module:
    """Weighted BCE (omega = N0/N1), AdamW, cosine schedule; keeps the best-validation-AUC epoch."""
    t = cfg["train"]
    set_seed(seed)
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True, generator=g, drop_last=False)
    y = train_ds.y.numpy()
    omega = float((y == 0).sum() / max((y == 1).sum(), 1))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(omega))
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["epochs"] * len(loader))
    beta = t.get("l2_beta", 0.0)

    best_auc, best_state = -1.0, None
    for epoch in range(t["epochs"]):
        model.train()
        total = 0.0
        for ct, curves, clin, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(ct, curves, clin), yb)
            if beta > 0:
                loss = loss + beta * sum((p ** 2).sum() for p in model.parameters())
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(yb)
        val_logits = predict_logits(model, val_ds)
        auc = roc_auc_score(val_ds.y.numpy(), val_logits)
        log.debug("epoch %d loss %.4f val_auc %.4f", epoch + 1, total / len(train_ds), auc)
        if auc > best_auc:
            best_auc, best_state = auc, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    log.info("  best val AUC %.4f", best_auc)
    return model


@torch.no_grad()
def predict_logits(model: nn.Module, ds, missing: str | None = None, batch_size: int = 256) -> np.ndarray:
    model.eval()
    out = []
    for ct, curves, clin, _ in DataLoader(ds, batch_size=batch_size):
        out.append(model(ct, curves, clin, missing=missing))
    return torch.cat(out).numpy()


def _enable_dropout_only(model: nn.Module):
    """Eval mode everywhere (BatchNorm uses running statistics) except Dropout layers."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def mc_dropout_logits(model: nn.Module, ds, n_samples: int = 30, seed: int = 0, batch_size: int = 256) -> np.ndarray:
    """Return (R, N) stochastic logits with dropout active at inference."""
    torch.manual_seed(seed)
    _enable_dropout_only(model)
    runs = []
    for _ in range(n_samples):
        out = []
        for ct, curves, clin, _ in DataLoader(ds, batch_size=batch_size):
            out.append(model(ct, curves, clin))
        runs.append(torch.cat(out).numpy())
    model.eval()
    return np.stack(runs)


def normalized_entropy(p: np.ndarray) -> np.ndarray:
    """u = H(p) / ln 2 in [0, 1]."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p)) / np.log(2)
