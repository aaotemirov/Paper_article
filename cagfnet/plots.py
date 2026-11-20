"""Figures: dataset samples, ROC/PR, noise robustness, domain shift, severity, uncertainty."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from .data.audiometry import FREQS_KHZ  # noqa: E402
from .data.phantom import CLASSES, window  # noqa: E402
from .models import MODEL_NAMES  # noqa: E402

COLORS = {"ct_only": "#4C72B0", "audio_only": "#DD8452", "late_fusion": "#8C8C8C",
          "early_fusion": "#55A868", "cagf_net": "#C44E52", "cagf_no_xattn": "#8172B3",
          "cagf_no_gate": "#937860", "cagf_no_moddrop": "#DA8BC3"}
LABELS = {"normal": "Normal", "etd": "ETD", "com": "COM", "chol": "Cholesteatoma"}


def fig_dataset(data, path):
    tr = data["train"]
    fig, ax = plt.subplots(2, 4, figsize=(12, 5.5))
    for k, cls in enumerate(CLASSES):
        idx = np.where(tr["cls"] == k)[0]
        if len(idx) == 0:
            continue
        sev = tr["severity"][idx]
        i = idx[np.argmax(sev)] if cls != "normal" else idx[0]
        ax[0, k].imshow(window(tr["hu"][i], 40, 400), cmap="gray")
        ax[0, k].set_title(f"{LABELS[cls]}" + (f" (s={tr['severity'][i]:.2f})" if cls != "normal" else ""))
        ax[0, k].axis("off")
        ac = tr["curves"][idx, 0] * 25 + 20
        bc = tr["curves"][idx, 1, :5] * 25 + 20
        for arr, col, lab in ((ac, "#C44E52", "AC"), (bc, "#4C72B0", "BC")):
            f = FREQS_KHZ[: arr.shape[1]]
            med, q1, q3 = np.median(arr, 0), np.percentile(arr, 25, 0), np.percentile(arr, 75, 0)
            ax[1, k].plot(f, med, "o-", color=col, label=lab)
            ax[1, k].fill_between(f, q1, q3, color=col, alpha=0.2)
        ax[1, k].set_xscale("log"); ax[1, k].set_xticks(FREQS_KHZ); ax[1, k].set_xticklabels([str(x).rstrip("0").rstrip(".") for x in FREQS_KHZ])
        ax[1, k].invert_yaxis(); ax[1, k].set_ylim(80, -10); ax[1, k].set_xlabel("Frequency (kHz)")
        if k == 0:
            ax[1, k].set_ylabel("Threshold (dB HL)"); ax[1, k].legend()
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_roc_pr(npz_path, models, path):
    z = np.load(npz_path)
    y = z["y__test"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.3))
    for m in models:
        if f"{m}__test" not in z.files:
            continue
        p = z[f"{m}__test"]
        fpr, tpr, _ = roc_curve(y, p)
        pr, rc, _ = precision_recall_curve(y, p)
        ax[0].plot(fpr, tpr, color=COLORS.get(m), label=MODEL_NAMES.get(m, m))
        ax[1].plot(rc, pr, color=COLORS.get(m), label=MODEL_NAMES.get(m, m))
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0].set(xlabel="1 - Specificity", ylabel="Sensitivity", title="(a) ROC, internal test")
    ax[1].set(xlabel="Recall", ylabel="Precision", title="(b) Precision-recall, internal test")
    ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_noise_shift(out_dir, per_seed, path):
    df = pd.read_csv(out_dir / "noise_robustness.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for m in ["ct_only", "early_fusion", "late_fusion", "cagf_net"]:
        d = df[df.model == m]
        if len(d):
            ax[0].errorbar(d.noise_hu, d.auc_mean, yerr=d.auc_std, marker="o", capsize=3,
                           color=COLORS.get(m), label=MODEL_NAMES.get(m, m))
    ax[0].set(xlabel="Gaussian noise added to CT, sigma (HU)", ylabel="AUC", title="(a) CT quality degradation")
    ax[0].legend(fontsize=8)
    models = [m for m in ["ct_only", "audio_only", "early_fusion", "late_fusion", "cagf_net"] if m in per_seed[0]["metrics"]]
    x = np.arange(len(models))
    for j, split in enumerate(("test", "external")):
        vals = [np.mean([r["metrics"][m][split]["auc"] for r in per_seed]) for m in models]
        ax[1].bar(x + (j - 0.5) * 0.38, vals, 0.38, label="Internal test" if split == "test" else "External domain",
                  color=["#9DB4D3", "#1F4E8C"][j])
    ax[1].set_xticks(x); ax[1].set_xticklabels([MODEL_NAMES[m].split(" (")[0] for m in models], rotation=15)
    ax[1].set(ylabel="AUC", title="(b) Other scanner and audiometer"); ax[1].set_ylim(0.5, 1.0); ax[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_severity(out_dir, path):
    df = pd.read_csv(out_dir / "severity_sensitivity.csv")
    models = [m for m in ["ct_only", "audio_only", "late_fusion", "early_fusion", "cagf_net"] if m in set(df.model)]
    groups = list(dict.fromkeys(df.severity))
    x = np.arange(len(groups)); w = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    for j, m in enumerate(models):
        d = df[df.model == m].set_index("severity").reindex(groups)
        ax.bar(x + j * w - 0.4 + w / 2, d.sens_mean, w, yerr=d.sens_std, capsize=2, color=COLORS.get(m), label=MODEL_NAMES[m])
    ax.set_xticks(x); ax.set_xticklabels([f"s {g}" for g in groups])
    ax.set(ylabel="Sensitivity", title="Detection sensitivity by cholesteatoma severity"); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_uncertainty(per_seed, coverage, path):
    r = per_seed[0]["uncertainty"]
    if "cagf_net" not in r:
        return
    u, err = np.array(r["cagf_net"]["_u"]), np.array(r["cagf_net"]["_err"])
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].hist(u[err == 0], bins=20, alpha=0.7, label="correct", color="#55A868", density=True)
    if err.any():
        ax[0].hist(u[err == 1], bins=20, alpha=0.7, label="error", color="#C44E52", density=True)
    ax[0].set(xlabel="Normalised entropy u", ylabel="Density", title="(a) MC-dropout entropy"); ax[0].legend()
    for m in ("ct_only", "cagf_net"):
        if m not in r:
            continue
        um, em = np.array(r[m]["_u"]), np.array(r[m]["_err"])
        order = np.argsort(um)
        covs = np.linspace(0.3, 1.0, 29)
        acc = [1 - em[order[: max(1, int(c * len(um)))]].mean() for c in covs]
        ax[1].plot(covs, acc, marker=".", color=COLORS[m], label=MODEL_NAMES[m])
    ax[1].axvline(coverage, color="k", ls="--", lw=0.8)
    ax[1].set(xlabel="Coverage (automatic decisions)", ylabel="Accuracy", title="(b) Selective prediction"); ax[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def make_all_figures(cfg, data, per_seed, out_dir: Path):
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_dataset(data, fig_dir / "fig_dataset_samples.png")
    seed0 = cfg["train"]["seeds"][0]
    fig_roc_pr(out_dir / f"predictions_seed{seed0}.npz",
               ["ct_only", "audio_only", "late_fusion", "early_fusion", "cagf_net"], fig_dir / "fig_roc_pr.png")
    fig_noise_shift(out_dir, per_seed, fig_dir / "fig_noise_domain_shift.png")
    fig_severity(out_dir, fig_dir / "fig_severity.png")
    fig_uncertainty(per_seed, cfg["eval"]["coverage"], fig_dir / "fig_uncertainty.png")
