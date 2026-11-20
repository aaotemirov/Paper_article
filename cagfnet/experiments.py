"""End-to-end experiment runner.

    python -m cagfnet.experiments --config configs/default.yaml

Produces in <output_dir>:
  table1_internal.(csv|md)   detection results on the internal test set (mean +/- std over seeds)
  table2_external.(csv|md)   results under the artificial domain shift
  ablation.(csv|md)          CAGF-Net component ablations
  noise_robustness.csv       AUC versus additional CT noise
  missing_modality.csv       CAGF-Net with one modality replaced by surrogate tokens
  severity_sensitivity.csv   sensitivity per severity group
  uncertainty.json           MC-dropout entropy: error-detection AUC, selective accuracy
  delong.csv                 DeLong tests CAGF-Net vs. each model (per seed)
  summary.json, predictions_seed*.npz, checkpoints/, figures/
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from .calibration import apply_calibration, calibrated_logit, fit_temperature_shift
from .config import load_config
from .data import HybridDataset, build_hybrid_dataset
from .metrics import binary_metrics, bootstrap_auc_ci, delong_test, youden_threshold
from .models import MODEL_NAMES, build_model
from .training import mc_dropout_logits, normalized_entropy, predict_logits, set_seed, train_model

log = logging.getLogger("cagfnet")
FUSION_WITH_SURROGATE = ("cagf_net", "cagf_no_xattn", "cagf_no_gate")
MAIN_MODELS = ["ct_only", "audio_only", "late_fusion", "early_fusion", "cagf_net"]
ABLATIONS = ["cagf_net", "cagf_no_xattn", "cagf_no_gate", "cagf_no_moddrop", "early_fusion"]


def _logit(p):
    return float(np.log(p / (1 - p)))


def run_seed(cfg, data, dsets, seed, out_dir):
    ev = cfg["eval"]
    rec = {}             # rec[model][key] -> probabilities / logits
    calib = {}
    for name in cfg["models"]:
        t0 = time.time()
        log.info("[seed %d] training %s", seed, name)
        set_seed(seed)                      # weight initialisation must be seeded too
        model = build_model(name, cfg)
        model = train_model(model, dsets["train"], dsets["val"], cfg, seed)
        (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_dir / "checkpoints" / f"{name}_seed{seed}.pt")

        z_val = predict_logits(model, dsets["val"])
        T, c = fit_temperature_shift(z_val, dsets["val"].y.numpy())
        calib[name] = (T, c)
        r = {"val_logit": calibrated_logit(z_val, T, c)}
        for split in ("test", "external"):
            r[f"{split}_logit"] = calibrated_logit(predict_logits(model, dsets[split]), T, c)
        for lvl, ds in dsets["noise"].items():
            r[f"noise{lvl}_logit"] = calibrated_logit(predict_logits(model, ds), T, c)
        if name in FUSION_WITH_SURROGATE:
            for miss in ("audio", "ct"):
                r[f"miss_{miss}_logit"] = calibrated_logit(predict_logits(model, dsets["test"], missing=miss), T, c)
        if name in ("cagf_net", "ct_only"):
            mc = mc_dropout_logits(model, dsets["test"], ev["mc_samples"], seed)
            r["mc_prob"] = apply_calibration(mc, T, c).mean(0)
        rec[name] = r
        log.info("  done in %.1fs (T=%.3f, c=%.3f)", time.time() - t0, T, c)

    # late fusion (Bayesian additive evidence): lambda = lambda_t + lambda_a - lambda_0
    if "ct_only" in rec and "audio_only" in rec:
        lam0 = _logit(float(dsets["train"].y.numpy().mean()))
        lf = {}
        for key in rec["ct_only"]:
            if key.endswith("_logit") and key in rec["audio_only"]:
                lf[key] = rec["ct_only"][key] + rec["audio_only"][key] - lam0
        for lvl in dsets["noise"]:   # noisy CT + clean audiometry
            lf[f"noise{lvl}_logit"] = rec["ct_only"][f"noise{lvl}_logit"] + rec["audio_only"]["test_logit"] - lam0
        rec["late_fusion"] = lf

    # convert logits -> probabilities, thresholds from validation (Youden)
    y = {s: dsets[s].y.numpy() for s in ("val", "test", "external")}
    res = {"metrics": {}, "noise": {}, "missing": {}, "severity": {}, "uncertainty": {}, "delong": []}
    probs = {}
    for name, r in rec.items():
        p = {k.replace("_logit", ""): 1 / (1 + np.exp(-v)) for k, v in r.items() if k.endswith("_logit")}
        if "mc_prob" in r:
            p["mc"] = r["mc_prob"]
        probs[name] = p
        thr = youden_threshold(y["val"], p["val"])
        m = {s: binary_metrics(y[s], p[s], thr, ev["ece_bins"]) for s in ("test", "external")}
        lo, hi = bootstrap_auc_ci(y["test"], p["test"], ev["bootstrap"], seed=seed)
        m["test"]["auc_ci95"] = [lo, hi]
        res["metrics"][name] = m
        res["noise"][name] = {lvl: float(roc_auc_score(y["test"], p[f"noise{lvl}"])) for lvl in dsets["noise"]}
        for miss in ("audio", "ct"):
            if f"miss_{miss}" in p:
                res["missing"].setdefault(name, {})[miss] = binary_metrics(y["test"], p[f"miss_{miss}"], thr, ev["ece_bins"])
        # sensitivity per severity group (cholesteatoma cases of the internal test set)
        sev = data["test"]["severity"]
        bins = ev["severity_bins"]
        sens = {}
        for a, b in zip(bins[:-1], bins[1:]):
            mask = (y["test"] == 1) & (sev >= a) & (sev < b)
            if mask.any():
                sens[f"{a:.2f}-{min(b, 1):.2f}"] = {"n": int(mask.sum()), "sensitivity": float((p["test"][mask] >= thr).mean())}
        res["severity"][name] = sens
        if "mc" in p:
            pm = p["mc"]
            u = normalized_entropy(pm)
            pred = (pm >= thr).astype(int)
            err = (pred != y["test"]).astype(int)
            keep = u <= np.quantile(u, ev["coverage"])
            res["uncertainty"][name] = {
                "error_detection_auc": float(roc_auc_score(err, u)) if 0 < err.sum() < len(err) else float("nan"),
                "accuracy_all": float((pred == y["test"]).mean()),
                f"accuracy_at_{int(ev['coverage'] * 100)}pct_coverage": float((pred[keep] == y["test"][keep]).mean()),
                "u_correct_mean": float(u[err == 0].mean()), "u_error_mean": float(u[err == 1].mean()) if err.any() else None,
            }
            res["uncertainty"][name]["_u"] = u.tolist()
            res["uncertainty"][name]["_err"] = err.tolist()
    if "cagf_net" in probs:
        for other in probs:
            if other == "cagf_net":
                continue
            for split in ("test", "external"):
                a1, a2, zst, pv = delong_test(y[split], probs["cagf_net"][split], probs[other][split])
                res["delong"].append({"seed": seed, "split": split, "model": other, "auc_cagf": a1,
                                      "auc_other": a2, "z": zst, "p_value": pv})
    np.savez_compressed(out_dir / f"predictions_seed{seed}.npz",
                        **{f"{m}__{k}": v for m, pk in probs.items() for k, v in pk.items()},
                        **{f"y__{s}": y[s] for s in y})
    res["calibration"] = {k: {"T": v[0], "c": v[1]} for k, v in calib.items()}
    return res, probs


def _mean_std(vals):
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), float("nan"))


def _table(per_seed, split, models, cols=("auc", "ap", "sensitivity", "specificity", "f1", "ece", "brier")):
    rows = []
    for m in models:
        if m not in per_seed[0]["metrics"]:
            continue
        row = {"Model": MODEL_NAMES.get(m, m)}
        for c in cols:
            mu, sd = _mean_std([r["metrics"][m][split][c] for r in per_seed])
            row[c.upper() if c in ("auc", "ap", "ece") else c.capitalize()] = f"{mu:.3f} ± {sd:.3f}" if c == "auc" else f"{mu:.3f}"
        rows.append(row)
    return pd.DataFrame(rows)


def _write(df: pd.DataFrame, path: Path):
    df.to_csv(path.with_suffix(".csv"), index=False)
    try:
        md = df.to_markdown(index=False)
    except ImportError:          # 'tabulate' not installed
        md = df.to_string(index=False)
    path.with_suffix(".md").write_text(md + "\n", encoding="utf-8")


def aggregate(cfg, per_seed, out_dir: Path):
    models_main = [m for m in MAIN_MODELS if m in per_seed[0]["metrics"]]
    _write(_table(per_seed, "test", models_main), out_dir / "table1_internal")
    ext = _table(per_seed, "external", models_main, ("auc", "sensitivity", "ece"))
    drops = []
    for m in models_main:
        drops.append(_mean_std([r["metrics"][m]["test"]["auc"] - r["metrics"][m]["external"]["auc"] for r in per_seed])[0])
    ext.insert(2, "AUC drop", [f"{d:.3f}" for d in drops])
    _write(ext, out_dir / "table2_external")
    _write(_table(per_seed, "test", [m for m in ABLATIONS if m in per_seed[0]["metrics"]]), out_dir / "ablation")

    levels = cfg["eval"]["noise_levels"]
    rows = []
    for m in per_seed[0]["noise"]:
        for lvl in levels:
            mu, sd = _mean_std([r["noise"][m][lvl] for r in per_seed])
            rows.append({"model": m, "noise_hu": lvl, "auc_mean": mu, "auc_std": sd})
    pd.DataFrame(rows).to_csv(out_dir / "noise_robustness.csv", index=False)

    rows = []
    for m in per_seed[0]["missing"]:
        for miss in ("audio", "ct"):
            mu, sd = _mean_std([r["missing"][m][miss]["auc"] for r in per_seed])
            se, _ = _mean_std([r["missing"][m][miss]["sensitivity"] for r in per_seed])
            rows.append({"model": m, "missing": miss, "auc_mean": mu, "auc_std": sd, "sensitivity_mean": se})
    pd.DataFrame(rows).to_csv(out_dir / "missing_modality.csv", index=False)

    rows = []
    for m in per_seed[0]["severity"]:
        for g in per_seed[0]["severity"][m]:
            mu, sd = _mean_std([r["severity"][m].get(g, {}).get("sensitivity") for r in per_seed])
            rows.append({"model": m, "severity": g, "n": per_seed[0]["severity"][m][g]["n"], "sens_mean": mu, "sens_std": sd})
    pd.DataFrame(rows).to_csv(out_dir / "severity_sensitivity.csv", index=False)

    unc = {}
    for m in per_seed[0]["uncertainty"]:
        keys = [k for k in per_seed[0]["uncertainty"][m] if not k.startswith("_")]
        unc[m] = {k: _mean_std([r["uncertainty"][m][k] for r in per_seed]) for k in keys}
    (out_dir / "uncertainty.json").write_text(json.dumps(unc, indent=2), encoding="utf-8")
    pd.DataFrame([d for r in per_seed for d in r["delong"]]).to_csv(out_dir / "delong.csv", index=False)

    clean = [{k: v for k, v in r.items() if k != "uncertainty"} for r in per_seed]
    (out_dir / "summary.json").write_text(json.dumps({"config": cfg, "per_seed": clean}, indent=2, default=str),
                                          encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description="CAGF-Net hybrid-simulation experiments")
    ap.add_argument("--config", default=None, help="YAML config (defaults = paper protocol)")
    ap.add_argument("--output", default=None, help="output directory")
    ap.add_argument("--source", choices=["nhanes", "synthetic"], default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    over = {}
    if a.output: over["output_dir"] = a.output
    if a.source: over.setdefault("data", {})["source"] = a.source
    if a.seeds: over.setdefault("train", {})["seeds"] = a.seeds
    if a.epochs: over.setdefault("train", {})["epochs"] = a.epochs
    if a.models: over["models"] = a.models
    cfg = load_config(a.config, over)
    torch.set_num_threads(cfg["train"]["num_threads"])
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    data = build_hybrid_dataset(cfg)
    dsets = {s: HybridDataset(data[s]) for s in ("train", "val", "test", "external")}
    dsets["noise"] = {lvl: HybridDataset(data["test"], extra_ct_noise=lvl, seed=123) for lvl in cfg["eval"]["noise_levels"]}
    log.info("Dataset: " + ", ".join(f"{s}={len(dsets[s])} (chol {int(dsets[s].y.sum())})"
                                     for s in ("train", "val", "test", "external")))

    per_seed = []
    for seed in cfg["train"]["seeds"]:
        res, _ = run_seed(cfg, data, dsets, seed, out_dir)
        per_seed.append(res)
    aggregate(cfg, per_seed, out_dir)
    if not a.no_figures:
        from .plots import make_all_figures
        make_all_figures(cfg, data, per_seed, out_dir)
    print((out_dir / "table1_internal.md").read_text(encoding="utf-8"))
    log.info("Results written to %s", out_dir.resolve())


if __name__ == "__main__":
    main()
