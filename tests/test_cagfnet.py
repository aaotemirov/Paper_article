"""Unit tests (run with:  python -m unittest discover -s tests -v)."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cagfnet.calibration import apply_calibration, fit_temperature_shift  # noqa: E402
from cagfnet.config import load_config  # noqa: E402
from cagfnet.data import HybridDataset, build_hybrid_dataset  # noqa: E402
from cagfnet.data.audiometry import audio_features, simulate_audiogram, simulate_tympanogram  # noqa: E402
from cagfnet.data.phantom import CLASSES, encode_ct, make_phantom, window  # noqa: E402
from cagfnet.metrics import binary_metrics, delong_test, ece_score  # noqa: E402
from cagfnet.models import build_model  # noqa: E402
from cagfnet.training import mc_dropout_logits, normalized_entropy, train_model  # noqa: E402


def tiny_cfg(tmp):
    return load_config(None, {
        "output_dir": str(Path(tmp) / "out"),
        "data": {"source": "synthetic", "cache_dir": str(Path(tmp) / "cache"),
                 "n_train": 64, "n_val": 32, "n_test": 32, "n_external": 32},
        "train": {"epochs": 1, "seeds": [0], "batch_size": 32},
    })


class TestData(unittest.TestCase):
    def test_phantom_shapes_and_windows(self):
        rng = np.random.default_rng(0)
        for cls in CLASSES:
            hu = make_phantom(cls, 0.7, rng)
            self.assertEqual(hu.shape, (80, 80))
            x = encode_ct(hu)
            self.assertEqual(x.shape, (2, 80, 80))
            self.assertTrue(0.0 <= x.min() and x.max() <= 1.0)
        self.assertEqual(window(np.array([-160.0]), 40, 400)[0], 0.0)
        self.assertEqual(window(np.array([240.0]), 40, 400)[0], 1.0)

    def test_conductive_component_and_features(self):
        rng = np.random.default_rng(1)
        donor = np.array([10, 10, 15, 20, 25, 30, 35], np.float32)
        ac, bc = simulate_audiogram("chol", 1.0, donor, rng)
        abg = ac[:5] - bc
        self.assertGreater(abg.mean(), 25)                  # 16 + 28 s dB conductive loss
        self.assertGreater(abg[2:5].mean(), abg[:2].mean())  # larger at high frequencies
        ac0, bc0 = simulate_audiogram("normal", 0.0, donor, rng)
        self.assertLess(abs((ac0[:5] - bc0).mean()), 6)
        tymp = simulate_tympanogram("etd", 0.5, {"p0": 0, "ve": 1.2, "w": 100, "y": 0.8}, rng)
        curves, clin = audio_features(ac, bc, tymp)
        self.assertEqual(curves.shape, (3, 7))
        self.assertEqual(clin.shape, (11,))
        np.testing.assert_array_equal(curves[2], [1, 1, 1, 1, 1, 0, 0])

    def test_dataset_build_and_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = tiny_cfg(tmp)
            d1 = build_hybrid_dataset(cfg)
            d2 = build_hybrid_dataset(cfg)           # from cache
            np.testing.assert_allclose(d1["test"]["hu"], d2["test"]["hu"])
            self.assertEqual(len(d1["external"]["y"]), 32)
            internal = set(d1["train"]["donor"]) | set(d1["test"]["donor"])
            self.assertFalse(internal & set(d1["external"]["donor"]))   # participant-level split


class TestModels(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        self.batch = (torch.rand(5, 2, 80, 80), torch.randn(5, 3, 7), torch.randn(5, 11))

    def test_all_models_forward(self):
        for name in self.cfg["models"]:
            m = build_model(name, self.cfg).eval()
            out = m(*self.batch)
            self.assertEqual(out.shape, (5,), name)

    def test_missing_modality_and_attention(self):
        m = build_model("cagf_net", self.cfg).eval()
        for miss in (None, "ct", "audio"):
            self.assertTrue(torch.isfinite(m(*self.batch, missing=miss)).all())
        m(*self.batch)
        self.assertEqual(tuple(m.xattn.last_attn.shape), (5, 100, 4))
        self.assertEqual(tuple(m.fusion.last_gate.shape), (5, 64))

    def test_modality_dropout_changes_training_output(self):
        torch.manual_seed(0)
        m = build_model("cagf_net", self.cfg)
        m.p_md = 1.0
        m.train()
        a = m(*self.batch)
        self.assertEqual(a.shape, (5,))


class TestMetrics(unittest.TestCase):
    def test_delong_matches_sklearn_auc(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 300)
        p1 = y * 0.6 + rng.random(300)
        p2 = y * 0.2 + rng.random(300)
        a1, a2, _, pv = delong_test(y, p1, p2)
        self.assertAlmostEqual(a1, roc_auc_score(y, p1), places=6)
        self.assertAlmostEqual(a2, roc_auc_score(y, p2), places=6)
        self.assertLess(pv, 0.01)
        self.assertAlmostEqual(delong_test(y, p1, p1)[3], 1.0)

    def test_ece_and_binary_metrics(self):
        y = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(ece_score(y, np.array([0.0, 0.0, 1.0, 1.0])), 0.0)
        m = binary_metrics(y, np.array([0.1, 0.6, 0.7, 0.9]), 0.5)
        self.assertAlmostEqual(m["sensitivity"], 1.0)
        self.assertAlmostEqual(m["specificity"], 0.5)

    def test_calibration_recovers_temperature(self):
        rng = np.random.default_rng(0)
        true = rng.normal(0, 2, 4000)
        y = (rng.random(4000) < 1 / (1 + np.exp(-true))).astype(float)
        T, c = fit_temperature_shift(true * 3 + 1.5, y)      # over-confident, shifted logits
        self.assertAlmostEqual(T, 3.0, delta=0.4)
        self.assertAlmostEqual(c, -0.5, delta=0.25)
        p = apply_calibration(np.array([0.0]), 1.0, 0.0)
        self.assertAlmostEqual(float(p[0]), 0.5)

    def test_entropy(self):
        u = normalized_entropy(np.array([0.5, 1e-9, 1 - 1e-9]))
        self.assertAlmostEqual(u[0], 1.0, places=6)
        self.assertLess(u[1], 1e-6)


class TestTrainingSmoke(unittest.TestCase):
    def test_one_epoch_and_mc_dropout(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = tiny_cfg(tmp)
            data = build_hybrid_dataset(cfg)
            tr, va = HybridDataset(data["train"]), HybridDataset(data["val"])
            m = train_model(build_model("cagf_net", cfg), tr, va, cfg, seed=0)
            mc = mc_dropout_logits(m, va, n_samples=3)
            self.assertEqual(mc.shape, (3, len(va)))
            self.assertFalse(np.allclose(mc[0], mc[1]))   # dropout active at inference


if __name__ == "__main__":
    unittest.main()
