"""Exercise split isolation, metric accounting, and the public training/export path."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import torch
from checkpoint import load_checkpoint, open_checkpoint_data
from data import SignalData, prepare_features
from metrics import Metrics
from models import count_params

ROOT = Path(__file__).resolve().parent


def fixture_table():
    rng = np.random.default_rng(7)
    return pa.table(
        {
            "wav_stem": [f"24010{day}_000000" for day in (1, 1, 2, 2, 3, 3)],
            "spectrogram32": rng.normal(size=(6, 2 * 32 * 79))
            .astype(np.float32)
            .tolist(),
            "codeword": rng.integers(0, 2, size=(6, 174)).tolist(),
            "snr_db": [-25, -12, -5, 0, -20, 12],
            "llrs": rng.normal(size=(6, 4 * 174)).astype(np.float32).tolist(),
        }
    )


class PipelineTests(unittest.TestCase):
    def test_phase_features_ignore_global_phase(self):
        torch.manual_seed(4)
        raw = torch.randn(3, 2, 32, 79)
        angle = torch.tensor(0.7)
        rotated = torch.stack(
            (
                raw[:, 0] * angle.cos() - raw[:, 1] * angle.sin(),
                raw[:, 0] * angle.sin() + raw[:, 1] * angle.cos(),
            ),
            dim=1,
        )
        features = prepare_features(raw, "magnitude-phase")
        changed = prepare_features(rotated, "magnitude-phase")
        torch.testing.assert_close(features, changed, atol=2e-6, rtol=2e-6)
        self.assertTrue((features[:, 1, :, 0] == 0).all())
        torch.testing.assert_close(prepare_features(raw, "raw-complex"), raw)

    def test_metrics_weight_batches_and_keep_other(self):
        logits = torch.tensor([-2.0, 0.0, 3.0]).unsqueeze(1).expand(3, 174)
        labels = torch.tensor([0.0, 1.0, 1.0]).unsqueeze(1).expand(3, 174)
        snr = torch.tensor([-20.0, 10.0, -5.0])
        metric = Metrics()
        metric.update(logits[:2], labels[:2], snr[:2])
        metric.update(logits[2:], labels[2:], snr[2:])
        result = metric.result()
        self.assertAlmostEqual(result["overall"]["bit_accuracy"], 2 / 3)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels
        ).item()
        self.assertAlmostEqual(result["overall"]["bce"], expected, places=6)
        self.assertEqual(result["other"]["examples"], 1)
        self.assertEqual(
            sum(v["examples"] for k, v in result.items() if k != "overall"), 3
        )
        with self.assertRaises(ValueError):
            metric.update(logits * float("nan"), labels, snr)

    def test_split_and_pinned_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "signals.lance"
            table = fixture_table()
            lance.write_dataset(table, str(path))
            data = SignalData(path)
            split = data.make_split("240103")
            membership = [
                set(data.indices(split, name))
                for name in ("train", "validation", "test")
            ]
            self.assertEqual(set.union(*membership), set(range(6)))
            self.assertTrue(
                all(
                    not membership[i] & membership[j]
                    for i in range(3)
                    for j in range(i)
                )
            )
            self.assertEqual(membership[2], {4, 5})
            with self.assertRaises(ValueError):
                data.indices(dict(split, validation=split["train"]), "test")
            checkpoint = {"feature_source": "spectrogram32", "dataset": data.identity()}
            lance.write_dataset(table, str(path), mode="append")
            pinned = open_checkpoint_data(path, checkpoint)
            self.assertEqual(len(pinned.stems), 6)
            bad = table.set_column(
                table.schema.get_field_index("codeword"),
                "codeword",
                pa.array([[2] * 174] * 6),
            )
            bad_path = Path(directory) / "bad.lance"
            lance.write_dataset(bad, str(bad_path))
            with self.assertRaisesRegex(ValueError, "binary"):
                next(SignalData(bad_path).batches([0], 1, "raw-complex"))

    def test_train_evaluate_export_all_models(self):
        counts = {"mlp": 8515, "conv1d": 115139, "conv2d": 66083, "transformer": 159363}
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            dataset = base / "signals.lance"
            lance.write_dataset(fixture_table(), str(dataset))
            env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")

            def run(script, *args):
                result = subprocess.run(
                    [sys.executable, str(ROOT / script), *map(str, args)],
                    env=env,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            for name, count in counts.items():
                with self.subTest(model=name):
                    output = base / name
                    representation = (
                        "raw-complex" if name == "conv1d" else "magnitude-phase"
                    )
                    run(
                        "train.py",
                        "--data",
                        dataset,
                        "--output",
                        output,
                        "--model",
                        name,
                        "--representation",
                        representation,
                        "--test-date",
                        "240103",
                        "--epochs",
                        1,
                        "--batch-size",
                        1,
                    )
                    checkpoint = output / "best.pt"
                    model, metadata = load_checkpoint(checkpoint)
                    self.assertEqual(count_params(model), count)
                    self.assertEqual(metadata["representation"], representation)
                    self.assertTrue(np.isfinite(metadata["validation"]["bce"]))
                    report = output / "evaluation.json"
                    run(
                        "evaluate.py",
                        "--data",
                        dataset,
                        "--checkpoint",
                        checkpoint,
                        "--output",
                        report,
                        "--baselines",
                        "--batch-size",
                        1,
                    )
                    results = json.loads(report.read_text())["results"]
                    self.assertEqual(len(results), 5)
                    self.assertTrue(
                        all(
                            value["overall"]["examples"] == 2
                            for value in results.values()
                        )
                    )
                    run(
                        "export_onnx.py",
                        "--checkpoint",
                        checkpoint,
                        "--output",
                        output / "model.onnx",
                        "--batch-size",
                        2,
                    )
                    contract = json.loads((output / "model.json").read_text())
                    self.assertEqual(contract["input_shape"], [2, 2, 32, 79])
                    self.assertEqual(contract["representation"], representation)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
