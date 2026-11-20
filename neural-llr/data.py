"""Read prepared Lance examples and preserve recording-level split membership."""

import hashlib
import json
import random
import re

import lance
import numpy as np
import torch

REPRESENTATIONS = ("raw-complex", "magnitude-phase")


def tensor_column(table, column, shape):
    values = np.asarray(table[column].to_pylist(), dtype=np.float32)
    if values.shape != (len(table), int(np.prod(shape))):
        raise ValueError(f"Expected flat {column} rows of length {int(np.prod(shape))}")
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite values in {column}")
    return torch.from_numpy(values.reshape(len(table), *shape))


def prepare_features(raw, representation):
    """Convert raw complex channels to the selected model input representation."""
    if representation == "raw-complex":
        return raw
    if representation != "magnitude-phase":
        raise ValueError(f"Unknown representation: {representation}")
    real, imag = raw[:, 0], raw[:, 1]
    phase_change = torch.zeros_like(real)
    phase_change[..., 1:] = torch.atan2(
        imag[..., 1:] * real[..., :-1] - real[..., 1:] * imag[..., :-1],
        real[..., 1:] * real[..., :-1] + imag[..., 1:] * imag[..., :-1],
    )
    return torch.stack((torch.hypot(real, imag), phase_change), dim=1)


class SignalData:
    """Read signal tensors on demand without retaining the full corpus in memory."""

    def __init__(self, path, feature_source="spectrogram32", version=None):
        if feature_source not in ("spectrogram", "spectrogram32"):
            raise ValueError("Feature source must be spectrogram or spectrogram32")
        self.dataset = lance.dataset(str(path), version=version)
        self.feature_source = feature_source
        self.freq_bins = 32 if feature_source == "spectrogram32" else 8
        required = {feature_source, "codeword", "snr_db", "wav_stem"}
        missing = required - set(self.dataset.schema.names)
        if missing:
            raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
        self.stems = self.dataset.to_table(columns=["wav_stem"])["wav_stem"].to_pylist()
        if not self.stems or any(
            not isinstance(stem, str) or not re.match(r"^\d{6}_", stem)
            for stem in self.stems
        ):
            raise ValueError(
                "Expected nonempty data with wav_stem values starting YYMMDD_"
            )

    def identity(self):
        digest = hashlib.sha256(json.dumps(self.stems).encode()).hexdigest()
        return {
            "version": self.dataset.version,
            "rows": len(self.stems),
            "stems_sha256": digest,
        }

    def make_split(self, test_date, val_fraction=0.1, seed=42):
        if not 0 < val_fraction < 1:
            raise ValueError("Validation fraction must be between zero and one")
        dates = sorted({stem[:6] for stem in self.stems})
        if test_date not in dates:
            raise ValueError(f"Test date {test_date} is absent from the dataset")
        remaining = [date for date in dates if date != test_date]
        if len(remaining) < 2:
            raise ValueError(
                "At least three date groups are needed for train/validation/test"
            )
        random.Random(seed).shuffle(remaining)
        n_val = min(len(remaining) - 1, max(1, round(len(remaining) * val_fraction)))
        return {
            "train": sorted(remaining[n_val:]),
            "validation": sorted(remaining[:n_val]),
            "test": [test_date],
        }

    def indices(self, split, name):
        groups = [set(split[key]) for key in ("train", "validation", "test")]
        if any(not group for group in groups) or any(
            groups[i] & groups[j] for i in range(3) for j in range(i)
        ):
            raise ValueError("Split date groups must be nonempty and disjoint")
        if set.union(*groups) != {stem[:6] for stem in self.stems}:
            raise ValueError("Split does not cover exactly this dataset's date groups")
        return np.array(
            [i for i, stem in enumerate(self.stems) if stem[:6] in split[name]]
        )

    def batches(self, indices, batch_size, representation, seed=None, baselines=False):
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        order = np.array(indices, dtype=np.int64, copy=True)
        if seed is not None:
            np.random.default_rng(seed).shuffle(order)
        columns = [self.feature_source, "codeword", "snr_db"]
        if baselines:
            if "llrs" not in self.dataset.schema.names:
                raise ValueError("Baseline evaluation requires the llrs column")
            columns.append("llrs")
        for start in range(0, len(order), batch_size):
            table = self.dataset.take(
                order[start : start + batch_size].tolist(), columns=columns
            )
            raw = tensor_column(table, self.feature_source, (2, self.freq_bins, 79))
            labels = tensor_column(table, "codeword", (174,))
            if not ((labels == 0) | (labels == 1)).all():
                raise ValueError("Codeword labels must be binary")
            snr = torch.tensor(table["snr_db"].to_pylist(), dtype=torch.float32)
            if not torch.isfinite(snr).all():
                raise ValueError("SNR values must be finite")
            features = prepare_features(raw, representation)
            if not torch.isfinite(features).all():
                raise ValueError("Feature conversion produced nonfinite values")
            batch = {"features": features, "labels": labels, "snr": snr}
            if baselines:
                batch["llrs"] = tensor_column(table, "llrs", (4, 174))
            yield batch
