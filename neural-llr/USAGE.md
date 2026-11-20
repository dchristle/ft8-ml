# Neural LLR Training, Evaluation, and ONNX Export

Scripts for training the four soft-demodulation architectures described in
[README.md](README.md), evaluating held-out bit accuracy and binary
cross-entropy (BCE), and exporting fixed-batch ONNX models. The workflow
operates on pre-extracted time-frequency patches in Lance format; WAV signal
extraction and downstream LDPC/OSD decoding are external.

## Environment

Requires Python 3.11 or later:

```sh
uv venv
uv pip install -r requirements.txt
```

Training and evaluation run in `float32` and default to `--device cpu`; pass
`--device cuda` or `--device mps` to use an accelerator.

## Dataset Schema

Supply a Lance dataset directory (such as `/path/to/training.lance`) with one
row per labeled signal example. Tensor columns are stored as flat row-major
`float32` lists of length `prod(shape)`:

| Column | Logical shape | Description |
| --- | --- | --- |
| `spectrogram32` | `(2, 32, 79)` | `[real, imag]` complex symbol FFT patch (default `--feature-source spectrogram32`). |
| `spectrogram` | `(2, 8, 79)` | Narrow 8-tone patch, used only with `--feature-source spectrogram`. |
| `codeword` | `(174,)` | Ground-truth binary `{0, 1}` codeword bits in transmission order. |
| `snr_db` | Scalar | Finite SNR estimate in dB. |
| `wav_stem` | String | Recording identifier prefixed with `YYMMDD_`. |
| `llrs` | `(4, 174)` | Optional baseline LLRs ordered `(llra, llrb, llrc, llrd)`, used with `evaluate.py --baselines`. |

Only the selected spectrogram column (`spectrogram32` or `spectrogram`) is
required. Positive LLR values favor bit `1`; for hard-decision bit accuracy,
`LLR > 0` maps to `1` and `LLR <= 0` maps to `0`. Stored baseline LLRs must use
the same sign convention. Loader batches validate tensor shapes, finiteness, and
binary codeword values on read.

`train.py` requires an explicit `--representation` flag (neither representation
normalizes per-example amplitude):

- `raw-complex`: passes `(real, imag)` directly as channels `(0, 1)`. This
  matches the experiments reported in [README.md](README.md).
- `magnitude-phase`: converts `(real, imag)` to magnitude and wrapped same-bin
  temporal phase increment:
  - channel `0`: `hypot(real, imag)`
  - channel `1`: `0` at symbol `t = 0`, and for `1 <= t < 79` the phase of
    `z[f, t] * conj(z[f, t-1])` with `z = real + i*imag`, computed per frequency bin `f`:
    `atan2(imag[f, t] * real[f, t-1] - real[f, t] * imag[f, t-1], real[f, t] * real[f, t-1] + imag[f, t] * imag[f, t-1])`

The selected representation is saved in the checkpoint and applied
automatically by `evaluate.py`.

## Train

```sh
.venv/bin/python train.py --data /path/to/training.lance \
  --output runs/transformer --model transformer \
  --representation raw-complex --test-date 241123 \
  --epochs 50 --batch-size 128
```

`--model` accepts `mlp`, `conv1d`, `conv2d`, or `transformer`. `--test-date`
holds out all recordings with that `YYMMDD` prefix for testing; the remaining
date groups (at least two required, so three distinct dates minimum overall)
are shuffled with `--seed` (default `42`) and partitioned into validation
(`--val-fraction`, default `0.1`) and training. Pass identical split arguments
when comparing architectures.

Training uses `AdamW` (`--lr`, default `1e-3`; `weight_decay=1e-4`), gradient
norm clipping at `1.0`, and `ReduceLROnPlateau` (`factor=0.5`, `patience=20`)
on validation BCE. Each run writes a new `--output` directory containing:

- `run.json`: model, feature contract, pinned Lance version and `wav_stem`
  SHA-256 digest, date-group split membership, seed, parameter count, and
  hyperparameters.
- `history.jsonl`: per-epoch sample-weighted training and validation
  `bit_accuracy` and `bce`.
- `best.pt`: `format_version: 1` checkpoint storing the configuration, split,
  and model weights from the epoch with lowest validation BCE.

## Evaluate Held-Out Proxy Metrics

```sh
.venv/bin/python evaluate.py --data /path/to/training.lance \
  --checkpoint runs/transformer/best.pt \
  --output runs/transformer/evaluation.json --baselines
```

Omit `--baselines` when the dataset has no `llrs` column. `evaluate.py` reopens
the dataset at the Lance version recorded in `best.pt`, verifies the row count
and ordered `wav_stem` digest, and evaluates the checkpoint's held-out `test`
split. Keep the training Lance version intact for reproducible evaluation.

`--output` must be a new path. The JSON report records dataset identity, test
dates, recording count, and `bit_accuracy` and `bce` across `overall`, the six
SNR bins `[-30, -20)` through `[0, 10)` dB, and `other` (`null` when a bin is
empty).

## Export to ONNX

```sh
.venv/bin/python export_onnx.py --checkpoint runs/transformer/best.pt \
  --output runs/transformer/model.onnx --batch-size 1
```

Exports an opset-17 ONNX graph (`features -> llrs`) from a `train.py`
checkpoint with a fixed `--batch-size`, runs `onnx.checker.check_model`, and
verifies numerical agreement between PyTorch and ONNX Runtime
(`CPUExecutionProvider`) on a synthetic input tensor. A companion `.json` file
records the model metadata, `representation`, fixed input/output shapes, and
`positive_llr_bit: 1`.

Note that the ONNX graph consumes **prepared `(batch, 2, F, 79)` features**,
not raw audio; callers running a `magnitude-phase` model must apply the
magnitude/phase-difference transform above before inference.

## Tests

```sh
.venv/bin/python -m unittest -v test_pipeline
```

Runs synthetic unit and integration tests covering global-phase invariance of
`magnitude-phase`, batch-weighted metric accumulation and out-of-range SNR
bins, date-group split isolation, Lance version pinning, and train/evaluate/ONNX
round trips for all four models.
