"""Train a neural FT8 soft demodulator on prepared signal examples."""

import argparse
import json
import math
from pathlib import Path

import torch
from data import REPRESENTATIONS, SignalData
from metrics import Metrics
from models import MODEL_NAMES, build_model, count_params


def run_epoch(model, batches, device, optimizer=None):
    model.train(optimizer is not None)
    metrics = Metrics()
    with torch.set_grad_enabled(optimizer is not None):
        for batch in batches:
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)
            logits = model(features)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training or validation loss")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                )
                optimizer.step()
            metrics.update(logits.detach().cpu(), labels.detach().cpu(), batch["snr"])
    return metrics.result()["overall"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, required=True, help="Path to a prepared Lance dataset"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory for this run"
    )
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--representation", choices=REPRESENTATIONS, required=True)
    parser.add_argument(
        "--feature-source",
        choices=("spectrogram", "spectrogram32"),
        default="spectrogram32",
    )
    parser.add_argument(
        "--test-date", required=True, help="Held-out YYMMDD recording group"
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    args = parser.parse_args()
    if (
        args.epochs < 1
        or args.batch_size < 1
        or not math.isfinite(args.lr)
        or args.lr <= 0
    ):
        parser.error(
            "Epochs, batch size, and learning rate must be positive and finite"
        )
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    data = SignalData(args.data, args.feature_source)
    split = data.make_split(args.test_date, args.val_fraction, args.seed)
    train_rows = data.indices(split, "train")
    validation_rows = data.indices(split, "validation")
    model = build_model(args.model, freq_bins=data.freq_bins).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=20, factor=0.5
    )
    args.output.mkdir(parents=True, exist_ok=False)
    config = {
        "format_version": 1,
        "model": args.model,
        "freq_bins": data.freq_bins,
        "feature_source": args.feature_source,
        "representation": args.representation,
        "dataset": data.identity(),
        "split": split,
        "seed": args.seed,
        "parameters": count_params(model),
        "torch_version": str(torch.__version__),
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "device": args.device,
    }
    (args.output / "run.json").write_text(json.dumps(config, indent=2) + "\n")
    best = float("inf")
    with (args.output / "history.jsonl").open("w") as history:
        for epoch in range(1, args.epochs + 1):
            training = run_epoch(
                model,
                data.batches(
                    train_rows,
                    args.batch_size,
                    args.representation,
                    seed=args.seed + epoch,
                ),
                device,
                optimizer,
            )
            validation = run_epoch(
                model,
                data.batches(validation_rows, args.batch_size, args.representation),
                device,
            )
            record = {"epoch": epoch, "train": training, "validation": validation}
            history.write(json.dumps(record, allow_nan=False) + "\n")
            history.flush()
            print(json.dumps(record, allow_nan=False), flush=True)
            if validation["bce"] < best:
                best = validation["bce"]
                checkpoint = dict(
                    config,
                    epoch=epoch,
                    validation=validation,
                    state_dict={
                        k: v.detach().cpu() for k, v in model.state_dict().items()
                    },
                )
                temporary = args.output / "best.tmp"
                torch.save(checkpoint, temporary)
                temporary.replace(args.output / "best.pt")
            scheduler.step(validation["bce"])


if __name__ == "__main__":
    main()
