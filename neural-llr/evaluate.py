"""Report held-out bit accuracy and BCE; these are proxies, not decode counts."""

import argparse
import json
from pathlib import Path

import torch
from checkpoint import load_checkpoint, open_checkpoint_data
from metrics import Metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--baselines",
        action="store_true",
        help="Also evaluate stored llra/llrb/llrc/llrd",
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("Batch size must be positive")
    model, checkpoint = load_checkpoint(args.checkpoint)
    model.to(args.device)
    data = open_checkpoint_data(args.data, checkpoint)
    rows = data.indices(checkpoint["split"], "test")
    names = [checkpoint["model"]] + (
        ["llra", "llrb", "llrc", "llrd"] if args.baselines else []
    )
    metrics = {name: Metrics() for name in names}
    with torch.no_grad():
        for batch in data.batches(
            rows,
            args.batch_size,
            checkpoint["representation"],
            baselines=args.baselines,
        ):
            predictions = [model(batch["features"].to(args.device)).cpu()]
            if args.baselines:
                predictions.extend(batch["llrs"].unbind(1))
            for name, logits in zip(names, predictions):
                metrics[name].update(logits, batch["labels"], batch["snr"])
    report = {
        "metric_scope": "Bit accuracy and BCE proxy metrics; no decoding or false-positive analysis",
        "dataset": data.identity(),
        "test_dates": checkpoint["split"]["test"],
        "recordings": len({data.stems[i] for i in rows}),
        "representation": checkpoint["representation"],
        "model_epoch": checkpoint["epoch"],
        "results": {name: metric.result() for name, metric in metrics.items()},
    }
    with args.output.open("x") as output:
        output.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
