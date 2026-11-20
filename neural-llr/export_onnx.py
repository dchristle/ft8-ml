"""Export a fixed-batch neural LLR model and verify ONNX Runtime agreement."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New .onnx file")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("Batch size must be positive")
    metadata_path = args.output.with_suffix(".json")
    if args.output.suffix != ".onnx" or args.output.exists() or metadata_path.exists():
        parser.error("Use a new .onnx output path with no existing metadata file")
    model, checkpoint = load_checkpoint(args.checkpoint)
    torch.manual_seed(0)
    example = torch.randn(args.batch_size, 2, checkpoint["freq_bins"], 79)
    if checkpoint["representation"] == "magnitude-phase":
        example[:, 0].abs_()
    fastpath = torch.backends.mha.get_fastpath_enabled()
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        with torch.no_grad():
            torch.onnx.export(
                model,
                example,
                str(args.output),
                input_names=["features"],
                output_names=["llrs"],
                opset_version=17,
                dynamo=False,
            )
            expected = model(example).numpy()
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)
    onnx.checker.check_model(str(args.output))
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"features": example.numpy()})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    metadata = {
        key: checkpoint[key]
        for key in ("model", "freq_bins", "representation", "feature_source", "epoch")
    }
    metadata.update(
        input_shape=list(example.shape),
        output_shape=list(actual.shape),
        positive_llr_bit=1,
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Exported and verified {args.output}")


if __name__ == "__main__":
    main()
