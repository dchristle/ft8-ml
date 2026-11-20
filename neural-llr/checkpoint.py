"""Load a model together with its input contract and saved dataset split."""

import torch
from data import SignalData
from models import build_model


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Expected a version 1 checkpoint created by train.py")
    model = build_model(checkpoint["model"], freq_bins=checkpoint["freq_bins"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def open_checkpoint_data(path, checkpoint):
    data = SignalData(
        path, checkpoint["feature_source"], checkpoint["dataset"]["version"]
    )
    if data.identity() != checkpoint["dataset"]:
        raise ValueError(
            "Dataset version or recording membership differs from the training run"
        )
    return data
