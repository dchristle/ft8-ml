"""Streaming bit-accuracy and BCE proxy metrics, including out-of-range SNRs."""

import torch
from torch.nn.functional import binary_cross_entropy_with_logits

SNR_BINS = ((-30, -20), (-20, -15), (-15, -10), (-10, -5), (-5, 0), (0, 10))


class Metrics:
    def __init__(self):
        self.totals = {}

    def update(self, logits, labels, snr):
        if not torch.isfinite(logits).all():
            raise ValueError("Model produced nonfinite LLRs")
        losses = binary_cross_entropy_with_logits(logits, labels, reduction="none").sum(
            1
        )
        correct = ((logits > 0) == labels.bool()).sum(1)
        masks = {"overall": torch.ones_like(snr, dtype=torch.bool)}
        covered = torch.zeros_like(snr, dtype=torch.bool)
        for lo, hi in SNR_BINS:
            mask = (snr >= lo) & (snr < hi)
            masks[f"[{lo},{hi})"] = mask
            covered |= mask
        masks["other"] = ~covered
        for name, mask in masks.items():
            total = self.totals.setdefault(name, [0, 0, 0.0])
            total[0] += int(mask.sum())
            total[1] += int(correct[mask].sum())
            total[2] += float(losses[mask].double().sum())

    def result(self):
        return {
            name: {
                "examples": n,
                "bit_accuracy": correct / (n * 174) if n else None,
                "bce": loss / (n * 174) if n else None,
            }
            for name, (n, correct, loss) in self.totals.items()
        }
