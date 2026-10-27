from __future__ import annotations

from pathlib import Path

import torch


def load_checkpoint_into_model(model, checkpoint_path: str | Path, device: str = "cpu"):
    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]

    cleaned = {}
    for key, value in checkpoint.items():
        cleaned[key.removeprefix("module.")] = value
    model.load_state_dict(cleaned)
    model.to(device)
    model.eval()
    return model
