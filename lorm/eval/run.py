"""Load a trained run directory (config.yaml and checkpoint.pt) for measurement, in float32 with PyTorch's attention."""

from pathlib import Path

import torch
import yaml

from ..model import ModelConfig, build_model


def load_run(path: Path, device: torch.device):
    config = yaml.safe_load((Path(path) / "config.yaml").read_text())
    # Flash Attention needs bfloat16, whose rounding would hide the differences the measurements look for.
    model = build_model(ModelConfig.from_dict({**config["model"], "attention_backend": "sdpa"}))
    model.load_state_dict(torch.load(Path(path) / "checkpoint.pt", map_location="cpu", weights_only=True)["model"])
    return model.to(device).eval()
