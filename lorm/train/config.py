"""Experiment YAML loading and the training controls."""

from copy import deepcopy
from dataclasses import dataclass, field, fields
import math
from pathlib import Path

import yaml


def merge(base, override):
    """Override base: mappings merge key by key, other values replace, and a mapping with `_replace: true` replaces
    the base's mapping instead of merging into it."""
    if not isinstance(override, dict):
        return deepcopy(override)
    override = dict(override)
    if override.pop("_replace", False) or not isinstance(base, dict):
        base = {}
    merged = deepcopy(base)
    for key, value in override.items():
        merged[key] = merge(base.get(key), value)
    return merged


def load_config(path) -> dict:
    """Read an experiment YAML merged over the YAML its `base` names, relative to this one."""
    path = Path(path)
    with path.open() as source:
        config = yaml.safe_load(source)
    if "base" not in config:
        return config
    return merge(load_config(path.parent / config.pop("base")), config)


@dataclass(frozen=True)
class TrainConfig:
    max_steps: int
    batch_size: int
    gradient_accumulation: int = 1
    device: str = "cuda"
    precision: str = "bfloat16"
    # Warmup, then constant ("wsd": warmup-stable-decay, decaying over the last decay_steps) or cosine.
    lr_schedule: str = "wsd"
    warmup_steps: int = 0
    decay_steps: int = 0
    decay_type: str = "linear"
    min_lr_ratio: float = 0.0
    # torch.compile each Transformer block in place ("blocks"), or not at all ("none").
    compile: str = "none"
    # Dense bfloat16 peak TFLOPs of one GPU; logs perf/mfu when set.
    peak_tflops: float | None = None
    max_grad_norm: float | None = 1.0
    log_every: int = 10
    eval_every: int = 100
    checkpoint_every: int = 100
    checkpoint_keep: int = 2
    # Steps whose checkpoints stay as checkpoint-step<N>.pt, so that a WSD decay can branch from them (branch_from).
    branch_points: list[int] = field(default_factory=list)
    shuffle: bool = True
    # Whether each epoch draws a new order; False repeats the first epoch's order.
    reshuffle: bool = True

    def __post_init__(self):
        for name in ("max_steps", "batch_size", "gradient_accumulation", "log_every"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("warmup_steps", "decay_steps", "eval_every", "checkpoint_every"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer.")
        if self.warmup_steps >= self.max_steps:
            raise ValueError("warmup_steps must be below max_steps.")
        if self.lr_schedule not in ("wsd", "cosine") or self.decay_type not in ("linear", "cosine"):
            raise ValueError("lr_schedule must be wsd or cosine, and decay_type linear or cosine.")
        if self.lr_schedule == "wsd" and not 0 < self.decay_steps <= self.max_steps - self.warmup_steps:
            raise ValueError("WSD needs 0 < decay_steps <= max_steps - warmup_steps.")
        if self.compile not in ("none", "blocks"):
            raise ValueError("compile must be none or blocks.")
        if self.peak_tflops is not None and (not math.isfinite(self.peak_tflops) or self.peak_tflops <= 0):
            raise ValueError("peak_tflops must be positive and finite, or null.")
        points = self.branch_points
        if (type(points) is not list or points != sorted(set(points))
                or any(type(step) is not int or not 0 < step < self.max_steps for step in points)):
            raise ValueError("branch_points must list increasing steps between 0 and max_steps.")
        if self.checkpoint_keep not in (1, 2) or type(self.shuffle) is not bool or type(self.reshuffle) is not bool:
            raise ValueError("checkpoint_keep must be one or two, and shuffle and reshuffle booleans.")
        if not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio must be between zero and one.")
        if self.max_grad_norm is not None and (not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0):
            raise ValueError("max_grad_norm must be positive and finite, or null to disable clipping.")
        if self.device not in ("cpu", "cuda") or self.precision not in ("float32", "bfloat16"):
            raise ValueError("Use device cpu or cuda with precision float32 or bfloat16.")

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def evaluates_at(self, step: int) -> bool:
        return bool(self.eval_every) and (step % self.eval_every == 0 or step == self.max_steps)
