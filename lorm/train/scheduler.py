"""Native Transformers schedules, advanced after each optimizer update.

Upstream warmup and endpoint conventions are retained.
"""

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup, get_wsd_schedule

from .config import TrainConfig


def build_scheduler(optimizer: Optimizer, config: TrainConfig) -> LambdaLR:
    if config.lr_schedule == "cosine":
        return get_cosine_with_min_lr_schedule_with_warmup(
            optimizer, num_warmup_steps=config.warmup_steps,
            num_training_steps=config.max_steps, min_lr_rate=config.min_lr_ratio,
        )
    return get_wsd_schedule(
        optimizer, num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_steps, num_decay_steps=config.decay_steps,
        min_lr_ratio=config.min_lr_ratio, warmup_type="linear", decay_type=config.decay_type,
    )
