"""The feedback Transformer: each position's input adds a linear map of the previous position's RMS-normalized
top-layer state (Fan et al., 2020), u_t = e_t + W rms(h_{t-1}).

Training runs passes over the whole sequence: the first reads no feedback, and each later pass reads the previous
pass's top states, detached. Token-by-token decoding is the fixed point the passes approach.
"""

from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..registry import MODELS
from .config import InstanceConfig, ModelConfig
from .transformer import Transformer


def _check_passes(passes) -> None:
    if type(passes) is not int or passes < 1:
        raise ValueError("passes must be a positive integer.")


@dataclass(frozen=True)
class FeedbackConfig(InstanceConfig):
    name: ClassVar[str] = "feedback"
    # Passes of a default call, which evaluation uses; training draws its own (lorm.train.algo.SampledPasses).
    passes: int = 2

    def __post_init__(self) -> None:
        _check_passes(self.passes)


class FeedbackTransformer(Transformer):
    config_class = FeedbackConfig
    # Single-pass updates leave the fusion without gradients.
    conditional_parameters = True

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.fusion = nn.Linear(config.dim, config.dim, bias=False)

    def mix(self, embedding: Tensor, top: Tensor) -> Tensor:
        """e_t + W rms(top_{t-1}); the first position has no previous one and keeps its embedding."""
        fed = self.fusion(F.rms_norm(top, (top.shape[-1],), eps=self.config.norm_eps))
        return torch.cat([embedding[:, :1], embedding[:, 1:] + fed[:, :-1]], 1)

    def passes_of(self, passes: int | None) -> int:
        passes = self.config.instance.passes if passes is None else passes
        _check_passes(passes)
        return passes

    def forward(self, input_ids: Tensor, *, passes: int | None = None) -> Tensor:
        """The logits of the last of `passes` passes; earlier passes run without gradients."""
        passes = self.passes_of(passes)
        x, cos, sin = self.embed(input_ids)
        state = x
        for index in range(passes):
            with torch.set_grad_enabled(torch.is_grad_enabled() and index == passes - 1):
                if index:
                    state = self.mix(x, top.detach())
                top = self.blocks(state, cos, sin)
        return self.readout(top)

    @classmethod
    def eval_variants(cls, config: ModelConfig, passes: list[int] = ()) -> dict[str, dict]:
        """eval/nll@P{k}: the loss after k passes; P1 is the model without feedback."""
        for k in passes:
            _check_passes(k)
        return {f"P{k}": {"passes": k} for k in passes}

    def feedback_macs(self, shape: tuple[int, int], passes: int) -> int:
        """The blocks and fusion that every pass after the first adds."""
        batch, length = shape
        return (passes - 1) * (self.backbone.forward_macs((batch, length, self.config.dim))
                               + batch * length * self.fusion.weight.numel())

    def forward_macs(self, shape: tuple[int, int], *, passes: int | None = None, **options) -> int:
        return super().forward_macs(shape, **options) + self.feedback_macs(shape, self.passes_of(passes))

    def no_grad_macs(self, shape: tuple[int, int], *, passes: int | None = None, **options) -> int:
        """The earlier passes run without gradients; the fusion that feeds the last pass computes only its weight's
        gradient, half a backward."""
        passes = self.passes_of(passes)
        if passes == 1:
            return 0
        batch, length = shape
        fusion = batch * length * self.fusion.weight.numel()
        return (passes - 1) * self.backbone.forward_macs((batch, length, self.config.dim)) + (passes - 2) * fusion + fusion // 2

    def hidden_matrices(self) -> list[nn.Parameter]:
        return super().hidden_matrices() + [self.fusion.weight]


MODELS["feedback"] = FeedbackTransformer
