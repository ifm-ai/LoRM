"""Block Attention Residuals (Kimi Team, 2026, https://arxiv.org/abs/2603.15031) with latent feedback.

There is no single residual sum. Every sublayer reads a softmax mix of the sources that exist at its depth: the
token embedding b0, the blocks already completed (each the sum of `block_layers` layers' sublayer outputs) and the
current block's running sum,

    l_i = s q^T rms(v_i),   x = sum_i softmax_i(l) v_i,

with a learned query q per sublayer, starting at zero, and s = dim^-1/2 with `scaled_logits` (Inkling 2's reference;
without it the queries grow with the learning rate and the mix turns near one-hot). A final mix over b0 and every
block feeds the head.

Feedback runs passes over the whole sequence; each later pass reads the earlier pass, detached unless
`full_gradient`:
- `top`: the earlier pass's output mix o, as b0 = e_t + W rms(o_{t-1}) (the latent feedback Transformer's form);
- `blocks`: each of its blocks b'_j as an extra source of every mix, through W rms(b'_j) (`feedback_map: linear`) or
  unchanged (`identity`).
`shift: 1` reads the previous position (generation runs one pass per token); `shift: 0` the position itself. With
`blocks`, `identity`, `shift: 0`, `full_gradient` and two passes, the model is a 2x loop of the stack whose second
visit reads the first visit's blocks, as Block AttnRes on a looped core does.
"""

from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from lorm.model.block import Block
from lorm.model.config import InstanceConfig, ModelConfig
from lorm.model.transformer import Transformer
from lorm.registry import MODELS


@dataclass(frozen=True)
class AttnResConfig(InstanceConfig):
    name: ClassVar[str] = "attnres"
    # Layers per AttnRes block.
    block_layers: int = 2
    # Logits times dim^-1/2.
    scaled_logits: bool = True
    # What a later pass reads of the earlier one: "none" (one pass), "top" or "blocks".
    feedback: str = "none"
    # How fed-back blocks enter the mixes: "linear" (W rms(b')) or "identity" (b' as it is).
    feedback_map: str = "linear"
    # 1 feeds the previous position's states, 0 the position's own.
    shift: int = 1
    # Passes of a default call; training may draw its own (lorm.train.algo.SampledPasses).
    passes: int = 1
    # Keep gradients through the fed-back states and every pass.
    full_gradient: bool = False

    def __post_init__(self) -> None:
        if type(self.block_layers) is not int or self.block_layers < 1:
            raise ValueError("block_layers must be a positive integer.")
        if self.feedback not in ("none", "top", "blocks"):
            raise ValueError("feedback must be none, top or blocks.")
        if self.feedback_map not in ("linear", "identity") or (self.feedback_map == "identity"
                                                                and self.feedback != "blocks"):
            raise ValueError("feedback_map must be linear, or identity with blocks feedback.")
        if self.shift not in (0, 1) or type(self.passes) is not int or self.passes < 1:
            raise ValueError("shift must be 0 or 1 and passes a positive integer.")
        if self.feedback == "none" and self.passes != 1:
            raise ValueError("A model without feedback runs one pass.")
        if type(self.scaled_logits) is not bool or type(self.full_gradient) is not bool:
            raise ValueError("scaled_logits and full_gradient must be booleans.")

    def check(self, config: ModelConfig) -> None:
        if config.n_layers % self.block_layers:
            raise ValueError("block_layers must divide n_layers.")


class DepthMix(nn.Module):
    """softmax_i(s q^T rms(v_i)) over stacked sources (sources, batch, length, dim) and an optional running sum."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.query, self.eps = nn.Parameter(torch.zeros(config.dim)), config.norm_eps
        self.scale = config.dim ** -0.5 if config.instance.scaled_logits else 1.0

    def logits(self, values: Tensor) -> Tensor:
        return (values * self.query).sum(-1) * (values.square().mean(-1) + self.eps).rsqrt() * self.scale

    def forward(self, completed: Tensor, partial: Tensor | None = None) -> Tensor:
        values = completed if partial is None else torch.cat((completed, partial.unsqueeze(0)))
        return (self.logits(values).softmax(0).unsqueeze(-1) * values).sum(0).to(completed.dtype)


class AttnResBlock(Block):
    """A pre-norm block whose attention and MLP each read their own mix and add to the current block's running sum."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.attention_residual, self.mlp_residual = DepthMix(config), DepthMix(config)

    def forward(self, completed: Tensor, partial: Tensor | None, cos: Tensor, sin: Tensor) -> Tensor:
        update = self.attention(self.norm1(self.attention_residual(completed, partial)), cos, sin)
        partial = update.to(completed.dtype) if partial is None else partial + update
        return partial + self.mlp(self.norm2(self.mlp_residual(completed, partial)))


class AttnResStack(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(AttnResBlock(config) for _ in range(config.n_layers))
        self.output_residual, self.block_layers = DepthMix(config), config.instance.block_layers

    def forward(self, context: Tensor, cos: Tensor, sin: Tensor, extra: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """The output mix, and the blocks this call completes (blocks, batch, length, dim); `extra` sources follow
        the context in every mix."""
        completed = context.unsqueeze(0)
        if extra is not None:
            completed = torch.cat((completed, extra.to(context.dtype)))
        given, partial = len(completed), None
        for index, block in enumerate(self.blocks, 1):
            partial = block(completed, partial, cos, sin)
            if index % self.block_layers == 0:
                completed, partial = torch.cat((completed, partial.unsqueeze(0))), None
        return self.output_residual(completed), completed[given:]

    def forward_macs(self, shape: tuple[int, int, int]) -> int:
        # The mixes are not matrix products.
        return sum(block.forward_macs(shape) for block in self.blocks)


class AttnResTransformer(Transformer):
    config_class = AttnResConfig

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        spec = config.instance
        self.backbone = AttnResStack(config)
        self.fusion = (nn.Linear(config.dim, config.dim, bias=False)
                       if spec.feedback != "none" and spec.feedback_map == "linear" else None)
        # Single-pass updates leave the fusion without gradients.
        self.conditional_parameters = self.fusion is not None
        self.fed_back_states = config.n_layers // spec.block_layers if spec.feedback == "blocks" else 1

    def passes_of(self, passes: int | None) -> int:
        passes = self.config.instance.passes if passes is None else passes
        if type(passes) is not int or passes < 1 or (passes > 1 and self.config.instance.feedback == "none"):
            raise ValueError("passes must be a positive integer, and 1 without feedback.")
        return passes

    def shifted(self, values: Tensor) -> Tensor:
        """Position t receives position t - 1's value with shift 1 (zeros at the first position)."""
        return F.pad(values[..., :-1, :], (0, 0, 1, 0)) if self.config.instance.shift else values

    def mapped(self, states: Tensor) -> Tensor:
        return states if self.fusion is None else self.fusion(F.rms_norm(states, (states.shape[-1],),
                                                                         eps=self.config.norm_eps))

    def forward(self, input_ids: Tensor, *, passes: int | None = None) -> Tensor:
        """The logits of the last of `passes` passes; earlier passes run without gradients unless full_gradient."""
        spec, passes = self.config.instance, self.passes_of(passes)
        x, cos, sin = self.embed(input_ids)
        state, extra = x, None
        for index in range(passes):
            with torch.set_grad_enabled(torch.is_grad_enabled() and (index == passes - 1 or spec.full_gradient)):
                if index:
                    fed = source if spec.full_gradient else source.detach()
                    if spec.feedback == "blocks":
                        extra = self.shifted(self.mapped(fed))
                    elif spec.shift:
                        # The first position has no previous one and keeps its embedding.
                        state = torch.cat([x[:, :1], x[:, 1:] + self.mapped(fed)[:, :-1]], 1)
                    else:
                        state = x + self.mapped(fed)
                top, blocks = self.backbone(state if torch.is_grad_enabled() else state.detach(), cos, sin, extra)
                source = blocks if spec.feedback == "blocks" else top
        return self.readout(top)

    @classmethod
    def eval_variants(cls, config: ModelConfig, passes: list[int] = ()) -> dict[str, dict]:
        """eval/nll@P{k}: the loss after k passes; P1 is the model without feedback."""
        return {f"P{k}": {"passes": k} for k in passes}

    def fusion_macs(self, shape: tuple[int, int]) -> int:
        return 0 if self.fusion is None else shape[0] * shape[1] * self.fusion.weight.numel() * self.fed_back_states

    def forward_macs(self, shape: tuple[int, int], *, passes: int | None = None, **options) -> int:
        earlier = self.passes_of(passes) - 1
        stack = self.backbone.forward_macs((*shape, self.config.dim))
        return super().forward_macs(shape, **options) + earlier * (stack + self.fusion_macs(shape))

    def no_grad_macs(self, shape: tuple[int, int], *, passes: int | None = None, **options) -> int:
        """Earlier passes run without gradients; the fusion feeding the last pass computes only its weight's
        gradient, half a backward."""
        earlier = self.passes_of(passes) - 1
        if not earlier or self.config.instance.full_gradient:
            return 0
        fusion = self.fusion_macs(shape)
        return earlier * self.backbone.forward_macs((*shape, self.config.dim)) + (earlier - 1) * fusion + fusion // 2

    def hidden_matrices(self) -> list[nn.Parameter]:
        return super().hidden_matrices() + ([] if self.fusion is None else [self.fusion.weight])

    def preserved_parameters(self) -> list[nn.Parameter]:
        # The queries start at zero, so every mix starts as the mean of its sources.
        return [m.query for m in self.modules() if isinstance(m, DepthMix)]


MODELS["attnres"] = AttnResTransformer
