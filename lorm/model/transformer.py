"""Causal language model over packed, unpadded sequences; the loss belongs to the caller.

Works subclass `Transformer` and register the subclass under their instance name (lorm.registry.MODELS). A subclass
names its settings with `config_class` and may override `eval_variants`, `forward_macs`, `hidden_matrices` and
`preserved_parameters`.
"""

import torch
from torch import Tensor, nn

from ..registry import MODELS
from .block import BlockStack, rotary_frequencies
from .config import InstanceConfig, ModelConfig


class Transformer(nn.Module):
    config_class: type[InstanceConfig] | None = None
    # Whether some updates leave parameters without gradients, which DDP must then look for.
    conditional_parameters = False

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.backbone = BlockStack(config, config.n_layers)
        self.norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.head.weight = self.embedding.weight

    def embed(self, input_ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """(token embeddings, RoPE cos, RoPE sin) for positions 0..length-1."""
        if input_ids.ndim != 2 or not 1 <= input_ids.shape[1] <= self.config.max_seq_len:
            raise ValueError("input_ids must be (batch, length) with 1..max_seq_len tokens.")
        cos, sin = rotary_frequencies(input_ids.shape[1], self.config.head_dim, self.config.rope_theta, input_ids.device)
        return self.embedding(input_ids), cos, sin

    def blocks(self, x: Tensor, cos: Tensor, sin: Tensor, start: int = 0, stop: int | None = None) -> Tensor:
        """Run backbone blocks start..stop-1."""
        for block in self.backbone.blocks[start:stop]:
            x = block(x, cos, sin)
        return x

    def readout(self, state: Tensor) -> Tensor:
        return self.head(self.norm(state))

    def forward(self, input_ids: Tensor) -> Tensor:
        """(batch, length, vocabulary) logits; the logits at t predict token t + 1."""
        x, cos, sin = self.embed(input_ids)
        return self.readout(self.blocks(x, cos, sin))

    @classmethod
    def eval_variants(cls, config: ModelConfig, **settings) -> dict[str, dict]:
        """Metric suffix -> forward options of extra validation calls (lorm.eval.nll), from the YAML's
        `eval.variants` settings; the plain model takes none."""
        if settings:
            raise ValueError(f"{cls.__name__} takes no eval.variants settings.")
        return {}

    def forward_macs(self, shape: tuple[int, int], **options) -> int:
        """Multiply-adds of one forward call on input_ids of `shape`, matrix products only."""
        batch, length = shape
        return self.backbone.forward_macs((batch, length, self.config.dim)) + batch * length * self.head.weight.numel()

    def no_grad_macs(self, shape: tuple[int, int], **options) -> int:
        """The part of forward_macs(shape, **options) that runs without gradients, so has no backward."""
        return 0

    def hidden_matrices(self) -> list[nn.Parameter]:
        """The matrices that Muon-type optimizers train; embeddings, norms and the head stay on AdamW."""
        return [p for block in self.backbone.blocks for p in block.matrices()]

    def preserved_parameters(self) -> list[nn.Parameter]:
        """Parameters `initialize` leaves as their modules made them."""
        return []


def initialize(model: Transformer, std: float = 0.02) -> None:
    """Gains of one and every other weight from N(0, std^2), as in OLMo 2, in parameter order; parameters the model
    preserves keep their own initialization."""
    preserved = {id(p) for p in model.preserved_parameters()}
    for parameter in model.parameters():
        if id(parameter) in preserved:
            continue
        if parameter.ndim == 1:
            nn.init.ones_(parameter)
        else:
            nn.init.normal_(parameter, mean=0.0, std=std)


def model_class(config: ModelConfig) -> type[Transformer]:
    return Transformer if config.instance is None else MODELS[config.instance.name]


def build_model(config: ModelConfig) -> Transformer:
    return model_class(config)(config)
