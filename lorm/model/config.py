"""Model settings, serializable to and from the `model` section of an experiment YAML.

`instance` selects a registered Transformer subclass (lorm.registry.MODELS) and carries its settings, so that a work
adds a model variant without adding fields here.
"""

from dataclasses import dataclass, fields
import math
from typing import ClassVar


def _check_positive_int(name: str, value) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


@dataclass(frozen=True)
class InstanceConfig:
    """Settings of a model variant; subclasses add fields and name the variant they build."""

    name: ClassVar[str]
    # Fields that a run branched from another run's checkpoint may change (lorm.train.checkpoint).
    branch_fields: ClassVar[frozenset[str]] = frozenset()

    def to_dict(self) -> dict:
        return {"name": self.name, **{f.name: getattr(self, f.name) for f in fields(self)}}

    def check(self, config: "ModelConfig") -> None:
        """Reject a model configuration the variant cannot run."""


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    dim: int
    n_heads: int
    n_kv_heads: int
    hidden_dim: int
    max_seq_len: int
    n_layers: int
    tie_embeddings: bool = False
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    # "flash" runs Flash Attention on CUDA (bfloat16 or float16 inputs); "sdpa" lets PyTorch choose.
    attention_backend: str = "flash"
    # RMSNorm with a learned gain on each head's queries and keys, before RoPE, as in Qwen3.
    qk_norm: bool = True
    instance: InstanceConfig | None = None

    def __post_init__(self) -> None:
        for name in ("vocab_size", "dim", "n_heads", "n_kv_heads", "hidden_dim", "max_seq_len", "n_layers"):
            _check_positive_int(name, getattr(self, name))
        if self.dim % self.n_heads or self.head_dim % 2 or self.n_heads % self.n_kv_heads:
            raise ValueError("dim must split into n_heads heads of even size, and n_heads into n_kv_heads groups.")
        if self.attention_backend not in ("flash", "sdpa"):
            raise ValueError("attention_backend must be flash or sdpa.")
        if type(self.tie_embeddings) is not bool or type(self.qk_norm) is not bool:
            raise ValueError("tie_embeddings and qk_norm must be booleans.")
        for name in ("rope_theta", "norm_eps"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite.")
        if self.instance is not None:
            if not isinstance(self.instance, InstanceConfig):
                raise TypeError("instance must be an InstanceConfig; use ModelConfig.from_dict for YAML.")
            self.instance.check(self)

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    def to_dict(self) -> dict:
        values = {f.name: getattr(self, f.name) for f in fields(self)}
        values["instance"] = None if self.instance is None else self.instance.to_dict()
        return values

    @classmethod
    def from_dict(cls, settings: dict) -> "ModelConfig":
        from ..registry import MODELS

        values = dict(settings)
        if values.get("instance") is not None:
            instance = dict(values["instance"])
            name = instance.pop("name", None)
            if name not in MODELS:
                raise ValueError(f"instance.name must be one of {sorted(MODELS)}; import the work that registers it.")
            values["instance"] = MODELS[name].config_class(**instance)
        return cls(**values)
