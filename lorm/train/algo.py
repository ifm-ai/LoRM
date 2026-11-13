"""How one optimizer update supervises the model.

The loop owns accumulation, gradient synchronization, clipping and optimizer steps; an algorithm owns the forward pass
and the loss. `begin_update` runs once per update before any microbatch; per-update choices derive from (seed, step),
so every rank and microbatch agrees and a resumed run repeats them. `loss` returns the summed token loss of one
microbatch, which the loop divides by the global number of targets. Works register algorithms in
lorm.registry.ALGORITHMS under the `algo.name` that selects them.
"""

from dataclasses import asdict, dataclass, field
import math

import numpy as np
from torch import Tensor

from ..registry import ALGORITHMS
from .loss import causal_loss


def training_flops_per_token(model, sequence_length: int, **options) -> int:
    """Model FLOPs per token of one forward and backward pass: two per multiply-add, the backward twice the forward,
    and no backward for work run without gradients. Recomputation is not counted."""
    shape = (1, sequence_length)
    return (6 * model.forward_macs(shape, **options) - 4 * model.no_grad_macs(shape, **options)) // sequence_length


@dataclass(frozen=True)
class NoOptions:
    pass


class Algorithm:
    """The plain next-token loss of the model's default call."""

    config_class = NoOptions

    def __init__(self, config=NoOptions(), seed: int = 0) -> None:
        self.config, self.seed = config, seed

    def begin_update(self, step: int) -> None:
        """Prepare the update that follows `step` completed updates."""

    def loss(self, model, inputs: Tensor, labels: Tensor) -> Tensor:
        return causal_loss(model(inputs), labels)

    def flops_per_token(self, model, sequence_length: int) -> int:
        return training_flops_per_token(model, sequence_length)

    def metrics(self) -> dict[str, float]:
        """Metrics accumulated since the previous call."""
        return {}

    def state_dict(self) -> dict:
        """This rank's metric window, so that a resumed run logs the same values."""
        return {}

    def load_state_dict(self, state: dict) -> None:
        pass


@dataclass(frozen=True)
class PassMix:
    # The probabilities of 1, 2, ... passes.
    pass_mix: list[float] = field(default_factory=lambda: [0.31, 0.69])

    def __post_init__(self) -> None:
        mix = self.pass_mix
        if (type(mix) is not list or not mix or any(type(p) not in (int, float) or not 0 <= p <= 1 for p in mix)
                or not math.isclose(sum(mix), 1)):
            raise ValueError("pass_mix must list the probabilities of 1, 2, ... passes, summing to one.")


class SampledPasses(Algorithm):
    """For models whose call takes `passes` (lorm.model.feedback): each update draws its number of passes from
    `pass_mix` and trains the last pass's next-token loss. Single-pass updates keep the model without feedback
    trained; multi-pass updates feed the last pass states closer to token-by-token decoding."""

    config_class = PassMix

    def __init__(self, config: PassMix, seed: int) -> None:
        super().__init__(config, seed)
        self.passes, self.window = 1, []

    def begin_update(self, step: int) -> None:
        rng = np.random.default_rng([self.seed, step])
        self.passes = int(rng.choice(len(self.config.pass_mix), p=self.config.pass_mix)) + 1
        self.window.append(self.passes)

    def loss(self, model, inputs, labels):
        return causal_loss(model(inputs, passes=self.passes), labels)

    def flops_per_token(self, model, sequence_length):
        return training_flops_per_token(model, sequence_length, passes=self.passes)

    def metrics(self) -> dict[str, float]:
        if not self.window:
            return {}
        metrics = {"train/passes_mean": sum(self.window) / len(self.window)}
        self.window.clear()
        return metrics

    def state_dict(self) -> dict:
        return {"passes": list(self.window)}

    def load_state_dict(self, state: dict) -> None:
        self.window = list(state["passes"])


ALGORITHMS["standard"] = Algorithm
ALGORITHMS["sampled_passes"] = SampledPasses


def resolve_algorithm(settings: dict | None) -> dict:
    """The `algo` section with its options validated by the algorithm's config class."""
    settings = dict(settings or {"name": "standard"})
    name = settings.pop("name", None)
    if name not in ALGORITHMS:
        raise ValueError(f"algo.name must be one of {sorted(ALGORITHMS)}; import the work that registers it.")
    return {"name": name, **asdict(ALGORITHMS[name].config_class(**settings))}


def build_algorithm(settings: dict, *, seed: int) -> Algorithm:
    algorithm = ALGORITHMS[settings["name"]]
    return algorithm(algorithm.config_class(**{k: v for k, v in settings.items() if k != "name"}), seed)
