"""Feedback Transformers with multi-token prediction (MTP) heads, and the training of their stand-ins.

A feedback Transformer feeds each position's top-layer state into the input of the next position (Fan et al., 2020):
at decoding time u_t = e_t + W rms(h_{t-1}). Training runs passes over the whole sequence: the first reads no
feedback, and each later pass reads the previous pass's top states, detached (lorm.model.feedback).

DeepSeek-V3-style MTP heads (`MTPModule`) chain one block per future token: g_1 = M_1(h_t, e_{t+1}) and
g_k = M_k(g_{k-1}, e_{t+k}); head k predicts token t + k + 1 from g_k. In speculative decoding, the position that
verifies draft k reads a stand-in a_k in place of h, which the same verification call is still computing. The stand-in
is zeros or g_k; `stand_in` selects how training prepares g_k for that use:

- "none": the MTP loss alone, which trains g_k to predict a token and nothing to be read.
- "state_alignment": a map P, an MLP starting at the identity, pulls P(g_k at t) towards the first pass's state at t + k
  with a smooth L1 loss; only the map trains on it.
- "in_place": the last pass itself reads the stand-ins, made by the MTP chain on the earlier pass's states, by
  verification window (`window_offsets`), in a `share` of the sequences; the ordinary next-token loss trains them. No
  pass, parameter or loss weight is added.
"""

from dataclasses import dataclass
import math
from typing import ClassVar, NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from lorm.model import BlockStack, InstanceConfig, ModelConfig, Transformer
from lorm.model.feedback import FeedbackConfig, FeedbackTransformer
from lorm.registry import MODELS

STAND_INS = ("none", "state_alignment", "in_place")


@dataclass(frozen=True)
class MTPConfig(InstanceConfig):
    """A Transformer with `heads` chained MTP heads, whose losses share `mtp_weight` equally."""

    name: ClassVar[str] = "mtp"
    heads: int = 3
    mtp_weight: float = 0.3

    def __post_init__(self) -> None:
        if type(self.heads) is not int or self.heads < 1:
            raise ValueError("instance.heads must be a positive integer.")
        if type(self.mtp_weight) not in (int, float) or not 0 < self.mtp_weight < math.inf:
            raise ValueError("instance.mtp_weight must be positive and finite.")


@dataclass(frozen=True)
class MTPFeedbackConfig(FeedbackConfig, MTPConfig):
    name: ClassVar[str] = "mtp_feedback"
    # A run that branches from another's checkpoint may change how the stand-in trains (the decay-only arm).
    branch_fields: ClassVar[frozenset[str]] = frozenset({"stand_in", "window", "share", "alignment_weight"})
    stand_in: str = "none"
    # A verification window goes on past each position with this probability, for up to `heads` stand-ins, as
    # drafts accepted at that rate do ("in_place").
    window: float = 0.75
    # "in_place": the share of sequences in each multi-pass update that read stand-ins.
    share: float = 1.0
    alignment_weight: float = 1.0

    def __post_init__(self) -> None:
        MTPConfig.__post_init__(self)
        FeedbackConfig.__post_init__(self)
        if self.stand_in not in STAND_INS:
            raise ValueError(f"instance.stand_in must be one of {', '.join(STAND_INS)}.")
        if type(self.window) is not float or not 0 < self.window < 1:
            raise ValueError("instance.window must be a probability in (0, 1).")
        if type(self.share) not in (int, float) or not 0 < self.share <= 1 or (self.share < 1 and self.stand_in != "in_place"):
            raise ValueError("instance.share must be in (0, 1], and below 1 only with stand_in in_place.")
        if type(self.alignment_weight) not in (int, float) or not 0 < self.alignment_weight < math.inf:
            raise ValueError("instance.alignment_weight must be positive and finite.")


class MTPOutput(NamedTuple):
    # Head k's logits of token t + k + 1 at positions t < length - k.
    logits: tuple[Tensor, ...]
    # The summed alignment loss of stand_in "state_alignment", unweighted; None otherwise.
    stand_in: Tensor | None = None


class FeedbackMap(nn.Module):
    """A linear map plus a one-hidden-layer MLP of its RMS-normalized input that starts at zero, so that the whole
    starts at the identity."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.linear = nn.Linear(config.dim, config.dim, bias=False)
        self.up = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.down = nn.Linear(config.hidden_dim, config.dim, bias=False)
        nn.init.eye_(self.linear.weight)
        nn.init.normal_(self.up.weight, std=0.02)
        nn.init.zeros_(self.down.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x) + self.down(F.silu(self.up(F.rms_norm(x, (x.shape[-1],)))))


class MTPModule(nn.Module):
    """DeepSeek-V3's MTP module of depth 1 (arXiv:2412.19437, Sec. 2.2): it normalizes a state and the embedding of
    the next token, concatenates and projects them, and runs one block."""

    def __init__(self, config: ModelConfig, feedback_map: bool = False) -> None:
        super().__init__()
        self.state_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.embedding_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.projection = nn.Linear(2 * config.dim, config.dim, bias=False)
        self.blocks = BlockStack(config, 1)
        self.feedback = None
        if feedback_map:
            # Forked, so that the model's other parameters start as they do without the map.
            with torch.random.fork_rng(devices=[]):
                self.feedback = FeedbackMap(config)

    def forward(self, state: Tensor, embedding: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        fused = torch.cat([self.state_norm(state), self.embedding_norm(embedding)], -1)
        return self.blocks(self.projection(fused), cos, sin)

    def forward_macs(self, shape: tuple[int, int, int]) -> int:
        return math.prod(shape[:-1]) * self.projection.weight.numel() + self.blocks.forward_macs(shape)


def alignment_loss(prediction: Tensor, target: Tensor, offset: int) -> Tensor:
    """The summed smooth L1 distance, averaged over channels, between RMS-normalized predictions at t and targets at
    t + offset; the targets carry no gradient."""
    prediction, target = (F.rms_norm(x.float(), (x.shape[-1],)) for x in (prediction, target[:, offset:].detach()))
    return F.smooth_l1_loss(prediction, target, reduction="none").mean(-1).sum()


class MTPTransformer(Transformer):
    """A Transformer with MTP heads; `forward(..., mtp=True)` also returns their logits."""

    config_class = MTPConfig

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        instance = config.instance
        self.mtp = MTPModule(config, feedback_map=getattr(instance, "stand_in", None) == "state_alignment")
        self.mtp_later = nn.ModuleList(MTPModule(config) for _ in range(instance.heads - 1))

    def heads(self) -> list[MTPModule]:
        return [self.mtp, *self.mtp_later]

    def mtp_chain(self, state: Tensor, embedding: Tensor, cos: Tensor, sin: Tensor) -> list[Tensor]:
        """Every head's state at each position: head k at t reads head k - 1's (the model's for k = 1) and token
        t + k, so it covers positions t < length - k."""
        chain = []
        for k, module in enumerate(self.heads(), 1):
            chain.append(module(state[:, :-1], embedding[:, k:], cos[:-k], sin[:-k]))
            state = chain[-1]
        return chain

    def forward(self, input_ids: Tensor, *, mtp: bool = False) -> Tensor | tuple[Tensor, MTPOutput]:
        x, cos, sin = self.embed(input_ids)
        top = self.blocks(x, cos, sin)
        return self.mtp_readout(top, x, cos, sin) if mtp else self.readout(top)

    def mtp_readout(self, top: Tensor, embedding: Tensor, cos: Tensor, sin: Tensor,
                    stand_in_loss=None) -> tuple[Tensor, MTPOutput]:
        """The next-token logits and the heads' logits; `stand_in_loss(chain)` adds a loss of the heads' states."""
        chain = self.mtp_chain(top, embedding, cos, sin)
        logits = tuple(self.readout(state) for state in chain)
        return self.readout(top), MTPOutput(logits, None if stand_in_loss is None else stand_in_loss(chain))

    def forward_macs(self, shape: tuple[int, int], *, mtp: bool = False, **options) -> int:
        return super().forward_macs(shape) + (self.mtp_macs(shape) if mtp else 0)

    def mtp_macs(self, shape: tuple[int, int]) -> int:
        """The heads and their readouts; head k runs over the first length - k positions."""
        batch, length = shape
        return sum(module.forward_macs((batch, length - k, self.config.dim)) + batch * (length - k) * self.head.weight.numel()
                   for k, module in enumerate(self.heads(), 1))

    def hidden_matrices(self) -> list[nn.Parameter]:
        matrices = super().hidden_matrices()
        for module in self.heads():
            matrices += [module.projection.weight, *([] if module.feedback is None else module.feedback.parameters()),
                         *module.blocks.blocks[0].matrices()]
        return matrices

    def preserved_parameters(self) -> list[nn.Parameter]:
        # The feedback map starts at the identity.
        return [] if self.mtp.feedback is None else list(self.mtp.feedback.parameters())

    def clipped_apart(self) -> list[nn.Parameter]:
        """The feedback map is clipped on its own, so that its gradient cannot rescale the model's."""
        return self.preserved_parameters()


def window_offsets(batch: int, length: int, heads: int, window: float, device) -> Tensor:
    """Each position's offset c_j in its verification window, (batch, length): position j reads the stand-in that head
    c_j made at j - 1 - c_j, or for c_j = 0 the earlier pass's state at j - 1. Windows tile each sequence from position
    1, each going on past a position with probability `window` for up to `heads` stand-ins."""
    stand_ins = torch.distributions.Geometric(probs=torch.tensor(1 - window, device=device)).sample((batch, length))
    ends = torch.cumsum(1 + stand_ins.long().clamp(max=heads), 1)
    positions = torch.arange(length - 1, device=device).expand(batch, -1)
    starts = F.pad(ends[:, :-1], (1, 0)).gather(1, torch.searchsorted(ends, positions.contiguous(), right=True))
    return F.pad(positions - starts, (1, 0))


class MTPFeedbackTransformer(FeedbackTransformer, MTPTransformer):
    """The feedback Transformer (lorm.model.feedback) with MTP heads and the training of their stand-ins."""

    config_class = MTPFeedbackConfig

    def forward(self, input_ids: Tensor, *, passes: int | None = None,
                mtp: bool = False) -> Tensor | tuple[Tensor, MTPOutput]:
        """The logits of the last of `passes` passes, and with `mtp` the heads' output. Earlier passes run without
        gradients and feed back detached states. A training call of two or more passes trains the stand-in as
        `instance.stand_in` says."""
        passes = self.passes_of(passes)
        x, cos, sin = self.embed(input_ids)
        spec, training = self.config.instance, mtp and torch.is_grad_enabled() and passes > 1
        state, earlier, first = x, None, None
        for index in range(passes):
            last = index == passes - 1
            with torch.set_grad_enabled(torch.is_grad_enabled() and last):
                if index:
                    read = earlier
                    if training and last and spec.stand_in == "in_place":
                        read = self._substituted(earlier, self.mtp_chain(earlier, x, cos, sin), spec.share)
                    state = self.mix(x, read)
                top = self.blocks(state, cos, sin)
            first = top if index == 0 else first
            if not last:
                earlier = top.detach()
        if not mtp:
            return self.readout(top)
        loss = None
        if training and spec.stand_in == "state_alignment":
            loss = lambda chain: sum(alignment_loss(self.mtp.feedback(s.detach()), first, k)
                                     for k, s in enumerate(chain, 1)) / spec.heads
        return self.mtp_readout(top, x, cos, sin, loss)

    def _substituted(self, read: Tensor, chain: list[Tensor], share: float = 1.0) -> Tensor:
        """`read` with each position's entry replaced, by its offset in a verification window, by a stand-in, in a
        random `share` of the sequences. Index m of `read` is read by position m + 1; head c's stand-in for it was made
        at m - c."""
        batch, length = read.shape[:2]
        offsets = F.pad(window_offsets(batch, length, len(chain), self.config.instance.window, read.device)[:, 1:], (0, 1))
        if share < 1:
            offsets = offsets * (torch.rand(batch, 1, device=read.device) < share)
        for c, stand_in in enumerate(chain, 1):
            read = torch.where((offsets == c)[..., None], F.pad(stand_in, (0, 0, c, 0)), read)
        return read


MODELS["mtp"] = MTPTransformer
MODELS["mtp_feedback"] = MTPFeedbackTransformer
