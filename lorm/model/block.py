"""Llama-style pre-norm Transformer block: RMSNorm, attention with adjacent-pair RoPE, SwiGLU."""

from contextlib import nullcontext
import math
from typing import Protocol

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .config import ModelConfig


def rotary_frequencies(positions: int | Tensor, head_dim: int, theta: float,
                       device: torch.device) -> tuple[Tensor, Tensor]:
    """RoPE cos and sin for positions 0..positions-1, or for the given position tensor. Computed in float32, so that
    model.to(bfloat16) cannot round them."""
    if isinstance(positions, int):
        positions = torch.arange(positions, device=device)
    powers = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim
    angles = positions.float()[:, None] * theta ** (-powers[None, :])
    return angles.cos(), angles.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    values = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    even, odd = values[..., 0::2], values[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2).to(dtype=x.dtype)


class KVCache(Protocol):
    """A layer's key-value store for incremental decoding."""

    def __call__(self, keys: Tensor, values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Store this call's rotated keys and values; return every key and value its queries may read and the
        boolean (queries, keys) mask of which."""


class Attention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads, self.n_kv_heads, self.head_dim = config.n_heads, config.n_kv_heads, config.head_dim
        self.flash = config.attention_backend == "flash"
        self.q_norm = nn.RMSNorm(config.head_dim, eps=config.norm_eps) if config.qk_norm else None
        self.k_norm = nn.RMSNorm(config.head_dim, eps=config.norm_eps) if config.qk_norm else None
        self.q_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.k_proj = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
        self.out_proj = nn.Linear(config.dim, config.dim, bias=False)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, cache: KVCache | None = None) -> Tensor:
        """Causal self-attention over x's positions, whose RoPE angles cos and sin give; with `cache`, the queries
        read the keys and values the cache returns instead."""
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        grouped = self.n_heads != self.n_kv_heads
        if cache is None:
            if self.flash and q.is_cuda and q.dtype not in (torch.float16, torch.bfloat16):
                raise ValueError("Flash Attention needs float16 or bfloat16 inputs; use autocast or attention_backend sdpa.")
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION) if self.flash and q.is_cuda else nullcontext():
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=grouped)
        else:
            k, v, mask = cache(k, v)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=grouped)
        return self.out_proj(y.transpose(1, 2).reshape(batch, length, -1))

    def forward_macs(self, shape: tuple[int, int, int]) -> int:
        """Projections plus QK^T and AV under the causal mask, which halves the scores."""
        batch, length, _ = shape
        projections = sum(p.weight.numel() for p in (self.q_proj, self.k_proj, self.v_proj, self.out_proj))
        return batch * (length * projections + self.n_heads * self.head_dim * length * length)


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.down_proj = nn.Linear(config.hidden_dim, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    def forward_macs(self, shape: tuple[int, ...]) -> int:
        return math.prod(shape[:-1]) * sum(p.weight.numel() for p in (self.gate_proj, self.up_proj, self.down_proj))


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.attention = Attention(config)
        self.norm2 = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.mlp = SwiGLU(config)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, cache: KVCache | None = None) -> Tensor:
        x = x + self.attention(self.norm1(x), cos, sin, cache)
        return x + self.mlp(self.norm2(x))

    def forward_macs(self, shape: tuple[int, int, int]) -> int:
        return self.attention.forward_macs(shape) + self.mlp.forward_macs(shape)

    def matrices(self) -> list[nn.Parameter]:
        attention, mlp = self.attention, self.mlp
        return [attention.q_proj.weight, attention.k_proj.weight, attention.v_proj.weight, attention.out_proj.weight,
                mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight]


class BlockStack(nn.Module):
    def __init__(self, config: ModelConfig, depth: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(Block(config) for _ in range(depth))

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x, cos, sin)
        return x

    def forward_macs(self, shape: tuple[int, int, int]) -> int:
        return sum(block.forward_macs(shape) for block in self.blocks)
