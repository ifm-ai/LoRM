"""Decode attention over per-request caches for packed queries: row r owns counts[r] consecutive queries from
offsets[r], each reading only its own slot's keys up to its own position, so the cost follows each request's context
rather than the buffer (FlashDecoding), and rows of different widths share one launch.

Keys split into fixed chunks of CHUNK positions, so a row's arithmetic depends on its own context alone and not on the
other rows. A row whose keys fit one chunk writes its output directly; longer rows combine their chunks' partial
results.

q: (T, H, D) packed; caches: (slots, H_kv, L, D) with the new keys already written; rows: (R,) slots; before: (R,) keys
cached before a row's query 0, so its query i sees keys 0 .. before + i; width: the most queries any row owns. Packed
positions no row owns come out zero.
"""
import math

import torch
import triton
import triton.language as tl

CHUNK = 512


@triton.jit
def _partial(Q, K, V, Rows, Before, Offsets, Counts, Out, Lse, Part,
             q_t, q_h, k_s, k_h, k_l, o_t, o_h, scale,
             HKV: tl.constexpr, W: tl.constexpr, G: tl.constexpr, D: tl.constexpr, BLOCK_M: tl.constexpr,
             BLOCK_N: tl.constexpr, CHUNK: tl.constexpr, SPLITS: tl.constexpr):
    rh, split = tl.program_id(0), tl.program_id(1)
    r, h = rh // HKV, rh % HKV
    slot, before = tl.load(Rows + r), tl.load(Before + r)
    offset, count = tl.load(Offsets + r), tl.load(Counts + r)
    keys = tl.where(count > 0, before + count, 0)
    lo = split * CHUNK
    hi = tl.minimum(lo + CHUNK, keys)
    m = tl.arange(0, BLOCK_M)
    g, i = m // W, m % W
    live = (m < G * W) & (i < count)
    d = tl.arange(0, D)
    head = h * G + g
    at = (offset + i).to(tl.int64)
    q = tl.load(Q + at[:, None] * q_t + head[:, None] * q_h + d[None, :], mask=live[:, None], other=0.0)
    position = before + i
    top = tl.full([BLOCK_M], float("-inf"), tl.float32)
    total = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    base = slot.to(tl.int64) * k_s + h * k_h
    for start in range(lo, hi, BLOCK_N):
        n = start + tl.arange(0, BLOCK_N)
        inside = n < hi
        k = tl.load(K + base + n[:, None] * k_l + d[None, :], mask=inside[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale
        s = tl.where((n[None, :] <= position[:, None]) & inside[None, :], s, float("-inf"))
        new = tl.maximum(top, tl.max(s, 1))
        safe = tl.where(new == float("-inf"), 0.0, new)
        p = tl.exp(s - safe[:, None])
        alpha = tl.exp(top - safe)
        v = tl.load(V + base + n[:, None] * k_l + d[None, :], mask=inside[:, None], other=0.0)
        total = total * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        top = new
    if keys <= CHUNK:
        if split == 0:
            tl.store(Out + at[:, None] * o_t + head[:, None] * o_h + d[None, :],
                     (acc / total[:, None]).to(Out.dtype.element_ty), mask=live[:, None])
    elif lo < keys:
        slot_at = (rh * SPLITS + split) * BLOCK_M + m
        tl.store(Part + slot_at[:, None] * D + d[None, :], acc / tl.maximum(total, 1e-30)[:, None])
        tl.store(Lse + slot_at, tl.where(total > 0, top + tl.log(tl.maximum(total, 1e-30)), float("-inf")))


@triton.jit
def _combine(Before, Offsets, Counts, Out, Lse, Part, o_t, o_h,
             HKV: tl.constexpr, W: tl.constexpr, G: tl.constexpr, D: tl.constexpr, BLOCK_M: tl.constexpr,
             CHUNK: tl.constexpr, SPLITS: tl.constexpr):
    rh = tl.program_id(0)
    r, h = rh // HKV, rh % HKV
    offset, count = tl.load(Offsets + r), tl.load(Counts + r)
    keys = tl.where(count > 0, tl.load(Before + r) + count, 0)
    if keys > CHUNK:
        m = tl.arange(0, BLOCK_M)
        g, i = m // W, m % W
        live = (m < G * W) & (i < count)
        d = tl.arange(0, D)
        top = tl.full([BLOCK_M], float("-inf"), tl.float32)
        for split in range(SPLITS):
            used = split * CHUNK < keys
            lse = tl.load(Lse + (rh * SPLITS + split) * BLOCK_M + m, mask=used & live, other=float("-inf"))
            top = tl.maximum(top, lse)
        total = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, D], tl.float32)
        for split in range(SPLITS):
            used = split * CHUNK < keys
            slot_at = (rh * SPLITS + split) * BLOCK_M + m
            lse = tl.load(Lse + slot_at, mask=used & live, other=float("-inf"))
            weight = tl.where(lse == float("-inf"), 0.0, tl.exp(lse - top))
            total += weight
            acc += weight[:, None] * tl.load(Part + slot_at[:, None] * D + d[None, :], mask=(used & live)[:, None],
                                             other=0.0)
        head = h * G + g
        at = (offset + i).to(tl.int64)
        tl.store(Out + at[:, None] * o_t + head[:, None] * o_h + d[None, :],
                 (acc / total[:, None]).to(Out.dtype.element_ty), mask=live[:, None])


def decode_attention(q, k_cache, v_cache, rows, before, offsets, counts, width):
    T, H, D = q.shape
    R = rows.shape[0]
    HKV, L = k_cache.shape[1], k_cache.shape[2]
    G = H // HKV
    BLOCK_M = max(16, triton.next_power_of_2(G * width))
    splits = triton.cdiv(L, CHUNK)
    # The kernels index the per-row tensors with unit stride; a draft's before is a column of the starts table.
    rows, before, offsets, counts = (t.contiguous() for t in (rows, before, offsets, counts))
    out = torch.zeros_like(q)
    lse = torch.empty(R * HKV * splits * BLOCK_M, device=q.device, dtype=torch.float32)
    part = torch.empty(R * HKV * splits * BLOCK_M, D, device=q.device, dtype=torch.float32)
    _partial[(R * HKV, splits)](q, k_cache, v_cache, rows, before, offsets, counts, out, lse, part,
                                q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
                                out.stride(0), out.stride(1), 1 / math.sqrt(D), HKV=HKV, W=width, G=G, D=D,
                                BLOCK_M=BLOCK_M, BLOCK_N=64, CHUNK=CHUNK, SPLITS=splits, num_warps=4, num_stages=2)
    _combine[(R * HKV,)](before, offsets, counts, out, lse, part, out.stride(0), out.stride(1), HKV=HKV, W=width, G=G,
                         D=D, BLOCK_M=BLOCK_M, CHUNK=CHUNK, SPLITS=splits, num_warps=4)
    return out
