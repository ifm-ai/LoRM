"""Batched pipelined correction with continuous batching (Figure 7): engine.py's decoding loop over many requests at
once, greedy.

Every request keeps its own depth D (or exact decoding), pending window, draft-chain starts and committed tokens. The
bookkeeping that engine.Engine.generate runs in Python per call runs here as tensor ops over all requests, captured with
the forward pass in CUDA graphs, so a step costs one host round trip at any batch: the draft chain, then the
verification call and the commit. Requests live in slots of preallocated buffers; slot `slots` is a spare that padding
rows and rows that do not draft write to.

A step's forward pass runs over packed positions: each request computes only its own pending positions in
verification and its own chain span in the draft, and the step's position count, summed over requests, is rounded up
to a token tier. Attention reads each request's own cache up to its own position (decode_attention.py), so its cost
follows the request's context; attention="sdpa" masks the whole buffer instead, as engine.py does, for checks off GPU.

`bench`: a queue of max(300, 8 B) first turns of Spec-Bench and SPEED-Bench (prompts.py), shuffled once (seed 0) and
cycled, served by B slots, 256 new tokens at most, stopping at EOS. Prefill runs between steps and is timed apart;
steady state is the steps with all B slots busy, before the queue runs dry, and throughput counts the tokens after a
request's first.

    python -m works.feedback_shortcut.measure.batched RUN --stand-in state_alignment --depths exact,0,1,2,3 \
        --batches 1,2,4,8,16,32,64,128,256 --prompts data/prompts --out results/batched/RUN
"""

import argparse
import json
import math
import random
import time
from collections import deque
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torch.nn import functional as F

from lorm.eval.run import load_run
from lorm.model import rotary_frequencies
from lorm.model.block import apply_rope
from .. import model as _  # noqa: F401  (registration)
from .engine import STAND_INS
from .quality import PRECISIONS

INF = 1 << 30
WIDTHS = (1, 2, 4, 8, 12, 16)
SPANS = (4, 8, 12, 16, 24, 32)


def fit(x, tiers):
    return next(t for t in tiers if t >= x)


def tier(x):
    """Packed position counts: every count up to 16, then eight tiers per octave, in multiples of 8."""
    if x <= 16:
        return max(x, 1)
    e = (x - 1).bit_length() - 1
    return next(t for t in (-(-math.ceil(2 ** (e + j / 8)) // 8) * 8 for j in range(1, 9)) if t >= x)


def buckets(batch):
    """Graph row counts: powers of two below the batch, and the batch."""
    return sorted({1 << i for i in range(batch.bit_length()) if 1 << i < batch} | {batch})


class Packed(NamedTuple):
    """A step's packed positions: row r (slot rows[r]) owns counts[r] of them from offsets[r], at positions before[r]
    onwards; packed position t is row ri[t]'s j[t]-th, at slot[t] and pos[t]. Positions past the last row's are
    padding: ri = R, on the spare slot."""
    rows: torch.Tensor
    before: torch.Tensor
    offsets: torch.Tensor
    counts: torch.Tensor
    width: int
    ri: torch.Tensor
    j: torch.Tensor
    slot: torch.Tensor
    pos: torch.Tensor


class BatchedEngine:
    def __init__(self, model, slots: int, stand_in: str, *, length: int = 2112, attention: str = "triton",
                 passes: int = 4, new: int = 256, end: int = 2) -> None:
        self.model, self.slots, self.length, self.new, self.end = model, slots, length, new, end
        config, weight = model.config, model.head.weight
        device, dtype = weight.device, weight.dtype
        self.modules = model.heads()
        self.heads = K = len(self.modules)
        self.feedback = hasattr(model, "fusion")
        self.passes = passes if self.feedback else 1
        self.stand_in, self.attention = stand_in, attention
        if attention == "triton":
            from .decode_attention import decode_attention
            self.decode_attention = decode_attention
        self.cos, self.sin = rotary_frequencies(length, config.head_dim, config.rope_theta, device)
        self.keys = torch.arange(length, device=device)
        S, self.spare = slots + 1, slots
        cache = lambda: [torch.zeros(S, config.n_kv_heads, length, config.head_dim, device=device, dtype=dtype)  # noqa: E731
                         for _ in range(2)]
        states = lambda: torch.zeros(S, length, config.dim, device=device, dtype=dtype)  # noqa: E731
        ints = lambda *shape: torch.zeros(*shape, dtype=torch.long, device=device)  # noqa: E731
        self.kv = [cache() for _ in model.backbone.blocks]
        self.head_kv = [[cache() for _ in m.blocks.blocks] for m in self.modules]
        self.tops, self.reads, self.outs = states(), states(), [states() for _ in self.modules]
        self.standin = torch.zeros(S, length, dtype=torch.bool, device=device)
        # Device constants: an index assignment of a Python scalar copies it from the host, which a graph cannot.
        self.yes, self.no = torch.ones((), dtype=torch.bool, device=device), torch.zeros((), dtype=torch.bool, device=device)
        self.tape = ints(S, length)
        # Per slot: c (last position whose top is committed), the token committed at c + 1, the depth of the last
        # committed position, and the pending positions c + 1 .. c + n: token, depth, and from the last verification
        # the predicted next token and the depth it was computed at.
        M = WIDTHS[-1] + K + 1
        self.c, self.n, self.nexttok, self.lastdep = ints(S), ints(S), ints(S), ints(S)
        self.tok, self.dep, self.nxt, self.lev = ints(S, M), ints(S, M), ints(S, M), ints(S, M)
        self.starts = ints(S, K)
        self.need, self.win, self.ngen, self.limit = ints(S), ints(S), ints(S), ints(S)
        self.exact = torch.zeros(S, dtype=torch.bool, device=device)
        self.fresh = torch.zeros(S, dtype=torch.bool, device=device)  # drafted at admission, verifies first
        self.gen = ints(S, new + M + 1)  # the last column takes masked writes
        self.rows, self.host_rows, self.summary, self.host_summary = {}, {}, {}, {}
        self.graphs, self.pool, self.eager = {}, None, device.type != "cuda"

    def source(self, state):
        return self.model.fusion(F.rms_norm(state, (state.shape[-1],), eps=self.model.config.norm_eps))

    def pack(self, rows, before, counts, width, total):
        R = rows.shape[0]
        ends = counts.cumsum(0)
        offsets = ends - counts
        t = torch.arange(total, device=rows.device)
        ri = torch.searchsorted(ends, t, right=True)
        r = ri.clamp(max=R - 1)
        j = t - offsets[r]
        live = ri < R
        slot = torch.where(live, rows[r], self.spare)
        pos = torch.where(live, before[r] + j, 0).clamp(max=self.length - 1)
        return Packed(rows, before, offsets, counts, width, ri, j, slot, pos)

    def unpack(self, z, p):
        """(1, T, ...) packed to (R, width, ...) rows, positions a row does not own zero."""
        R = p.rows.shape[0]
        out = z.new_zeros(R + 1, p.width, *z.shape[2:])
        out[p.ri, p.j.clamp(max=p.width - 1)] = z[0]
        return out[:R]

    def attend(self, block, cache, x, p, prompt=False):
        """x (1, T, dim) at the packed positions p; with `prompt`, one row at positions 0 .. T - 1, which attend among
        themselves only."""
        a = block.attention
        T = x.shape[1]
        q = a.q_proj(x).view(1, T, a.n_heads, a.head_dim)
        k = a.k_proj(x).view(1, T, a.n_kv_heads, a.head_dim)
        v = a.v_proj(x).view(1, T, a.n_kv_heads, a.head_dim)
        if a.q_norm is not None:
            q, k = a.q_norm(q), a.k_norm(k)
        cos, sin = self.cos[p.pos][None].unsqueeze(2), self.sin[p.pos][None].unsqueeze(2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        gqa = a.n_heads != a.n_kv_heads
        cache[0][p.slot, :, p.pos] = k[0]
        cache[1][p.slot, :, p.pos] = v[0]
        if self.attention == "sdpa":
            positions = p.before[:, None] + torch.arange(p.width, device=x.device)
            mask = self.keys[None, None, :] <= positions[:, :, None]
            y = F.scaled_dot_product_attention(self.unpack(q, p).transpose(1, 2), cache[0][p.rows], cache[1][p.rows],
                                               attn_mask=mask[:, None], enable_gqa=gqa).transpose(1, 2)
            R = p.rows.shape[0]
            y = torch.where((p.ri < R)[:, None, None], y[p.ri.clamp(max=R - 1), p.j.clamp(max=p.width - 1)], 0)[None]
        elif prompt:
            y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True,
                                               enable_gqa=gqa).transpose(1, 2)
        else:
            y = self.decode_attention(q[0], cache[0], cache[1], p.rows, p.before, p.offsets, p.counts, p.width)[None]
        return a.out_proj(y.reshape(1, T, -1))

    def block(self, block, cache, x, p, prompt=False):
        x = x + self.attend(block, cache, block.norm1(x), p, prompt)
        return x + block.mlp(block.norm2(x))

    def clear_spare(self):
        s = slice(self.spare, self.spare + 1)
        for t in (self.c, self.n, self.starts, self.win, self.ngen):
            t[s] = 0
        self.exact[s] = False
        self.fresh[s] = False

    def draft(self, rows, width, total, prompt=False):
        """The draft chain for the rows that draft this step: drafts of last + 2 .. last + 1 + K at last = c + n, their
        stand-ins, each head's states from its start, and the drafts appended to the pending window. Rows that do not
        draft, and padding rows, run on the spare slot."""
        m, K, L = self.model, self.heads, self.length
        self.clear_spare()
        drafting = ~self.exact[rows] & ~self.fresh[rows] & (self.n[rows] + 1 + K <= self.win[rows])
        rows = torch.where(drafting, rows, self.spare)
        at_row = rows[:, None]
        c, n, starts = self.c[rows], self.n[rows], self.starts[rows]
        last = c + n
        has = n > 0
        previous = (n - 1).clamp(min=0)[:, None]
        head = torch.where(has, self.nxt[rows].gather(1, previous)[:, 0], self.nexttok[rows])
        head_depth = (torch.where(has, self.lev[rows].gather(1, previous)[:, 0], self.lastdep[rows]) + 1).clamp(max=INF)
        self.tape[rows, last + 1] = head
        self.standin[rows, last + 1] = self.no
        drafts, states = [], []
        for k, module in enumerate(self.modules, 1):
            begin = starts[:, k - 1]
            p = self.pack(rows, begin, (last - begin + 1) * drafting, width, total)
            state = (self.tops if k == 1 else self.outs[k - 2])[p.slot, p.pos]
            embedding = m.embedding(self.tape[p.slot, (p.pos + k).clamp(max=L - 1)])
            x = module.projection(torch.cat([module.state_norm(state), module.embedding_norm(embedding)], -1))[None]
            for b, cache in zip(module.blocks.blocks, self.head_kv[k - 1]):
                x = self.block(b, cache, x, p, prompt)
            self.outs[k - 1][p.slot, p.pos] = x[0]
            g = x[0][(p.offsets + last - begin).clamp(0, total - 1)]
            token = m.readout(g[:, None])[:, 0].argmax(-1)
            self.tape[rows, last + 1 + k] = token
            drafts.append(token)
            states.append(g)
        g = torch.stack(states, 1)
        at = last[:, None] + 2 + torch.arange(K, device=rows.device)
        self.reads[at_row, at] = (torch.zeros_like(g) if self.stand_in == "zeros" else
                                  m.mtp.feedback(g) if self.stand_in == "state_alignment" else g)
        self.standin[at_row, at] = self.yes
        cols = n[:, None] + torch.arange(1 + K, device=rows.device)
        self.tok[at_row, cols] = torch.cat([head[:, None], torch.stack(drafts, 1)], 1)
        self.dep[at_row, cols] = torch.cat([head_depth[:, None], torch.zeros_like(cols[:, 1:])], 1)
        self.n[rows] = n + 1 + K
        self.starts[rows] = (c[:, None] + 2 - torch.arange(1, K + 1, device=rows.device)).clamp(min=0)

    def verify(self, rows, width, total):
        """The verification call over each row's pending positions c + 1 .. c + n, then the commit: the longest prefix
        whose positions reach the row's depth and whose tokens follow the predictions before them, and the agreeing
        positions after it kept, one correction deeper. Writes the step's summary."""
        m, K = self.model, self.heads
        self.clear_spare()
        R, at_row = rows.shape[0], rows[:, None]
        i = torch.arange(width, device=rows.device)
        exact = self.exact[rows]
        c = self.c[rows]
        n = torch.where(exact, 1, self.n[rows])
        tok, dep = self.tok[rows][:, :width], self.dep[rows][:, :width]
        tok[:, 0] = torch.where(exact, self.nexttok[rows], tok[:, 0])
        dep[:, 0] = torch.where(exact, INF, dep[:, 0])
        valid = i < n[:, None]
        pos = c[:, None] + 1 + i
        self.tape[at_row, pos] = torch.where(valid, tok, self.tape[at_row, pos])
        p = self.pack(rows, c + 1, n, width, total)
        read = torch.where(self.standin[p.slot, p.pos][:, None], self.reads[p.slot, p.pos],
                           self.tops[p.slot, (p.pos - 1).clamp(min=0)])
        # Positions past the pending ones are drafted again, with their stand-in flags, before they are verified.
        self.standin[at_row, pos] = self.standin[at_row, pos] & ~valid
        x = m.embedding(self.tape[p.slot, p.pos])
        if self.feedback:  # a Transformer's verification reads nothing
            x = x + self.source(read) * (p.pos > 0)[:, None]
        x = x[None]
        for b, cache in zip(m.backbone.blocks, self.kv):
            x = self.block(b, cache, x, p)
        self.tops[p.slot, p.pos] = x[0]
        pred = self.unpack(m.readout(x).argmax(-1)[..., None], p)[..., 0]
        chain = tok == torch.cat([tok[:, :1], pred[:, :-1]], 1)
        done = (valid & (dep >= self.need[rows][:, None]) & chain).long().cumprod(1).sum(1)
        kept = ((i < done[:, None]) | (valid & chain)).long().cumprod(1).sum(1) - done
        ngen = self.ngen[rows]
        live = i < done[:, None]
        self.gen[at_row, torch.where(live, ngen[:, None] + i, self.gen.shape[1] - 1)] = pred
        hit = (live & (pred == self.end)).any(1)
        self.ngen[rows] = ngen = ngen + done
        final = (done - 1).clamp(min=0)[:, None]
        self.nexttok[rows] = pred.gather(1, final)[:, 0]
        self.lastdep[rows] = dep.gather(1, final)[:, 0]
        self.c[rows] = c = c + done
        src = (done[:, None] + i).clamp(max=width - 1)
        cols = i.expand(R, width)
        self.tok[at_row, cols] = tok.gather(1, src)
        self.lev[at_row, cols] = dep.gather(1, src)
        self.nxt[at_row, cols] = pred.gather(1, src)
        self.dep[at_row, cols] = (dep.gather(1, (src - 1).clamp(min=0)) + 1).clamp(max=INF)
        self.n[rows] = kept
        self.fresh[rows] = self.no
        drafting = ~exact & (kept + 1 + K <= self.win[rows])
        self.summary[R].copy_(torch.stack([
            done, torch.where(exact, 1, kept + drafting * (1 + K)), drafting.long(),
            c + kept + 1 - self.starts[rows][:, K - 1], (hit | (ngen >= self.limit[rows])).long(), n], 1))

    # Prefill and admission, eager and per request.

    def prefill(self, slot, prompt):
        """`passes` parallel passes over the prompt (one for a Transformer); returns the first new token."""
        m, device = self.model, self.tops.device
        n = len(prompt)
        rows, zero = torch.tensor([slot], device=device), torch.zeros(1, dtype=torch.long, device=device)
        p = self.pack(rows, zero, torch.tensor([n], device=device), n, n)
        self.standin[slot] = False
        self.tape[slot, :n] = torch.tensor(prompt, device=device)
        embedding = m.embedding(self.tape[slot, :n])[None]
        for k in range(self.passes):
            x = embedding
            if k:
                x = x + self.source(self.tops[slot, :n].roll(1, 0))[None] * (p.pos > 0)[None, :, None]
            for b, cache in zip(m.backbone.blocks, self.kv):
                x = self.block(b, cache, x, p, True)
            self.tops[slot, :n] = x[0]
        return int(m.readout(x[:, -1:])[0].argmax(-1))

    def admit(self, slot, prompt, depth, new):
        """Prefill, then (pipelined correction) the heads' first pass over the whole prompt, which drafts the first
        window. Returns the first token."""
        first = self.prefill(slot, prompt)
        n, exact = len(prompt), depth == "exact"
        d = 0 if exact else int(depth)
        values = dict(c=n - 1, n=0, nexttok=first, lastdep=INF, need=d, win=1 if exact else (self.heads + 1) * (d + 1),
                      ngen=1, limit=new)
        for key, value in values.items():
            getattr(self, key)[slot] = value
        self.starts[slot] = 0
        self.exact[slot], self.fresh[slot] = exact, False
        self.gen[slot, 0] = first
        if not exact and first != self.end and new > 1:
            self.draft(torch.tensor([slot], device=self.tops.device), n, n, prompt=True)
            self.fresh[slot] = True
        return first

    # Graphs, captured on first use, on spare rows, before the step that needs them uploads its rows.

    def buffers(self, batch):
        device = self.tops.device
        pin = device.type == "cuda"
        for R in buckets(batch):
            if R not in self.rows:
                self.rows[R] = torch.full((R,), self.spare, dtype=torch.long, device=device)
                self.host_rows[R] = torch.full((R,), self.spare, dtype=torch.long, pin_memory=pin)
                self.summary[R] = torch.zeros(R, 6, dtype=torch.long, device=device)
                self.host_summary[R] = torch.zeros(R, 6, dtype=torch.long, pin_memory=pin)

    def capture(self, key):
        if self.eager or key in self.graphs:
            return
        kind, R, width, total = key
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        step = getattr(self, kind)
        self.rows[R].fill_(self.spare)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            step(self.rows[R], width, total)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            step(self.rows[R], width, total)
        torch.cuda.synchronize()
        self.graphs[key] = graph

    def replay(self, key):
        kind, R, width, total = key
        if self.eager:
            getattr(self, kind)(self.rows[R], width, total)
        else:
            self.graphs[key].replay()

    def synchronize(self):
        if self.tops.device.type == "cuda":
            torch.cuda.synchronize()

    @torch.no_grad()
    def serve(self, requests: list[list[int]], depths: list, batch: int, new: int | None = None):
        """Continuous batching of `requests` (prompt ids) over `batch` slots, each at its depth ("exact" or an integer).
        Prefill runs between steps and is timed apart; a step is timed from its row upload to its summary on the host.
        Returns per-request outputs (cut at the end token) and per-step records."""
        new = new or self.new
        K = self.heads
        window = max(1 if d == "exact" else (K + 1) * (int(d) + 1) for d in depths)
        # A step takes the widest pending window and chain span any request can reach (a span is at most
        # done + kept + K - 1 after the commit, so window + K - 1).
        verify_width, draft_width = fit(window, WIDTHS), fit(window + K - 1, SPANS)
        self.buffers(batch)
        queue, free, active = deque(range(len(requests))), list(range(batch))[::-1], []
        plan = np.zeros((self.slots, 3), dtype=np.int64)  # per slot: next width, drafts next, next span
        owner, calls, prefill_s = {}, {}, {}
        steps, results = [], [None] * len(requests)
        self.synchronize()
        while queue or active:
            while queue and free:
                r, s = queue.popleft(), free.pop()
                start = time.perf_counter()
                first = self.admit(s, requests[r], depths[r], new)
                self.synchronize()
                prefill_s[r] = time.perf_counter() - start
                if first == self.end or new <= 1:
                    results[r] = dict(output=[] if first == self.end else [first], calls=[], prefill_s=prefill_s[r])
                    free.append(s)
                    continue
                owner[s], calls[r] = r, []
                plan[s] = (1 if depths[r] == "exact" else 1 + K, 0, 0)
                active.append(s)
            R = len(active)
            bucket = next(b for b in buckets(batch) if b >= R)
            p = plan[active]
            drafting = p[:, 1] > 0
            verify = ("verify", bucket, verify_width, tier(int(p[:, 0].sum())))
            draft = ("draft", bucket, draft_width, tier(int(p[drafting, 2].sum()))) if drafting.any() else None
            for key in (draft, verify):
                if key:
                    self.capture(key)
            rows = self.host_rows[bucket]
            rows.fill_(self.spare)
            rows[:R] = torch.tensor(active)
            start = time.perf_counter()
            self.rows[bucket].copy_(rows, non_blocking=True)
            if draft:
                self.replay(draft)
            self.replay(verify)
            out = self.host_summary[bucket]
            out.copy_(self.summary[bucket], non_blocking=True)
            self.synchronize()
            seconds = time.perf_counter() - start
            out = out[:R].numpy().copy()
            steps.append(dict(t=seconds, rows=R, tokens=verify[3]))
            finished = []
            for j, s in enumerate(active):
                done, n_next, drafts, next_span, fin, w = out[j]
                calls[owner[s]].append((len(steps) - 1, int(w), int(done)))
                plan[s] = (n_next, drafts, next_span)
                if fin:
                    finished.append(s)
            for s in finished:
                r = owner.pop(s)
                output = self.gen[s, :int(self.ngen[s])].tolist()[:new]
                if self.end in output:
                    output = output[:output.index(self.end)]
                results[r] = dict(output=output, calls=calls.pop(r), prefill_s=prefill_s[r])
                active.remove(s)
                free.append(s)
        for r in results:
            # Tokens after the first that count: each call's commits up to the output's length.
            left, r["kept"] = len(r["output"]) - 1, []
            for _, _, done in r["calls"]:
                r["kept"].append(max(0, min(done, left)))
                left -= r["kept"][-1]
        return results, steps


def metrics(results, steps, batch: int) -> dict:
    t = np.array([s["t"] for s in steps])
    full = np.array([s["rows"] for s in steps]) == batch
    kept = np.zeros(len(steps))
    for r in results:
        for (step, *_), k in zip(r["calls"], r["kept"]):
            kept[step] += k
    # Steady state ends at the first step that runs below capacity.
    end = int(np.argmin(full)) if not full.all() else len(steps)
    calls = [c for r in results for c in r["calls"]]
    commits = sum(c[2] for c in calls)
    return dict(batch=batch, requests=len(results), steps=len(steps), steady_steps=end,
                throughput=float(kept[:end].sum() / t[:end].sum()) if end else None,
                step_ms=float(1e3 * t[:end].mean()) if end else None,
                tokens_per_call=commits / len(calls), rho=sum(c[1] for c in calls) / commits,
                prefill_s=sum(r["prefill_s"] for r in results))


@torch.no_grad()
def main():
    from .prompts import first_turns
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--stand-in", default="mtp_state", choices=STAND_INS)
    parser.add_argument("--depths", default="exact,0,1,2,3")
    parser.add_argument("--batches", default="1,2,4,8,16,32,64,128,256")
    parser.add_argument("--prompts", type=Path, required=True, help="the prompt sets of prompts.py")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--new", type=int, default=256)
    parser.add_argument("--per-slot", type=int, default=8)
    parser.add_argument("--least", type=int, default=300)
    parser.add_argument("--precision", default="float16", choices=PRECISIONS)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    prompts, end = first_turns(args.prompts)
    order = list(range(len(prompts)))
    random.Random(0).shuffle(order)
    model = load_run(args.run, torch.device("cuda")).to(PRECISIONS[args.precision])
    for batch in map(int, args.batches.split(",")):
        queue = [order[i % len(order)] for i in range(max(args.least, args.per_slot * batch))]
        engine = BatchedEngine(model, batch, args.stand_in, new=args.new, end=end)
        for depth in args.depths.split(","):
            path = args.out / f"B{batch}-{depth}.json"
            if path.exists():
                continue
            results, steps = engine.serve([prompts[i] for i in queue], [depth] * len(queue), batch, args.new)
            m = metrics(results, steps, batch)
            path.write_text(json.dumps(dict(run=args.run.name, stand_in=args.stand_in, depth=depth,
                                            precision=args.precision, metrics=m)))
            print(f"B={batch} {depth}: " + " ".join(f"{k}={v:.4g}" for k, v in m.items() if isinstance(v, float)),
                  flush=True)
        del engine
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
