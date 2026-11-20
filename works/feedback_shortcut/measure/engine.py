"""Speculative decoding with pipelined correction at batch size 1, on static buffers that CUDA graphs replay.

Each call runs the pending positions after the last committed one in one forward pass: positions kept from the last
call, which re-run on the top states that call computed, then the next position and K drafts of the MTP chain, which
read stand-ins. A position's depth counts the calls since a stand-in entered the states it reads. A call commits the
longest prefix whose tokens agree with the predictions before them and whose depths reach `depth`; the agreeing
positions after it stay pending, one correction deeper, and the rest is dropped. Depth 0 verifies without corrections;
depth None commits only positions that read no stand-in, which is exact decoding, one position per call. A Transformer
reads no stand-ins, so at every depth it runs ordinary speculative decoding.

Stand-ins (STAND_INS): zeros, the MTP state g (which in-place substitution trains to be read) and state alignment's map
P(g).

The prompt is prefilled token by token, which is exact decoding, or in `prefill_passes` parallel passes of the training
iteration (one for a Transformer). Decoding is greedy, or at a temperature samples its drafts and verifies them by
speculative sampling (Leviathan et al., 2023): a position's distribution p accepts the token after it, drawn from r,
with probability min(1, p/r), and otherwise replaces it with a draw from max(0, p - r). A re-run verifies the token
again, with r the distribution that accepted it last, so a committed token follows its distribution at its commit
depth. With `target`, decoding is teacher-forced along a given text: drafts are kept while they equal it, and each of
its tokens is scored by the distribution of the call that commits it, at the depth it commits with.

Keys, values, top states, head states, stand-ins and tokens live in device buffers indexed by position, and each step
reads its positions from a small control tensor, so on CUDA a call or a draft chain is one graph replay at a padded
width. Padding positions go to the last buffer slot, which no real position reads.
"""

import math
import time
from typing import NamedTuple

import torch
from torch.nn import functional as F

from lorm.model import rotary_frequencies

EXACT = math.inf
STAND_INS = ("zeros", "mtp_state", "state_alignment")
# Graph widths; a step uses the narrowest that holds its positions, and a wider draft chain runs eagerly.
WIDTHS = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64)


class Generation(NamedTuple):
    tokens: list[int]  # the new tokens, up to `new` and before the end token
    calls: int  # verification calls after prefill
    widths: list[int]  # positions each call verified
    commits: list[int]  # tokens each call committed
    seconds: float  # decoding after prefill
    nll: list[float] | None  # with `target`: the NLL (nats) of each target token at its commit


class PositionCache:
    """One layer's keys and values at absolute positions: a step writes its positions and its queries read every
    position up to their own. Later slots, left from dropped drafts, stay masked."""

    def __init__(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        self.keys, self.values, self.positions = keys, values, None
        self.slots = torch.arange(keys.shape[2], device=keys.device)

    def __call__(self, keys, values):
        self.keys.index_copy_(2, self.positions, keys)
        self.values.index_copy_(2, self.positions, values)
        return self.keys, self.values, self.positions[:, None] >= self.slots[None]


class Engine:
    def __init__(self, model, stand_in: str, depth: int | None, length: int, *, temperature: float = 0.0,
                 prefill_passes: int | None = None, score: bool = False, graphs: bool | None = None) -> None:
        """`length` bounds prompt, new tokens and the window past them. `score` makes verification return
        distributions (temperature 1) for `generate(..., target=...)`, while drafts stay greedy."""
        if stand_in not in STAND_INS or (stand_in == "state_alignment" and model.mtp.feedback is None):
            raise ValueError(f"stand_in must be one of {STAND_INS}, state_alignment only for a model with its map.")
        config, parameter = model.config, model.head.weight
        device, dtype = parameter.device, parameter.dtype
        self.model, self.stand_in, self.temperature, self.score = model, stand_in, temperature, score
        self.prefill_passes = prefill_passes
        self.heads = config.instance.heads
        self.feedback = hasattr(model, "fusion")
        self.need = EXACT if depth is None else depth
        # Fresh drafts read stand-ins only in a feedback Transformer.
        self.fresh = 0 if self.feedback else EXACT
        # Room for the pending positions of `depth` corrections and a new window; exact decoding drafts nothing.
        self.width = 1 if depth is None else (self.heads + 1) * (depth + 1)
        self.cuda = device.type == "cuda"
        self.graphs = self.cuda if graphs is None else graphs
        self.scratch = length - 1
        kv = lambda: torch.zeros(1, config.n_kv_heads, length, config.head_dim, dtype=dtype, device=device)
        states = lambda: torch.zeros(length, config.dim, dtype=dtype, device=device)
        self.caches = [PositionCache(kv(), kv()) for _ in model.backbone.blocks]
        self.head_caches = [PositionCache(kv(), kv()) for _ in range(self.heads)]
        self.cos, self.sin = rotary_frequencies(length, config.head_dim, config.rope_theta, device)
        self.tape = torch.zeros(length, dtype=torch.long, device=device)  # the token at each position
        self.tops, self.stand_ins = states(), states()
        self.head_states = [states() for _ in range(self.heads)]
        # Whether a position reads its stand-in rather than the top state before it.
        self.reads_stand_in = torch.zeros(length, dtype=torch.bool, device=device)
        self.draft_dists = torch.zeros(self.heads, config.vocab_size, device=device)
        # c (the last committed position), the token at c + 1, the positions in the call, the drafting position q,
        # the token at q + 1 and each head's first position.
        self.control = torch.zeros(5 + self.heads, dtype=torch.long, device=device)
        self.captured = {}

    def _set(self, values: list[int]) -> None:
        self.control.copy_(torch.tensor(values + [0] * (len(self.control) - len(values))))

    def _run(self, blocks, caches, x, positions):
        cos, sin = self.cos[positions], self.sin[positions]
        for block, cache in zip(blocks, caches):
            cache.positions = positions
            x = block(x, cos, sin, cache)
        return x

    def _fed(self, read):
        model = self.model
        return model.fusion(F.rms_norm(read, (read.shape[-1],), eps=model.config.norm_eps))

    def _verified(self, logits):
        if self.temperature or self.score:
            return torch.softmax(logits.float() / (self.temperature or 1.0), -1)
        return logits.argmax(-1)

    def _call(self, width: int):
        """Positions c + 1 .. c + count, each reading its stand-in or the latest top state before it."""
        model, c, count = self.model, self.control[0], self.control[2]
        offsets = torch.arange(width, device=c.device)
        positions = torch.where(offsets < count, c + 1 + offsets, self.scratch)
        self.tape.index_copy_(0, (c + 1).view(1), self.control[1:2])
        x = model.embedding(self.tape[positions])
        if self.feedback:
            read = torch.where(self.reads_stand_in[positions, None], self.stand_ins[positions],
                               self.tops[(positions - 1).clamp(min=0)])
            x = x + self._fed(read) * (positions > 0)[:, None]
            self.reads_stand_in.index_fill_(0, positions, False)
        top = self._run(model.backbone.blocks, self.caches, x[None], positions)[0]
        self.tops[positions] = top
        return self._verified(model.readout(top))

    def _token(self, logits, k: int):
        """Head k's draft: greedy, or a Gumbel-max draw whose distribution is kept for speculative sampling."""
        if not self.temperature:
            return logits.argmax(-1)
        dist = torch.softmax(logits.float() / self.temperature, -1)
        self.draft_dists[k - 1] = dist[0]
        return (dist.log() - torch.empty_like(dist).exponential_().log()).argmax(-1)

    def _draft(self, width: int):
        """The chain at position q: drafts of tokens q + 2 .. q + 1 + K and their stand-ins. Head k re-runs from its
        first position, whose inputs earlier calls may have changed."""
        # Index tensors and in-place index ops throughout: host scalars and values would break graph capture.
        model, q = self.model, self.control[3:4]
        self.tape.index_copy_(0, q + 1, self.control[4:5])
        self.reads_stand_in.index_fill_(0, q + 1, False)
        drafts, states = [], []
        for k, module in enumerate(model.heads(), 1):
            positions = self.control[4 + k] + torch.arange(width, device=q.device)
            positions = torch.where(positions <= q, positions, self.scratch)
            state = (self.tops if k == 1 else self.head_states[k - 2])[positions]
            embedding = model.embedding(self.tape[(positions + k).clamp(max=self.scratch)])
            fused = module.projection(torch.cat([module.state_norm(state), module.embedding_norm(embedding)], -1))
            out = self._run(module.blocks.blocks, self.head_caches[k - 1:k], fused[None], positions)[0]
            self.head_states[k - 1][positions] = out
            g = self.head_states[k - 1][q]
            token = self._token(model.readout(g), k)
            self.tape.index_copy_(0, q + 1 + k, token)
            drafts.append(token)
            states.append(g)
        if self.feedback:
            # All K stand-ins at once: one matrix product per map weight, as the paper's measurements ran it.
            g, at = torch.cat(states), q + 2 + torch.arange(self.heads, device=q.device)
            self.stand_ins.index_copy_(0, at, torch.zeros_like(g) if self.stand_in == "zeros" else
                                       model.mtp.feedback(g) if self.stand_in == "state_alignment" else g)
            self.reads_stand_in.index_fill_(0, at, True)
        return torch.cat(drafts)

    def _step(self, kind: str, needed: int, most: int):
        """Run a call or a draft chain over `needed` positions: as a graph of the narrowest width that holds them, if
        it is at most `most`, captured on first use; otherwise eagerly."""
        step = getattr(self, f"_{kind}")
        width = min((w for w in (*WIDTHS, most) if needed <= w <= most), default=None)
        if not self.graphs or width is None:
            return step(needed)
        if (kind, width) not in self.captured:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                result = step(width).clone()  # this step itself, which also warms up before capture
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = step(width)
            self.captured[kind, width] = graph, output
            return result
        graph, output = self.captured[kind, width]
        graph.replay()
        return output

    @torch.no_grad()
    def capture(self) -> None:
        """Capture every call width up to the window and every draft width a window can need, so that timed decoding
        replays only. The dummy steps write positions that a generation rewrites before it reads them."""
        if not self.graphs:
            return
        for width in (w for w in (*WIDTHS, self.width) if w <= self.width):
            self._set([0, 0, width])
            self._step("call", width, self.width)
        if self.width > 1:
            most = min((w for w in WIDTHS if w >= self.width + self.heads - 1), default=WIDTHS[-1])
            for width in (w for w in WIDTHS if w <= most):
                self._set([0, 0, 0, width - 1, 0])
                self._step("draft", width, WIDTHS[-1])
        torch.cuda.synchronize()

    def _pick(self, output, record):
        if not self.temperature:
            return int(output.argmax() if self.score else output)
        if record is not None:
            record.append(output.clone())
        return int(torch.multinomial(output, 1))

    def prefill(self, prompt: list[int]):
        """Run the prompt; returns the verification output after its last token."""
        self.reads_stand_in.zero_()
        if self.prefill_passes is None:
            for t, token in enumerate(prompt):
                self._set([t - 1, token, 1])
                output = self._step("call", 1, 1)
            return output[-1]
        model, n = self.model, len(prompt)
        positions = torch.arange(n, device=self.tape.device)
        self.tape[:n] = torch.tensor(prompt, device=self.tape.device)
        embedding = model.embedding(self.tape[:n])
        for k in range(self.prefill_passes if self.feedback else 1):
            # Pass k reads pass k - 1's top state of the position before; position 0 reads nothing.
            x = embedding + self._fed(self.tops[:n].roll(1, 0)) * (positions > 0)[:, None] if k else embedding
            self.tops[:n] = self._run(model.backbone.blocks, self.caches, x[None], positions)[0]
        return self._verified(model.readout(self.tops[n - 1:n]))[-1]

    @torch.no_grad()
    def generate(self, prompt: list[int], new: int, *, end: int | None = None, target: list[int] | None = None,
                 record: list | None = None) -> Generation:
        """The continuation of `prompt` by up to `new` tokens, stopping at `end`; `record`, if given, collects the
        distribution of each generated token when sampling. With `target` (an engine built with `score`), decode
        along `target` instead and score each of its tokens."""
        forced = target is not None
        if forced and not self.score:
            raise ValueError("Teacher-forced decoding needs an engine built with score=True.")
        text = list(prompt) + list(target) if forced else None
        new = len(target) if forced else new
        output = self.prefill(prompt)
        tokens = list(prompt) + [target[0] if forced else self._pick(output, record)]
        nll = [-float(output[target[0]].clamp(min=1e-30).log())] if forced else None
        starts = [0] * self.heads  # each head's first position whose state is not final
        # Pending positions after the last committed one c: token, depth, and from the last call the prediction after
        # it (next), the depth it ran at (level) and, when sampling, its distribution (dist) and the distribution its
        # token was last drawn or accepted from (r).
        pending, last_depth, widths, commits = [], EXACT, [], []
        stop = tokens[-1] == end
        if self.cuda:
            torch.cuda.synchronize()
        started = time.perf_counter()
        while len(tokens) < len(prompt) + new and not stop:
            c = len(tokens) - 2
            previous = pending[-1] if pending else dict(level=last_depth, next=tokens[c + 1], dist=None)
            head = dict(token=previous["next"], depth=previous["level"] + 1, r=previous["dist"])
            if len(pending) + 1 + self.heads <= self.width:
                q = c + len(pending)
                self._set([c, tokens[c + 1], 0, q, head["token"], *starts])
                # The first chain covers the whole prompt and runs eagerly at its own width, as the paper's runs did.
                drafts = self._step("draft", q + 1 - starts[-1], 0 if not widths else WIDTHS[-1]).tolist()
                dists = self.draft_dists.clone() if self.temperature else [None] * self.heads
                starts = [max(c + 2 - k, 0) for k in range(1, self.heads + 1)]
                pending += [head] + [dict(token=d, depth=self.fresh, r=r) for d, r in zip(drafts, dists)]
            elif not pending:
                pending = [head]
            self._set([c, tokens[c + 1], len(pending)])
            output = self._step("call", len(pending), self.width)[:len(pending)]
            widths.append(len(pending))
            if forced:
                # Entry i predicts position c + 2 + i; past the target it is a placeholder that nothing commits.
                beyond = [c + 2 + i >= len(text) for i in range(len(pending))]
                predicted = [0 if b else text[c + 2 + i] for i, b in enumerate(beyond)]
                rows = torch.arange(len(pending), device=output.device)
                scores = (-output[rows, torch.tensor(predicted, device=output.device)].clamp(min=1e-30).log()).tolist()
                dists = [None] * len(pending)
            elif self.temperature:
                dists = output.clone()
                predicted = speculative(dists, [p["token"] for p in pending[1:]], [p["r"] for p in pending[1:]])
            else:
                dists, predicted = [None] * len(pending), output.tolist()
            for p, prediction, dist in zip(pending, predicted, dists):
                p.update(next=prediction, level=p["depth"], dist=dist)
            done = 0
            while done < len(pending) and pending[done]["depth"] >= self.need and not (forced and beyond[done]) and (
                    done == 0 or pending[done]["token"] == pending[done - 1]["next"]):
                done += 1
            tokens += [p["next"] for p in pending[:done]]
            commits.append(done)
            stop = end is not None and end in tokens[len(tokens) - done:]
            if forced:
                nll += scores[:done]
            if record is not None:
                record += [p["dist"] for p in pending[:done]]
            last_depth = pending[done - 1]["depth"]
            kept = 0
            while done + kept < len(pending) and pending[done + kept]["token"] == pending[done + kept - 1]["next"]:
                kept += 1
            before, pending = pending[done - 1:done + kept], pending[done:done + kept]
            for b, p in zip(before, pending):
                p.update(depth=b["level"] + 1, r=b["dist"])
        if self.cuda:
            torch.cuda.synchronize()
        out = tokens[len(prompt):len(prompt) + new]
        if end in out:
            out = out[:out.index(end)]
        return Generation(out, len(widths), widths, commits, time.perf_counter() - started,
                          nll[:new] if forced else None)

    @torch.no_grad()
    def exact_distributions(self, prompt: list[int], generated: list[int]) -> torch.Tensor:
        """Exact decoding's distribution of each generated token given the ones before it (sampling engines)."""
        dists = [self.prefill(prompt).clone()]
        sequence = list(prompt) + list(generated)
        for c in range(len(prompt), len(sequence) - 1):
            self._set([c - 1, sequence[c], 1])
            dists.append(self._step("call", 1, 1)[-1].clone())
        return torch.stack(dists)


def speculative(dists: torch.Tensor, tokens: list[int], drawn_from: list[torch.Tensor]) -> list[int]:
    """Each position's successor: position i's distribution p accepts tokens[i], drawn from drawn_from[i], with
    probability min(1, p/r), else replaces it with a draw from max(0, p - r); the last position draws from p. Either
    way the successor follows p."""
    successors = torch.multinomial(dists, 1)[:, 0]
    if tokens:
        p, r = dists[:-1], torch.stack(drawn_from)
        x = torch.tensor(tokens, device=p.device)[:, None]
        accept = torch.rand(len(tokens), device=p.device) * r.gather(1, x)[:, 0] < p.gather(1, x)[:, 0]
        residual = (p - r).clamp(min=0)
        # A zero residual means p = r, where every token is accepted.
        redraw = torch.multinomial(torch.where(residual.sum(-1, keepdim=True) > 0, residual, p), 1)[:, 0]
        successors[:-1] = torch.where(accept, x[:, 0], redraw)
    return successors.tolist()


def kl(dists: torch.Tensor, exact: torch.Tensor) -> float:
    """Mean KL(dists || exact) per token."""
    log_q, log_p = dists.clamp(min=1e-30).log(), exact.clamp(min=1e-30).log()
    return float((dists * (log_q - log_p)).sum(-1).mean())
