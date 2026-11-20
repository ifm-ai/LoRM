"""Quality on the raw Paloma subsets: exact decoding BPB, the one-call error of each stand-in and each head's acceptance
(Table 1 and the acceptance per head).

Exact BPB scores every document on its own, as Paloma does: BOS + tokens + EOS in chunks of at most 2,048 tokens that
overlap by one, with `passes` parallel passes of a feedback model.

The one-call error is teacher-forced on random windows of each subset. Positions run one at a time and read the stored
state of the position before, so the states are those of token-by-token decoding. A window starts at a position that
reads its real state, whose logits verify draft 1; a draft passes with the speculative-sampling probability
sum_v min(draft(v), verifier(v)), and while drafts pass, the next position runs in the same call and reads the
matching head's stand-in. A window ends at the first rejection or after K drafts. In the corrected context, each ended
window's stand-in reads are replaced by the real states before any later position runs, and drafts come from the
real states, so only the current window reads stand-ins: its cost is the one-call error. In the kept context every
position keeps what it read (the uncorrected error). A stand-in's cost is its mean NLL minus exact decoding's on the
same positions, converted to BPB per subset and averaged over subsets. Acceptance is each head's mean
sum_v min(draft, verifier) on exact decoding, head k scored where heads 1 to k - 1 passed.

Bytes are UTF-8: the space Llama's tokenizer prepends to each document's first token is not in the text, so it is
left out. Float16 by default, as decoding is served; log-probabilities in float32.

    python -m works.feedback_shortcut.measure.quality RUN [RUN ...] --paloma data/paloma/raw --out results/quality
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from lorm.data import token_byte_lengths
from lorm.eval.run import load_run
from .. import model as _  # noqa: F401  (registration)
from .engine import STAND_INS

BOS, EOS, CHUNK, BUDGET = 1, 2, 2048, 16 * 2048
WINDOWS, LENGTH, BURN_IN, BATCH = 64, 384, 32, 32  # positions before BURN_IN only let windows reach their steady state
TOKENIZER = ("meta-llama/Llama-2-7b-hf", "01c7f73d771dfac7d292323805ebc428287df4f9")
PRECISIONS = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def utf8_bytes(lengths: np.ndarray, targets: np.ndarray, previous: np.ndarray) -> int:
    """UTF-8 bytes of `targets`, less the prepended space of each document's first token (the one after BOS)."""
    return int(lengths[targets].sum()) - int(((previous == BOS) & (lengths[targets] > 0)).sum())


def read_shard(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype="<u2", offset=1024).astype(np.int64)


def windows(tokens: np.ndarray, count: int = WINDOWS, length: int = LENGTH) -> np.ndarray:
    """`count` random windows of length + 1 tokens, seed 0."""
    starts = np.random.default_rng(0).integers(0, len(tokens) - length - 2, count)
    return np.stack([tokens[s:s + length + 1] for s in starts])


def forward(model, ids, passes):
    return model(ids, passes=passes) if hasattr(model, "fusion") else model(ids)


@torch.no_grad()
def document_bpb(model, tokens: np.ndarray, lengths: torch.Tensor, passes: int) -> dict:
    """Exact BPB with each document scored alone."""
    device = lengths.device
    starts = np.flatnonzero(tokens == BOS)
    chunks = sorted(((tokens[c:min(c + CHUNK, b)], c == a) for a, b in zip(starts, np.append(starts[1:], len(tokens)))
                     for c in range(a, b - 1, CHUNK - 1)), key=lambda item: len(item[0]))
    nll = nbytes = 0.0
    batch = []
    for item in chunks + [None]:
        if batch and (item is None or (len(batch) + 1) * len(item[0]) > BUDGET):
            width = max(len(t) for t, _ in batch)
            ids = torch.full((len(batch), width), EOS, dtype=torch.long)
            labels = torch.full((len(batch), width), -100, dtype=torch.long)
            for i, (t, _) in enumerate(batch):
                ids[i, :len(t)] = labels[i, :len(t)] = torch.from_numpy(t)
            ids, target = ids.to(device), labels[:, 1:].to(device)
            loss = F.cross_entropy(forward(model, ids, passes)[:, :-1].float().transpose(1, 2), target,
                                   ignore_index=-100, reduction="none")
            size = lengths[target.clamp(min=0)] * (target != -100)
            starts_document = torch.tensor([s for _, s in batch], device=device)
            size[:, 0] -= (starts_document & (size[:, 0] > 0)).long()
            nll, nbytes = nll + loss[size > 0].sum().item(), nbytes + size.sum().item()
            batch = []
        if item is not None:
            batch.append(item)
    return {"bpb": nll / (math.log(2) * nbytes), "bytes": int(nbytes)}


@torch.no_grad()
def trajectory(model, ids: torch.Tensor, stand_in: str | None, generator: torch.Generator, exact=None):
    """Per-position NLL, window offset (0 reads a real state) and draft acceptance of one batch of windows, and the
    real states read; `stand_in` None is exact decoding, and `exact`, its states, selects the corrected context."""
    heads = model.config.instance.heads
    inputs, targets = ids[:, :-1], ids[:, 1:]
    batch, length = inputs.shape
    rows = torch.arange(batch, device=ids.device)
    x, cos, sin = model.embed(inputs)
    corrected = exact is not None
    # reads[:, j] is what position j + 1 reads.
    reads = exact[0].clone() if corrected else torch.zeros(batch, length, x.shape[-1], device=ids.device, dtype=x.dtype)
    tops = exact[1] if corrected else torch.zeros_like(reads)
    offset = torch.zeros(batch, dtype=torch.long, device=ids.device)
    drafts = stand_ins = None  # (batch, heads, ...): the window's drafts of the next tokens and their stand-ins
    steps = length - heads + 1  # the chain at j reads the embeddings of tokens up to j + heads
    nll = torch.zeros(batch, steps, device=ids.device)
    offsets = torch.zeros(batch, steps, dtype=torch.long, device=ids.device)
    acceptance = torch.zeros(batch, steps, device=ids.device)
    for j in range(steps):
        state = model.mix(x[:, :j + 1], reads[:, :j + 1]) if j else x[:, :1]
        top = model.blocks(state, cos[:j + 1], sin[:j + 1])
        verifier = F.log_softmax(model.readout(top[:, j]).float(), -1)
        nll[:, j] = -verifier.gather(1, targets[:, j:j + 1])[:, 0]
        offsets[:, j] = offset
        passes = torch.zeros(batch, dtype=torch.bool, device=ids.device)
        if not corrected:
            tops[:, j] = reads[:, j] = top[:, j]
        if drafts is not None:
            live, k = offset < heads, offset.clamp(max=heads - 1)
            overlap = torch.minimum(drafts[rows, k].exp(), verifier.exp()).sum(-1)
            acceptance[:, j] = overlap * live
            passes = live & (torch.rand(batch, device=ids.device, generator=generator) < overlap)
            if stand_in is not None:
                reads[:, j] = torch.where(passes[:, None], stand_ins[rows, k], reads[:, j])
        if corrected:
            # The window that ends at j: the read of its position at offset c sits at j - c.
            for c in range(1, heads + 1):
                fix = ~passes & (offset >= c)
                reads[fix, j - c] = exact[0][fix, j - c]
        if j + 1 >= steps:
            break
        g = model.mtp(tops[:, :j + 1], x[:, 1:j + 2], cos[:j + 1], sin[:j + 1])
        chain = [g[:, j]]
        for c, module in enumerate(model.mtp_later, 2):
            g = module(g, x[:, c:j + c + 1], cos[:j + 1], sin[:j + 1])
            chain.append(g[:, j])
        chain = torch.stack(chain, 1)
        new_drafts = F.log_softmax(model.readout(chain).float(), -1)
        new_stand_ins = (torch.zeros_like(chain) if stand_in == "zeros" else
                         model.mtp.feedback(chain) if stand_in == "state_alignment" else chain)
        # A window that goes on keeps the drafts made at its start.
        keep = passes[:, None, None]
        drafts = new_drafts if drafts is None else torch.where(keep, drafts, new_drafts)
        stand_ins = new_stand_ins if stand_ins is None else torch.where(keep, stand_ins, new_stand_ins)
        offset = torch.where(passes, offset + 1, torch.zeros_like(offset))
    return nll[:, BURN_IN:], offsets[:, BURN_IN:], acceptance[:, BURN_IN:], reads, tops


def stand_in_costs(model, ids: torch.Tensor, stand_ins: list[str]) -> dict:
    """Mean NLL cost (nats) of each stand-in in the kept and corrected contexts, and each head's acceptance; one
    generator per measurement, seeded 0, runs over the batches in order."""
    heads = model.config.instance.heads
    cells = [(s, context) for s in stand_ins for context in ("kept", "corrected")]
    generators = {cell: torch.Generator(device=ids.device).manual_seed(0) for cell in ("exact", *cells)}
    exact_nll, costs, offsets, accepted = [], {cell: [] for cell in cells}, [], []
    for start in range(0, len(ids), BATCH):
        chunk = ids[start:start + BATCH]
        nll, offset, acceptance, reads, tops = trajectory(model, chunk, None, generators["exact"])
        exact_nll.append(nll)
        offsets.append(offset)
        accepted.append(acceptance)
        for s, context in cells:
            other = trajectory(model, chunk, s, generators[s, context], (reads, tops) if context == "corrected" else None)
            costs[s, context].append(other[0] - nll)
    offsets, accepted = torch.cat(offsets), torch.cat(accepted)
    out = {"exact_nll": torch.cat(exact_nll).mean().item(),
           "acceptance": [accepted[offsets == h].mean().item() for h in range(heads)]}
    for (s, context), parts in costs.items():
        out.setdefault(s, {})[context] = torch.cat(parts).mean().item()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--paloma", type=Path, required=True, help="the raw shards of python -m lorm.eval.paloma raw")
    parser.add_argument("--out", type=Path, required=True, help="directory for one JSON per run")
    parser.add_argument("--stand-ins", default=",".join(STAND_INS))
    parser.add_argument("--passes", type=int, default=4, help="parallel passes of exact BPB")
    parser.add_argument("--precision", default="float16", choices=PRECISIONS)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])
    lengths = token_byte_lengths(tokenizer)
    shards = sorted(args.paloma.glob("*.bin"))
    args.out.mkdir(parents=True, exist_ok=True)
    for path in args.runs:
        model = load_run(path, device).to(PRECISIONS[args.precision])
        feedback = hasattr(model, "fusion")
        stand_ins = [s for s in args.stand_ins.split(",")
                     if feedback and (s != "state_alignment" or model.mtp.feedback is not None)]
        result = {"run": path.name, "precision": args.precision, "subsets": {}}
        for shard in shards:
            tokens = read_shard(shard)
            row = result["subsets"][shard.stem] = {"exact": document_bpb(model, tokens, lengths.to(device), args.passes)}
            if stand_ins:
                ids = windows(tokens)
                # nats per scored token -> bits per byte, with the bytes per token of the windows' scored span
                ratio = 1 / (math.log(2) * utf8_bytes(lengths.numpy(), ids[:, BURN_IN + 1:], ids[:, BURN_IN:-1])
                             / ids[:, BURN_IN + 1:].size)
                costs = stand_in_costs(model, torch.from_numpy(ids).to(device), stand_ins)
                row["acceptance"] = costs["acceptance"]
                row |= {s: {c: v * ratio for c, v in costs[s].items()} for s in stand_ins}
            print(path.name, shard.stem, json.dumps(row), flush=True)
        subsets = list(result["subsets"].values())
        # The paper's macro mean over subsets.
        result["mean"] = {key: ({c: float(np.mean([r[key][c] for r in subsets])) for c in subsets[0][key] if c != "bytes"}
                                if isinstance(subsets[0][key], dict) else np.mean([r[key] for r in subsets], 0).tolist())
                          for key in subsets[0]}
        (args.out / f"{path.name}.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
