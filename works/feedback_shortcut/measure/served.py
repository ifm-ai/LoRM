"""BPB that speculative decoding serves at each correction depth D, measured on a given text: drafts are kept while they
equal it, and each of its tokens is scored by the call that commits it, at the depth it commits with (engine.py's
teacher-forced decoding). "exact" is width-1 exact decoding after the same prefill.

- paloma: quality.py's windows of the raw Paloma subsets; each window's first BURN_IN + 1 tokens are prefilled in 4
  parallel passes and the rest is served. Per subset the mean NLL over the served tokens,
  converted to BPB with the windows' bytes per token, as quality.py converts the one-call error.
- downstream: the gold answers of the Olmo 3 Base Easy tasks (lorm.eval.task_bpb's conditional rows), each context
  prefilled in 4 parallel passes, keeping its end within the context window. Per task the mean over items of
  the answer's NLL over its UTF-8 bytes, as lorm.eval.task_bpb scores it; --shard/--shards split each task's items.

    python -m works.feedback_shortcut.measure.served RUN --stand-in state_alignment --source paloma \
        --data data/paloma/raw --out results/served/paloma-RUN.json
    ... --source downstream --data data/task-bpb --shard 0 --shards 8
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from lorm.data import token_byte_lengths
from lorm.eval.downstream import encode
from lorm.eval.task_bpb import gold_rows
from lorm.eval.run import load_run
from .. import model as _  # noqa: F401  (registration)
from .engine import STAND_INS, Engine
from .quality import BURN_IN, PRECISIONS, TOKENIZER, read_shard, utf8_bytes, windows

CONTEXT = 2048
# Room past the context for the drafts after its last token, which nothing commits.
LENGTH = CONTEXT + 32




def engines(model, stand_in: str, depths: list[int | None], prefill_passes: int | None) -> dict:
    return {d: Engine(model, stand_in, d, LENGTH, prefill_passes=prefill_passes, score=True) for d in depths}


def name(depth):
    return "exact" if depth is None else str(depth)


@torch.no_grad()
def serve_paloma(model, stand_in, depths, shards: list[Path], lengths: np.ndarray) -> dict:
    out = {}
    decoders = engines(model, stand_in, depths, prefill_passes=4)
    for shard in shards:
        ids = windows(read_shard(shard))
        ratio = 1 / (math.log(2) * utf8_bytes(lengths, ids[:, BURN_IN + 1:], ids[:, BURN_IN:-1]) / ids[:, BURN_IN + 1:].size)
        nll = {name(d): 0.0 for d in depths}
        calls = {name(d): 0 for d in depths}
        for window in ids.tolist():
            prompt, text = window[:BURN_IN + 1], window[BURN_IN + 1:]
            for d, engine in decoders.items():
                served = engine.generate(prompt, 0, target=text)
                nll[name(d)] += sum(served.nll)
                calls[name(d)] += served.calls
        tokens = ids[:, BURN_IN + 1:].size
        out[shard.stem] = dict(tokens=tokens, calls=calls, bpb={d: v / tokens * ratio for d, v in nll.items()})
        print(shard.stem, {d: round(v, 5) for d, v in out[shard.stem]["bpb"].items()}, flush=True)
    return out




def downstream(tokenizer, requests: Path, shard: int, shards: int) -> dict:
    out = {}
    for path in sorted(requests.glob("*.jsonl")):
        rows = gold_rows(path)
        out[path.stem] = []
        for i in range(shard, len(rows), shards):
            context, answer = encode(tokenizer, rows[i]["context"], rows[i]["continuation"])
            out[path.stem].append(dict(index=i, suite=rows[i]["suite"].removesuffix("_bpb"), answer=answer,
                                       prompt=(context + answer)[-CONTEXT:][:-len(answer)],
                                       bytes=max(len(rows[i]["continuation"].encode("utf-8")), 1)))
    return out


@torch.no_grad()
def serve_downstream(model, stand_in, depths, tasks: dict) -> dict:
    out = {}
    decoders = engines(model, stand_in, depths, prefill_passes=4)
    for task, rows in tasks.items():
        items = {name(d): [] for d in depths}
        for x in rows:
            for d, engine in decoders.items():
                items[name(d)].append(sum(engine.generate(x["prompt"], 0, target=x["answer"]).nll) / math.log(2) / x["bytes"])
        out[task] = dict(suite=rows[0]["suite"], index=[x["index"] for x in rows], items=items,
                         bpb={d: float(np.mean(v)) for d, v in items.items()})
        print(task, {d: round(v, 4) for d, v in out[task]["bpb"].items()}, flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--stand-in", required=True, choices=STAND_INS)
    parser.add_argument("--source", required=True, choices=["paloma", "downstream"])
    parser.add_argument("--data", type=Path, required=True,
                        help="the raw Paloma shards, or the task requests of lorm.eval.task_requests")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--depths", default="0,1,2,3,exact")
    parser.add_argument("--precision", default="float16", choices=PRECISIONS)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])
    model = load_run(args.run, device).to(PRECISIONS[args.precision])
    depths = [None if d == "exact" else int(d) for d in args.depths.split(",")]
    if args.source == "paloma":
        parts = serve_paloma(model, args.stand_in, depths, sorted(args.data.glob("*.bin")), token_byte_lengths(tokenizer).numpy())
    else:
        parts = serve_downstream(model, args.stand_in, depths, downstream(tokenizer, args.data, args.shard, args.shards))
    mean = {d: float(np.mean([p["bpb"][d] for p in parts.values()])) for d in map(name, depths)}
    result = dict(run=args.run.name, stand_in=args.stand_in, source=args.source, precision=args.precision,
                  shard=args.shard, shards=args.shards, parts=parts, bpb=mean)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
