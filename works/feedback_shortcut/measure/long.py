"""Long generations at correction depth D: 1,024 new tokens after 128-token prompts from the raw Paloma subsets,
prefilled token by token, batch 1, float16.

Greedy: each prompt is continued by exact decoding and at each depth. A position is at risk up to and including the
first token where the continuation departs from exact decoding's, and a near-tie is a position where exact decoding's
two best tokens are within 0.1 nats. A departure is at a near-tie when exact decoding scores the chosen token within
0.1 nats of its own. Per depth: the departures, the positions at risk, the near-ties among them, the departures at a
near-tie, and from the first departure on the mean log-probability exact decoding gives the served continuation and its
own.

Sampling at temperature 1 (speculative sampling) on the first prompts of each subset: the mean KL from the
distribution each committed token was drawn from to exact decoding's on the same prefix.

    python -m works.feedback_shortcut.measure.long RUN --stand-in state_alignment --paloma data/paloma/raw --out OUT.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lorm.eval.run import load_run
from .. import model as _  # noqa: F401  (registration)
from .engine import STAND_INS, Engine, kl
from .quality import PRECISIONS, read_shard

NEAR = 0.1


def prompts(shards: list[Path], count: int, length: int) -> list[list[int]]:
    """`count` prompts of `length` tokens from each shard, seed 1."""
    out = []
    for shard in shards:
        tokens = read_shard(shard)
        starts = np.random.default_rng(1).integers(0, len(tokens) - length - 1, count)
        out.append([tokens[s:s + length].tolist() for s in starts])
    return out


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--stand-in", required=True, choices=STAND_INS)
    parser.add_argument("--paloma", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompts", type=int, default=20, help="greedy prompts per subset")
    parser.add_argument("--sampled", type=int, default=5, help="of them, the prompts per subset also sampled")
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--new", type=int, default=1024)
    parser.add_argument("--depths", default="0,1,2", help="greedy")
    parser.add_argument("--sampled-depths", default="1,2")
    parser.add_argument("--precision", default="float16", choices=PRECISIONS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_run(args.run, device).to(PRECISIONS[args.precision])
    by_subset = prompts(sorted(args.paloma.glob("*.bin")), args.prompts, args.prompt_length)
    greedy, sampled = sum(by_subset, []), sum((p[:args.sampled] for p in by_subset), [])
    length = args.prompt_length + args.new + 64
    exact = Engine(model, "zeros", None, length)
    scorer = Engine(model, "zeros", None, length, temperature=1.0)
    log = lambda p: p.clamp(min=1e-30).log()  # noqa: E731
    result = dict(run=args.run.name, stand_in=args.stand_in, precision=args.precision, prompts=len(greedy),
                  sampled=len(sampled), new=args.new, greedy={}, kl={})
    references, scores = [], []
    for prompt in greedy:
        references.append(exact.generate(prompt, args.new).tokens)
        scores.append(log(scorer.exact_distributions(prompt, references[-1])))
    for depth in map(int, args.depths.split(",")):
        engine = Engine(model, args.stand_in, depth, length)
        g = dict(departed=0, at_risk=0, near_at_risk=0, at_near=0, served=[], exact=[])
        for prompt, reference, score in zip(greedy, references, scores):
            out = engine.generate(prompt, args.new).tokens
            first = next((i for i, (a, b) in enumerate(zip(out, reference)) if a != b), None)
            best = score.topk(2, -1).values
            near = (best[:, 0] - best[:, 1]) < NEAR
            stop = args.new if first is None else first + 1
            g["at_risk"] += stop
            g["near_at_risk"] += int(near[:stop].sum())
            if first is None:
                continue
            g["departed"] += 1
            g["at_near"] += int((score[first, reference[first]] - score[first, out[first]]).item() < NEAR)
            own = log(scorer.exact_distributions(prompt, out)).gather(1, torch.tensor(out, device=device)[:, None])[:, 0]
            g["served"].append(own[first:].mean().item())
            g["exact"].append(score.gather(1, torch.tensor(reference, device=device)[:, None])[first:, 0].mean().item())
        g["served"] = float(np.mean(g["served"])) if g["served"] else None
        g["exact"] = float(np.mean(g["exact"])) if g["exact"] else None
        result["greedy"][str(depth)] = g
        print("greedy", depth, json.dumps(g), flush=True)
    for depth in map(int, args.sampled_depths.split(",")):
        engine, divergences = Engine(model, args.stand_in, depth, length, temperature=1.0), []
        torch.manual_seed(args.seed)
        for prompt in sampled:
            record = []
            out = engine.generate(prompt, args.new, record=record).tokens
            divergences.append(kl(torch.stack(record[:args.new]), scorer.exact_distributions(prompt, out)))
        result["kl"][str(depth)] = float(np.mean(divergences))
        print("sampled", depth, result["kl"][str(depth)], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
