"""Where each depth mix looks: the mean softmax weight of every AttnRes read on each of its sources, and for a 2x loop
the loss increase when the second visit and the output mix lose one source of the first visit.

    python -m works.attnres_feedback.measure.patterns RUN [RUN ...] --val data/d3/<validation shard> [--out results/patterns]

Positions 1.. of 64 validation sequences of 2,048 tokens, float32. A read's sources are the context b0, then the
fed-back blocks f1..f6 (feedback) or the first visit's blocks (loop), then its own pass's completed blocks, then its
block's running sum. Reads are keyed "<attn|mlp> <layer>#<call>" and "output#<call>": the call is the pass (feedback,
evaluated at 4 passes) or the visit (loop).
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import types

import torch
from torch.nn import functional as F

from lorm.data import TokenDataset
from lorm.eval.run import load_run
from works.attnres_feedback.model import DepthMix

SEQUENCES, LENGTH, CHUNK = 64, 2048, 16


class State:
    record, calls, drop = None, None, ()


def recorded(self, completed, partial=None):
    values = completed if partial is None else torch.cat((completed, partial.unsqueeze(0)))
    logits = self.logits(values)
    State.calls[self.label] += 1
    call = State.calls[self.label]
    if State.drop and (call == 2 or self.label == "output"):
        logits = logits.index_fill(0, torch.tensor(State.drop, device=logits.device), float("-inf"))
    weights = logits.softmax(0)
    if State.record is not None:
        w = weights[:, :, 1:].float()
        slot = State.record.setdefault(f"{self.label}#{call}", [0, 0])
        slot[0], slot[1] = slot[0] + w.sum((1, 2)), slot[1] + w[0].numel()
    return (weights.unsqueeze(-1) * values).sum(0).to(completed.dtype)


@torch.no_grad()
def nll(model, chunks, passes, record=None, drop=()):
    State.record, State.drop, out = record, tuple(drop), []
    for c in chunks:
        State.calls = defaultdict(int)
        logits = model(c, passes=passes)
        out.append(F.cross_entropy(logits[:, :-1].flatten(0, 1), c[:, 1:].flatten()))
    State.record, State.drop = None, ()
    return torch.stack(out).mean().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "patterns")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = TokenDataset([args.val], LENGTH, offset=1024)
    chunks = torch.stack([data[i]["input_ids"] for i in range(SEQUENCES)]).long().to(device).split(CHUNK)
    args.out.mkdir(parents=True, exist_ok=True)
    for run in args.runs:
        model = load_run(run, device)
        for name, module in model.named_modules():
            if isinstance(module, DepthMix):
                part = name.split(".")
                module.label = "output" if part[-1] == "output_residual" else \
                    f"{'attn' if part[-1] == 'attention_residual' else 'mlp'} {int(part[-2]) + 1}"
                module.forward = types.MethodType(recorded, module)
        spec = model.config.instance
        loop = spec.full_gradient and spec.shift == 0
        passes = spec.passes if loop or spec.feedback == "none" else 4
        record = {}
        result = {"run": run.name, "nll": nll(model, chunks, passes, record),
                  "reads": {k: (v[0] / v[1]).tolist() for k, v in record.items()}}
        if loop:
            blocks = model.config.n_layers // spec.block_layers
            drops = {"b0": [0], **{f"visit 1 block {j}": [j] for j in range(1, blocks + 1)},
                     "visit 1 all": list(range(1, blocks + 1))}
            result["knockouts"] = {k: nll(model, chunks, passes, drop=d) - result["nll"] for k, d in drops.items()}
        (args.out / f"{run.name}.json").write_text(json.dumps(result))
        print(run.name, f"nll {result['nll']:.4f}", result.get("knockouts", ""))


if __name__ == "__main__":
    main()
