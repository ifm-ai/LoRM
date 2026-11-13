"""NLL of trained runs on evaluation shards, per shard, in 2048-token sequences. For a feedback Transformer, also the
NLL after each of `passes` passes, which at 4 passes matches token-by-token decoding within 1e-4.

    python -m lorm.eval.shards RUN [RUN ...] --data SHARD [SHARD ...] --out nll.json [--passes 1,4] [--work MODULE]
"""

import argparse
from importlib import import_module
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..data import TokenDataset
from ..model import FeedbackTransformer
from .nll import evaluate
from .run import load_run


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--data", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--passes", default="4", help="comma-separated pass counts of feedback models")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--work", help="module that registers the runs' models, e.g. works.latent_feedback")
    args = parser.parse_args()
    if args.work:
        import_module(args.work)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = json.loads(args.out.read_text()) if args.out.exists() else {}
    for path in args.runs:
        model = load_run(path, device)
        feedback = isinstance(model, FeedbackTransformer)
        variants = {f"P{k}": {"passes": int(k)} for k in args.passes.split(",")} if feedback else {}
        out = results.setdefault(path.name, {})
        for shard in args.data:
            loader = DataLoader(TokenDataset([shard], 2048, offset=1024), batch_size=args.batch)
            out[shard.stem] = evaluate(model, loader, device=device, precision="bfloat16" if device.type == "cuda"
                                       else "float32", variants=variants)
            print(path.name, shard.stem, json.dumps(out[shard.stem]), flush=True)
        args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
