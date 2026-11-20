"""Bits per byte of the gold continuation on the Olmo 3 Base Easy tasks (arXiv 2512.13961, Table 45), from the request
files lorm/eval/task_requests.py writes. Per item, as OLMES's bits_per_byte_corr: -log2(e) * log p(continuation |
context) / max(UTF-8 bytes of the continuation, 1); a task is the mean over its items. Multiple-choice tasks keep the
gold choice of the conditional requests only: OLMES also writes unconditional ones (doc_id >= 1e6, context "Answer:")
for acc_uncond, which bits_per_byte_corr does not score. A feedback run is scored at --passes (4 matches token-by-token
decoding). Writes, per run, each task's suite, mean and per-item values (for paired bootstraps) to OUT/<run>.json.

    python -m lorm.eval.task_bpb RUN [RUN ...] --requests DIR --out OUT [--passes 4] [--work works.NAME]
"""

import argparse
import json
import math
from importlib import import_module
from pathlib import Path

import torch
import yaml

from ..model import ModelConfig, build_model
from .downstream import encode, score

UNCONDITIONAL = 1_000_000
TOKENIZER = {"name": "meta-llama/Llama-2-7b-hf", "revision": "01c7f73d771dfac7d292323805ebc428287df4f9"}


def gold_rows(path: Path, limit: int | None = None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return [r for r in rows if r["doc_id"] < UNCONDITIONAL
            and (not isinstance(r["label"], int) or r["idx"] == r["label"])][:limit]


def item_bpb(logp: float, continuation: str) -> float:
    return -logp / math.log(2) / max(len(continuation.encode("utf-8")), 1)


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tasks", help="comma-separated request file stems; default all")
    parser.add_argument("--limit", type=int, help="first N gold items per task, for a quick check")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--passes", type=int, default=4, help="passes of feedback runs")
    parser.add_argument("--work", help="module that registers the runs' models, e.g. works.feedback_shortcut")
    parser.add_argument("--precision", default="bfloat16", choices=["bfloat16", "float16"],
                        help="bfloat16 autocast over float32 weights, or the whole model in float16")
    args = parser.parse_args()
    if args.work:
        import_module(args.work)
    files = sorted(args.requests.glob("*.jsonl"))
    if args.tasks:
        files = [f for f in files if f.stem in args.tasks.split(",")]
    tasks = {f.stem: gold_rows(f, args.limit) for f in files}
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for run in args.runs:
        config = yaml.safe_load((run / "config.yaml").read_text())
        model = build_model(ModelConfig.from_dict(config["model"]))
        model.load_state_dict(torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)["model"])
        model.to(device, torch.float16 if args.precision == "float16" else None).eval()
        options = {"passes": args.passes} if hasattr(model, "fusion") else {}
        settings = dict(config.get("tokenizer", TOKENIZER))
        tokenizer = AutoTokenizer.from_pretrained(settings.pop("name"), **settings)
        length = config["model"]["max_seq_len"]
        result = {"run": run.name, **options, "precision": args.precision, "tasks": {}}
        for task, rows in tasks.items():
            pairs = [encode(tokenizer, r["context"], r["continuation"]) for r in rows]
            scored = score(model, pairs, length, args.batch, device, autocast=args.precision == "bfloat16", **options)
            items = [item_bpb(logp, r["continuation"]) for (logp, _), r in zip(scored, rows)]
            result["tasks"][task] = {"suite": rows[0]["suite"].removesuffix("_bpb"), "bpb": sum(items) / len(items),
                                     "n": len(items), "items": items}
            print(run.name, task, f"{result['tasks'][task]['bpb']:.4f}", flush=True)
        (args.out / f"{run.name}.json").write_text(json.dumps(result))
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
