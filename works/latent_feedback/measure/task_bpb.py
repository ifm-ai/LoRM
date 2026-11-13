"""Bits per byte of the gold continuation on the Olmo 3 Base Easy tasks (arXiv 2512.13961, Table 45), from the
request files measure/requests.py writes. Per item, as OLMES's bits_per_byte_corr: -log2(e) * log p(continuation |
context) / max(UTF-8 bytes of the continuation, 1); a task is the mean over its items. Multiple-choice requests keep only
the gold choice. A feedback run is scored at --passes (4 matches token-by-token decoding). Writes, per run, the task
means and the per-item values (for paired bootstraps) to OUT/<run>.json.

    python -m works.latent_feedback.measure.task_bpb RUN [RUN ...] --requests DIR --out OUT [--passes 4]
"""

import argparse
import json
import math
from pathlib import Path

import torch
import yaml

from lorm.eval.downstream import encode, score
from lorm.model import ModelConfig, build_model


def gold_rows(path: Path, limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return [r for r in rows if not isinstance(r["label"], int) or r["idx"] == r["label"]][:limit]


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
    args = parser.parse_args()
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
        model.to(device).eval()
        options = {"passes": args.passes} if config["model"].get("instance") else {}
        settings = config["tokenizer"].copy()
        tokenizer = AutoTokenizer.from_pretrained(settings.pop("name"), **settings)
        length = config["model"]["max_seq_len"]
        result = {"run": run.name, **options, "tasks": {}}
        for name, rows in tasks.items():
            pairs = [encode(tokenizer, r["context"], r["continuation"]) for r in rows]
            scored = score(model, pairs, length, args.batch, device, **options)
            items = [-logp / math.log(2) / max(len(r["continuation"].encode("utf-8")), 1)
                     for (logp, _), r in zip(scored, rows)]
            result["tasks"][name] = {"bpb": sum(items) / len(items), "n": len(items), "items": items}
            print(run.name, name, f"{result['tasks'][name]['bpb']:.4f}", flush=True)
        (args.out / f"{run.name}.json").write_text(json.dumps(result))
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
