"""Write the experiment YAML of a run in configs/runs.csv: its arm's config with the run's seed, rates and schedule.

    python -m works.attnres_feedback.make_config NAME [NAME ...] [--gpus 8] [--out configs/runs]
    python -m works.attnres_feedback.make_config --all

All runs are 134M, one epoch (F = 1) of a U/N 100 or 400 prefix of D3, 256 sequences of 2,048 tokens per update.
Arms ending in -unscaled are the same model without the dim^-1/2 logit scale. `transformer` and `feedback` are the
latent feedback work's arms.
"""

import argparse
import csv
import math
from pathlib import Path

import yaml

from lorm.train.config import load_config
from works.latent_feedback.make_config import EMBEDDING_LR, HEAD_LR, SHARD_TOKENS

HERE = Path(__file__).resolve().parent
SEQUENCES = 256
CORE = {"transformer": "../../latent_feedback/configs/base.yaml", "feedback": "../../latent_feedback/configs/feedback.yaml"}


def runs() -> dict[str, dict]:
    with (HERE / "configs" / "runs.csv").open() as f:
        return {row["name"]: row for row in csv.DictReader(f)}


def make_config(row: dict, gpus: int = 8) -> dict:
    arm = row["arm"].removesuffix("-unscaled")
    config = load_config(HERE / "configs" / CORE.get(arm, f"{arm}.yaml"))
    if row["arm"].endswith("-unscaled"):
        config["model"]["instance"]["scaled_logits"] = False
    # Models with feedback keep a Transformer's memory with half the microbatch, accumulated twice.
    accumulation = 1 if arm in ("transformer", "transformer-attnres") else 2
    if SEQUENCES % (gpus * accumulation):
        raise ValueError(f"{SEQUENCES} sequences per update do not split over {gpus} GPUs.")
    lr, sequences = float(row["lr"]), int(row["train_sequences"])
    config["seed"] = int(row["seed"])
    config["output_dir"] = f"runs/{row['name']}"
    config["data"] |= {"num_sequences": sequences, "num_shards": math.ceil(sequences * 2048 / SHARD_TOKENS) + 1}
    config["optimizer"] |= {"lr": lr, "weight_decay": float(row["wd"]), "embedding_lr": lr * EMBEDDING_LR,
                            "head_lr": lr * HEAD_LR}
    config["train"] |= {"max_steps": int(row["max_steps"]), "warmup_steps": int(row["warmup_steps"]),
                        "decay_steps": int(row["decay_steps"]), "batch_size": SEQUENCES // gpus // accumulation,
                        "gradient_accumulation": accumulation}
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("names", nargs="*")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--out", type=Path, default=HERE / "configs" / "runs")
    args = parser.parse_args()
    table = runs()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in table if args.all else args.names:
        path = args.out / f"{name}.yaml"
        path.write_text(yaml.safe_dump(make_config(table[name], args.gpus), sort_keys=False))
        print(path)


if __name__ == "__main__":
    main()
