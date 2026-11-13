"""Write the experiment YAML of a run in configs/runs.csv: the shared recipe (configs/base.yaml, plus configs/feedback.yaml
for the latent feedback Transformer) with the run's size, unique-data prefix, rates, batch and schedule.

    python -m works.latent_feedback.make_config NAME [NAME ...] [--gpus 8] [--out configs/runs]
    python -m works.latent_feedback.make_config --all

`released` marks the runs whose checkpoints are published: the best learning rate and weight decay of each cell of
the paper, and the 134M seeds. The other rows are the trunks that some of their decay branches start from. A decay
branch (`trunk`, `branch_step`) resumes its trunk's checkpoint at the step where its own decay starts, so train the
trunk first; trunks keep those checkpoints (`branch_points`).
"""

import argparse
import csv
import math
from pathlib import Path

import yaml

from lorm.train.config import load_config

HERE = Path(__file__).resolve().parent
# Width, depth, heads and MLP width.
SIZES = {"60m": (512, 8, 4, 1536), "134m": (768, 12, 6, 2048), "271m": (1024, 16, 8, 2816), "778m": (1536, 24, 12, 4096)}
# GPUs per run as trained; make_config --gpus overrides it and keeps the sequences per update.
GPUS = {"60m": 8, "134m": 8, "271m": 8, "778m": 16}
# Embedding and head learning rates relative to the matrix learning rate.
EMBEDDING_LR, HEAD_LR = 0.9050966799189999, 0.08
SHARD_TOKENS = 100_000_000


def runs() -> dict[str, dict]:
    with (HERE / "configs" / "runs.csv").open() as f:
        return {row["name"]: row for row in csv.DictReader(f)}


def make_config(row: dict, gpus: int | None = None) -> dict:
    config = load_config(HERE / "configs" / ("feedback.yaml" if row["arm"] == "feedback" else "base.yaml"))
    gpus = gpus or GPUS[row["size"]]
    dim, layers, heads, hidden = SIZES[row["size"]]
    lr, beta2, sequences = float(row["lr"]), float(row["beta2"]), int(row["sequences"])
    # Latent feedback keeps a Transformer's memory with half the microbatch, accumulated twice.
    accumulation = 2 if row["arm"] == "feedback" else 1
    if sequences % (gpus * accumulation):
        raise ValueError(f"{sequences} sequences per update do not split over {gpus} GPUs.")
    config["seed"] = int(row["seed"])
    config["output_dir"] = f"runs/{row['name']}"
    config["data"] |= {"num_sequences": int(row["train_sequences"]),
                       "num_shards": math.ceil(int(row["train_sequences"]) * 2048 / SHARD_TOKENS) + 1}
    config["model"] |= {"dim": dim, "n_layers": layers, "n_heads": heads, "n_kv_heads": heads, "hidden_dim": hidden}
    config["optimizer"] |= {"lr": lr, "weight_decay": float(row["wd"]), "embedding_lr": lr * EMBEDDING_LR,
                            "head_lr": lr * HEAD_LR}
    config["optimizer"]["adamw"]["betas"] = [0.9, beta2]
    config["optimizer"]["muon"] |= {"beta2": beta2, "momentum_warmup_steps": int(row["momentum_warmup"])}
    config["train"] |= {"max_steps": int(row["max_steps"]), "warmup_steps": int(row["warmup_steps"]),
                        "decay_steps": int(row["decay_steps"]), "batch_size": sequences // gpus // accumulation,
                        "gradient_accumulation": accumulation,
                        "branch_points": [int(s) for s in row["branch_points"].split()]}
    if row["trunk"]:
        config["branch_from"] = f"runs/{row['trunk']}/checkpoint-step{row['branch_step']}.pt"
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("names", nargs="*")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--gpus", type=int, help="GPUs per run; default: as trained (8, or 16 at 778M)")
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
