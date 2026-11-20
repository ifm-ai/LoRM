"""Write the config of a paper run: its setting's config (configs/<size>/<setting>.yaml) with the run's seed and
output directory. configs/runs.csv lists the paper's 36 runs, named <setting>-<size> with -s1 and -s2 for the extra
seeds; the zeros and MTP-state stand-ins are read from the feedback runs.

    python -m works.feedback_shortcut.make_config in-place-135m-s1 [NAME ...] [--out DIR]
    python -m works.feedback_shortcut.make_config --all

A decay-only run branches from the feedback run of its seed where the decay starts, so train that run first.
"""

import argparse
import csv
import os
from pathlib import Path

import yaml

from lorm.train.config import load_config

HERE = Path(__file__).resolve().parent


def runs() -> dict[str, dict]:
    with (HERE / "configs" / "runs.csv").open() as f:
        return {row["name"]: row for row in csv.DictReader(f)}


def make_config(row: dict, out: Path) -> dict:
    setting = HERE / "configs" / row["size"] / f"{row['setting']}.yaml"
    seed, suffix = int(row["seed"]), "" if row["seed"] == "0" else f"-s{row['seed']}"
    config = {"base": os.path.relpath(setting, out), "seed": seed, "output_dir": f"runs/{row['name']}"}
    branch = load_config(setting).get("branch_from")
    if branch:
        trunk, checkpoint = Path(branch).parts[-2:]
        config["branch_from"] = f"runs/{trunk}{suffix}/{checkpoint}"
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--out", type=Path, default=HERE / "configs" / "runs")
    args = parser.parse_args()
    table = runs()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in table if args.all else args.names:
        path = args.out / f"{name}.yaml"
        path.write_text(yaml.safe_dump(make_config(table[name], args.out.resolve()), sort_keys=False))
        print(path)


if __name__ == "__main__":
    main()
