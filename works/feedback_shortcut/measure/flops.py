"""Expected training FLOPs per token of each setting over its pass mix, relative to the feedback model and to the
Transformer with the same MTP heads.

The model's own count (training_flops_per_token, which training logs) leaves out two stand-in costs, added here from
the modules' multiply-adds: in-place substitution's MTP chain on the first pass's detached states, forward and
backward, and state alignment's map on each head's detached state, with weight gradients everywhere and input
gradients past its first layers only. The decay-only setting substitutes in the decay steps alone.

    python -m works.feedback_shortcut.measure.flops [--out flops.json]
"""

import argparse
import json
from pathlib import Path

import torch

from lorm.model import ModelConfig, build_model
from lorm.train.algo import build_algorithm
from lorm.train.config import load_config
from lorm.train.main import resolve_config
from .. import algo as _  # noqa: F401  (registration)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SETTINGS = ("transformer-mtp", "feedback", "state-alignment", "in-place", "in-place-half", "in-place-decay")


def per_token(config: dict) -> float:
    with torch.device("meta"):
        model = build_model(ModelConfig.from_dict(config["model"]))
    instance, length = model.config.instance, config["data"]["sequence_length"]
    stand_in = getattr(instance, "stand_in", None)
    algorithm = build_algorithm(config["algo"], seed=0)
    total = 0.0
    # A Transformer's algorithm runs one pass.
    for passes, share in enumerate(getattr(algorithm.config, "pass_mix", [1.0]), 1):
        algorithm.passes = passes
        flops = algorithm.flops_per_token(model, length)
        if passes > 1 and stand_in == "in_place":
            chain = sum(m.forward_macs((1, length - k, model.config.dim)) for k, m in enumerate(model.heads(), 1))
            share_of_steps = config["train"]["decay_steps"] / config["train"]["max_steps"] if "branch_from" in config else 1
            flops += share_of_steps * 6 * chain / length
        if passes > 1 and stand_in == "state_alignment":
            p = model.mtp.feedback
            macs = sum(layer.weight.numel() for layer in (p.linear, p.up, p.down))
            first = p.linear.weight.numel() + p.up.weight.numel()
            rows = sum(length - k for k in range(1, instance.heads + 1))
            flops += 2 * (3 * macs - first) * rows / length
        total += share * flops
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = {}
    for size in ("135m", "780m"):
        flops = {s: per_token(resolve_config(load_config(CONFIGS / size / f"{s}.yaml"))) for s in SETTINGS}
        for setting, value in flops.items():
            row = out[f"{size} {setting}"] = dict(flops_per_token=value, extra=value / flops["feedback"] - 1,
                                                  over_transformer=value / flops["transformer-mtp"])
            print(f"{size} {setting:16s} {value / 1e9:8.3f} GFLOPs per token, extra {100 * row['extra']:5.1f}%, "
                  f"{row['over_transformer']:.3f}x the Transformer")
    if args.out:
        args.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
