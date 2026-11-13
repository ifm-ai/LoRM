"""Atomically save a completed optimizer step and each rank's replay state.

Model, optimizer and scheduler state is shared; loader, epoch, RNG and metric-window state belong to each rank.
Resume with the same world size and settings. A branch (`branch_from`) continues another run's checkpoint and may
change only the algorithm, evaluation, and the instance fields its config class lists in `branch_fields`, e.g. to turn
on a training objective for a learning-rate decay. Checkpoints load through PyTorch's safe loader.
"""

import os
import pickle
from pathlib import Path
import random

import numpy as np
import torch
from torch import distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions, get_model_state_dict, get_optimizer_state_dict, set_optimizer_state_dict,
)

from ..model import ModelConfig

BRANCH_SETTINGS = {"algo", "eval", "branch_from", "output_dir", "wandb"}


def rng_state(device):
    numpy = np.random.get_state()
    return {"python": random.getstate(), "numpy": (numpy[0], numpy[1].tolist(), *numpy[2:]),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def restore_rng(state, device):
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], np.array(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


def load_optimizer_state(model, optimizer, saved):
    """Restore a full optimizer state dict, including parameters without state: a parameter that has had no gradient
    yet, or one a branch adds. They borrow the fresh optimizer's initial state for loading and drop it afterwards, so
    they stay stateless as in an uninterrupted run."""
    options = StateDictOptions(full_state_dict=True)
    known = {fqn for group in saved["param_groups"] for fqn in group["params"]}
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    missing = [fqn for fqn in known if fqn not in saved["state"]]
    if missing or not names <= known:
        fresh = get_optimizer_state_dict(model, optimizer, options=options)
        if len(fresh["param_groups"]) != len(saved["param_groups"]):
            raise ValueError("The checkpoint's optimizer has different parameter groups.")
        groups = [{**group, "params": group["params"] + [fqn for fqn in current["params"] if fqn not in known]}
                  for group, current in zip(saved["param_groups"], fresh["param_groups"])]
        missing = [fqn for group in groups for fqn in group["params"] if fqn not in saved["state"]]
        saved = {"state": saved["state"] | {fqn: fresh["state"][fqn] for fqn in missing}, "param_groups": groups}
    set_optimizer_state_dict(model, optimizer, saved, options=options)
    parameters = dict(model.named_parameters(remove_duplicate=False))
    for fqn in missing:
        optimizer.state.pop(parameters[fqn], None)


def save_checkpoint(path, *, model, optimizers, schedulers, loader, progress, epoch, config, run_id, device,
                    keep=2, metrics=None):
    """Write `path` atomically; with keep 2 the previous generation stays as <name>.previous.pt, a hard link."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    local = {"loader": loader.state_dict(), "epoch": epoch, "generator": loader.generator.get_state(),
             "rng": rng_state(device), "metrics": metrics}
    ranks = [None] * world_size if rank == 0 else None
    if dist.is_initialized():
        dist.gather_object(local, ranks, dst=0)
    else:
        ranks[0] = local
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    model_state = get_model_state_dict(model, options=options)
    optimizer_states = {name: get_optimizer_state_dict(model, opt, options=options) for name, opt in optimizers.items()}
    error = [None]
    if rank == 0:
        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        previous = path.with_name(path.stem + ".previous" + path.suffix)
        previous_temporary = previous.with_suffix(previous.suffix + ".tmp")
        try:
            state = {"model": model_state, "optimizers": optimizer_states,
                     "schedulers": {name: obj.state_dict() for name, obj in schedulers.items()},
                     "ranks": ranks, "progress": progress, "config": config, "run_id": run_id}
            with temporary.open("wb") as target:
                torch.save(state, target)
                target.flush()
                os.fsync(target.fileno())
            if keep == 2 and path.exists():
                previous_temporary.unlink(missing_ok=True)
                os.link(path, previous_temporary)
                previous_temporary.replace(previous)
            temporary.replace(path)
            if keep == 1:
                previous.unlink(missing_ok=True)
        except Exception as exc:
            error[0] = f"Checkpoint write failed: {exc}"
        finally:
            temporary.unlink(missing_ok=True)
            previous_temporary.unlink(missing_ok=True)
    if dist.is_initialized():
        dist.broadcast_object_list(error, src=0)
    if error[0]:
        raise RuntimeError(error[0])


def _comparable(config: dict, branch: bool) -> dict:
    """The settings a resume or a branch must keep."""
    kept = {key: value for key, value in config.items() if key not in (BRANCH_SETTINGS if branch else {"output_dir"})}
    if branch:
        kept["train"] = {k: v for k, v in config["train"].items() if k != "branch_points"}
        model = ModelConfig.from_dict(config["model"])
        instance = config["model"].get("instance")
        if instance is not None:
            changeable = type(model.instance).branch_fields
            kept["model"] = {**config["model"], "instance": {k: v for k, v in instance.items() if k not in changeable}}
    return kept


def load_checkpoint(path, config, world_size, *, fallback=False, branch=False):
    """Read a checkpoint, falling back to its previous generation when `fallback` and the read fails on any rank."""
    path = Path(path)
    previous = path.with_name(path.stem + ".previous" + path.suffix)
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, EOFError, RuntimeError, pickle.UnpicklingError):
        if not fallback:
            raise
        state = None
    if fallback and dist.is_initialized():
        failed = [None] * dist.get_world_size()
        dist.all_gather_object(failed, state is None)
        if any(failed):
            state = None
    if state is None:
        state = torch.load(previous, map_location="cpu", weights_only=True)
    if not branch and state["run_id"] and any(config.get("wandb", {}).get(key) != state["config"].get("wandb", {}).get(key)
                                              for key in ("project", "entity")):
        raise ValueError("Resuming a W&B run requires the same project and entity.")
    if _comparable(state["config"], branch) != _comparable(config, branch) or len(state["ranks"]) != world_size:
        raise ValueError("Resuming or branching requires the same experiment settings and world size.")
    return state
