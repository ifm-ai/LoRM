"""Causal pretraining with DDP, gradient accumulation and step-boundary checkpoints.

Launch one process per GPU with torchrun. Each rank reads its own part of every epoch; the loss is normalized by the
global number of prediction targets, and only each update's last microbatch synchronizes gradients. SIGTERM and SIGINT
finish the current update, then every rank checkpoints and the run exits; `--resume auto` continues it. `branch_from`
starts a run from another run's checkpoint with fresh output, e.g. a learning-rate decay with another objective.
"""

import argparse
from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta
import os
from pathlib import Path
import random
import signal
from time import perf_counter

import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer
import yaml

from ..data import DATASETS, load_data, token_byte_lengths
from ..eval import evaluate
from ..model import Block, ModelConfig, build_model, initialize, model_class
from .algo import Algorithm, build_algorithm, resolve_algorithm
from .checkpoint import load_checkpoint, load_optimizer_state, restore_rng, save_checkpoint
from .config import TrainConfig, load_config
from .logger import Logger
from .monitor import Monitor
from .optim import build_optimizers
from .scheduler import build_scheduler


def resolve_config(config: dict) -> dict:
    """The experiment with every section validated and defaults filled in, as the run directory records it."""
    config = deepcopy(config)
    model = ModelConfig.from_dict(config["model"])
    config["model"] = model.to_dict()
    config["train"] = TrainConfig(**config["train"]).to_dict()
    config.setdefault("seed", 42)
    data = config["data"]
    if data.get("name") not in DATASETS:
        raise ValueError(f"data.name must be one of {sorted(DATASETS)}.")
    data.setdefault("revision", DATASETS[data["name"]][1])
    settings = config.setdefault("eval", {})
    if "num_sequences" not in settings:
        raise ValueError("Set the validation budget with eval.num_sequences.")
    model_class(model).eval_variants(model, **settings.get("variants", {}))
    config["algo"] = resolve_algorithm(config.get("algo"))
    if type(config.get("branch_from", "")) is not str:
        raise ValueError("branch_from must be a checkpoint path.")
    return config


def _loaders(config, settings, rank, world_size, device):
    """The training and validation loaders, and the tokenizer's byte lengths for eval/bpb."""
    options = config["tokenizer"].copy()
    tokenizer = AutoTokenizer.from_pretrained(options.pop("name"), **options)
    if len(tokenizer) > config["model"]["vocab_size"]:
        raise ValueError("The model vocabulary is smaller than the tokenizer vocabulary.")
    training, validation = load_data(tokenizer, **config["data"])
    # Random reads of uncached shards on a network filesystem wait on every page; one sequential read fills the
    # node's page cache, each local rank reading a disjoint part.
    local, local_ranks = int(os.environ.get("LOCAL_RANK", 0)), int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    for dataset in (training, validation):
        dataset.warm(local, local_ranks, threads=max(1, 4 // local_ranks))
    sampler = DistributedSampler(training, num_replicas=world_size, rank=rank, shuffle=settings.shuffle,
                                 seed=config["seed"], drop_last=True)
    loader = StatefulDataLoader(training, batch_size=settings.batch_size, sampler=sampler, drop_last=True,
                                pin_memory=device.type == "cuda",
                                generator=torch.Generator().manual_seed(config["seed"] + rank))
    count = config["eval"]["num_sequences"]
    if type(count) is not int or not 0 < count <= len(validation):
        raise ValueError("eval.num_sequences must select a nonempty available validation prefix.")
    validation = DataLoader(Subset(validation, range(rank, count, world_size)),
                            batch_size=config["eval"].get("batch_size", settings.batch_size),
                            pin_memory=device.type == "cuda")
    return loader, validation, token_byte_lengths(tokenizer)


def _seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def _batches(loader, iterator, epoch, count, reshuffle):
    batches = []
    for _ in range(count):
        batch = next(iterator, None)
        if batch is None:
            epoch += 1
            loader.sampler.set_epoch(epoch if reshuffle else 0)
            iterator = iter(loader)
            batch = next(iterator, None)
        if batch is None:
            break
        batches.append(batch)
    return batches, iterator, epoch


def clip_gradients(model, max_norm: float):
    """Clip the gradient by its global norm and return the norm. Parameters the model lists in `clipped_apart` are
    clipped on their own, so that their gradient cannot rescale the rest."""
    apart = list(getattr(model, "clipped_apart", lambda: [])())
    if apart:
        torch.nn.utils.clip_grad_norm_(apart, max_norm, error_if_nonfinite=True)
    ids = {id(p) for p in apart}
    return torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if id(p) not in ids], max_norm,
                                          error_if_nonfinite=True)


def run(config, *, resume=None):
    config = resolve_config(config)
    settings = TrainConfig(**config["train"])
    device = torch.device(settings.device)
    if device.type == "cuda":
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        torch.cuda.set_device(device)
    owns_group = "RANK" in os.environ and not dist.is_initialized()
    if owns_group:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo", timeout=timedelta(minutes=5))
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    logger, exit_code, stopped, handlers = None, 1, [0], {}

    def stop(signum, _frame):
        stopped[0] = signum

    try:
        handlers[signal.SIGTERM] = signal.signal(signal.SIGTERM, stop)
        output_dir = Path(config["output_dir"])
        checkpoint = output_dir / "checkpoint.pt"
        if resume == "auto":
            resume = "latest" if checkpoint.exists() else None
        if resume is None and checkpoint.exists():
            raise FileExistsError("A checkpoint already exists; use --resume or another output_dir.")
        state, branch = None, None
        if resume:
            state = load_checkpoint(checkpoint if resume == "latest" else resume, config, world_size,
                                    fallback=resume == "latest")
        elif "branch_from" in config:
            branch = config["branch_from"]
            state = load_checkpoint(branch, config, world_size, branch=True)
        fresh = state is None
        _seed(config["seed"])
        model = build_model(ModelConfig.from_dict(config["model"]))
        if config["data"]["sequence_length"] > model.config.max_seq_len:
            raise ValueError("data.sequence_length must fit model.max_seq_len.")
        if fresh or branch:
            initialize(model)
        if state is not None:
            # Parameters a branch adds keep their initialization.
            added = model.load_state_dict(state["model"], strict=not branch).missing_keys
        model.to(device)
        wrapped = (DDP(model, device_ids=[device.index] if device.type == "cuda" else None,
                       find_unused_parameters=model.conditional_parameters) if world_size > 1 else model)
        if settings.compile == "blocks":
            for block in (module for module in model.modules() if isinstance(module, Block)):
                block.compile()
        optimizers = build_optimizers(model, **config["optimizer"])
        schedulers = {name: build_scheduler(opt, settings) for name, opt in optimizers.items()}
        algorithm = build_algorithm(config["algo"], seed=config["seed"])
        loader, validation, token_bytes = _loaders(config, settings, rank, world_size, device)
        logger = Logger(output_dir, config, rank=rank, run_id=state["run_id"] if state and not branch else None)
        if rank == 0:
            (output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        update_sequences = settings.batch_size * settings.gradient_accumulation * world_size
        logger.info("%d parameters | %d ranks | effective batch %d sequences",
                    sum(p.numel() for p in model.parameters()), world_size, update_sequences)
        epoch_tokens = (world_size * settings.batch_size * (len(loader.sampler) // settings.batch_size)
                        * config["data"]["sequence_length"])
        monitor = Monitor(device, settings.peak_tflops)
        progress = {"step": 0, "input_tokens": 0, "target_tokens": 0, "flops": 0.0}
        epoch = 0
        if state:
            for name, opt in optimizers.items():
                load_optimizer_state(model, opt, state["optimizers"][name])
                schedulers[name].load_state_dict(state["schedulers"][name])
            progress = state["progress"].copy()
            local = state["ranks"][rank]
            epoch = local["epoch"]
            loader.sampler.set_epoch(epoch if settings.reshuffle else 0)
            loader.generator.set_state(local["generator"])
            loader.load_state_dict(local["loader"])
            if not branch:
                monitor.load_state_dict(local["metrics"]["monitor"])
                algorithm.load_state_dict(local["metrics"]["algorithm"])
        iterator = iter(loader)
        if state:
            # Building the iterator must not advance the next epoch's generator.
            loader.generator.set_state(local["generator"])
            restore_rng(local["rng"], device)
            if branch:
                logger.info("Branched at step %d from %s; new parameters: %s", progress["step"], branch,
                            ", ".join(added) or "none")
            else:
                logger.info("Resumed step %d from %s", progress["step"], resume)
            del state
        else:
            _seed(config["seed"] + rank)
        handlers[signal.SIGINT] = signal.signal(signal.SIGINT, stop)
        variants = type(model).eval_variants(model.config, **config["eval"].get("variants", {}))

        def stop_signal():
            """The largest signal any rank received, so that every rank takes the same branch."""
            received = torch.tensor(stopped[0], device=device)
            if dist.is_initialized():
                dist.all_reduce(received, op=dist.ReduceOp.MAX)
            return int(received.item())

        def validate():
            return evaluate(model, validation, device=device, precision=settings.precision, variants=variants,
                            token_bytes=token_bytes)

        logger.info("%.4g model FLOPs per token of the default call",
                    Algorithm().flops_per_token(model, config["data"]["sequence_length"]))
        model.train()
        while progress["step"] < settings.max_steps:
            started = perf_counter()
            batches, iterator, epoch = _batches(loader, iterator, epoch, settings.gradient_accumulation, settings.reshuffle)
            targets = sum((batch["labels"][:, 1:] != -100).sum().item() for batch in batches)
            tokens = sum(batch["input_ids"].numel() for batch in batches)
            counts = torch.tensor([len(batches) == settings.gradient_accumulation, targets, tokens],
                                  dtype=torch.float64, device=device)
            if dist.is_initialized():
                dist.all_reduce(counts)
            ready, global_targets, global_tokens = counts.tolist()
            if ready != world_size or global_targets == 0:
                raise ValueError("Every rank must provide a full training batch with prediction targets.")
            data_seconds = perf_counter() - started
            algorithm.begin_update(progress["step"])
            per_token = algorithm.flops_per_token(model, batches[0]["input_ids"].shape[1])
            for opt in optimizers.values():
                opt.zero_grad(set_to_none=True)
            loss_sum = torch.zeros((), device=device)
            for index, batch in enumerate(batches):
                last = index + 1 == len(batches)
                with wrapped.no_sync() if isinstance(wrapped, DDP) and not last else nullcontext():
                    inputs = batch["input_ids"].to(device, non_blocking=True)
                    labels = batch["labels"].to(device, non_blocking=True)
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=settings.precision == "bfloat16"):
                        loss = algorithm.loss(wrapped, inputs, labels)
                    (loss * world_size / global_targets).backward()
                    loss_sum += loss.detach()
            grad_norm = clip_gradients(model, settings.max_grad_norm or float("inf"))
            learning_rates = {f"lr/{name}": opt.param_groups[0]["lr"] for name, opt in optimizers.items()}
            for name, opt in optimizers.items():
                opt.step()
                schedulers[name].step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            monitor.update(loss=loss_sum.item(), targets=targets, tokens=tokens, grad_norm=grad_norm.item(),
                           seconds=perf_counter() - started, data_seconds=data_seconds, flops=per_token * tokens)
            progress["step"] += 1
            progress["input_tokens"] += int(global_tokens)
            progress["target_tokens"] += int(global_targets)
            progress["flops"] += per_token * global_tokens
            step = progress["step"]
            final = step == settings.max_steps
            if step % settings.log_every == 0 or final:
                logger.log(step, monitor.collect() | learning_rates | algorithm.metrics() | {
                    "train/input_tokens": progress["input_tokens"], "train/target_tokens": progress["target_tokens"],
                    "train/flops": progress["flops"], "train/epochs": progress["input_tokens"] / epoch_tokens})
            # Evaluation and checkpoints take long, so a signal during either is checked right after them.
            signum = stop_signal()
            if settings.evaluates_at(step) and (final or not signum):
                logger.log(step, validate() | {"train/flops": progress["flops"]})
                signum = signum or stop_signal()
            kept = step in settings.branch_points
            if signum or final or kept or (settings.checkpoint_every and step % settings.checkpoint_every == 0):
                save_checkpoint(checkpoint, keep=settings.checkpoint_keep, model=model, optimizers=optimizers,
                                schedulers=schedulers, loader=loader, progress=progress, epoch=epoch, config=config,
                                run_id=logger.run_id, device=device,
                                metrics={"monitor": monitor.state_dict(), "algorithm": algorithm.state_dict()})
                logger.info("Checkpoint saved at step %d", step)
                if kept and rank == 0:
                    # A hard link keeps this generation after the rolling checkpoint moves on.
                    branch_point = output_dir / f"checkpoint-step{step}.pt"
                    branch_point.unlink(missing_ok=True)
                    os.link(checkpoint, branch_point)
                signum = signum or stop_signal()
            if signum and not final:
                exit_code = 128 + signum
                raise SystemExit(exit_code)
        exit_code = 0
        return progress
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        try:
            if logger is not None:
                logger.close(exit_code=exit_code)
        finally:
            if owns_group:
                dist.destroy_process_group()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train a model from an experiment YAML.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", help="'auto', 'latest' or a checkpoint path")
    args = parser.parse_args(argv)
    run(load_config(args.config), resume=args.resume)
