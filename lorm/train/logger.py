"""Rank-zero console, JSONL and W&B logging; no custom logging backend.

W&B uses an explicit training-step axis and a checkpointed run ID. Resuming an
older checkpoint can append replayed steps; it does not rewind remote history.
W&B's own system monitor supplies host/device telemetry. A W&B failure after
initialization, such as its service exiting on preemption, disables W&B for the
rest of the run instead of interrupting training or checkpointing.
"""

import json
import logging
from pathlib import Path
import sys

import wandb


class Logger:
    def __init__(self, output_dir, config, *, rank=0, run_id=None):
        self.run = None
        self.run_id = None
        self.metrics = None
        self.handlers = []
        self.logger = logging.getLogger("lorm")
        if rank != 0:
            return
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
        for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(output_dir / "train.log")):
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.handlers.append(handler)
        self.metrics = (output_dir / "metrics.jsonl").open("a")
        options = config.get("wandb", {"mode": "disabled"}).copy()
        mode = options.get("mode", "online")
        options.setdefault("force", mode == "online")
        if mode != "disabled":
            try:
                self.run = wandb.init(
                    **options, dir=str(output_dir), config=config, id=run_id,
                    resume="allow" if run_id and mode == "online" else None,
                )
                self.run.define_metric("step")
                self.run.define_metric("*", step_metric="step")
                self.run_id = self.run.id
            except Exception:
                self.close(exit_code=1)
                raise

    def info(self, message, *args):
        if self.metrics is not None:
            self.logger.info(message, *args)

    def log(self, step, metrics):
        if self.metrics is None:
            return
        record = {"step": step, **metrics}
        self.metrics.write(json.dumps(record, allow_nan=False) + "\n")
        self.metrics.flush()
        if self.run is not None:
            self._wandb(self.run.log, record)
        self.info("step %d | %s", step, " | ".join(f"{key}={value:.5g}" for key, value in metrics.items()))

    def _wandb(self, call, *args, **kwargs):
        try:
            call(*args, **kwargs)
        except Exception as exc:
            self.run = None
            self.logger.warning("W&B disabled for the rest of this run: %r", exc)

    def close(self, exit_code=0):
        try:
            if self.run is not None:
                self._wandb(self.run.finish, exit_code=exit_code)
        finally:
            if self.metrics is not None:
                self.metrics.close()
            for handler in self.handlers:
                self.logger.removeHandler(handler)
                handler.close()
