"""Aggregate token-weighted loss and training-only throughput across ranks.

Timing excludes evaluation/checkpoints. CUDA completion is synchronized once per
optimizer step; peak memory is the maximum across ranks since the previous log.
Model FLOPs per GPU per second use the slowest rank's time; MFU divides them by
the configured per-GPU peak.
"""

import torch
from torch import distributed as dist


class Monitor:
    def __init__(self, device, peak_tflops=None):
        self.device, self.peak_tflops = device, peak_tflops
        self.reset()

    def reset(self):
        # Sums of the open logging window, checkpointed so a resumed run logs the same averages.
        self.loss = self.targets = self.grad_norm = 0.0
        self.updates = 0
        # This process's throughput since the previous log.
        self.tokens = self.flops = self.seconds = self.data_seconds = 0.0
        self.steps = 0
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def update(self, *, loss, targets, tokens, grad_norm, seconds, data_seconds, flops=0.0):
        self.flops += flops
        self.loss += loss
        self.targets += targets
        self.tokens += tokens
        self.grad_norm += grad_norm
        self.seconds += seconds
        self.data_seconds += data_seconds
        self.updates += 1
        self.steps += 1

    def state_dict(self) -> dict[str, float]:
        return {"loss": self.loss, "targets": self.targets, "grad_norm": self.grad_norm, "updates": self.updates}

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.loss, self.targets, self.grad_norm, self.updates = (
            state[name] for name in ("loss", "targets", "grad_norm", "updates"))

    def collect(self):
        totals = torch.tensor([self.loss, self.targets, self.tokens, self.grad_norm, self.updates, self.flops],
                              dtype=torch.float64, device=self.device)
        peaks = [self.seconds, self.data_seconds]
        if self.device.type == "cuda":
            peaks.extend([torch.cuda.max_memory_allocated(self.device) / 2**30,
                          torch.cuda.max_memory_reserved(self.device) / 2**30])
        peaks = torch.tensor(peaks, dtype=torch.float64, device=self.device)
        if dist.is_initialized():
            dist.all_reduce(totals)
            dist.all_reduce(peaks, op=dist.ReduceOp.MAX)
        loss, targets, tokens, grad_norm, updates, flops = totals.tolist()
        seconds, data_seconds, *memory = peaks.tolist()
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        metrics = {"train/loss": loss / targets, "train/grad_norm": grad_norm / updates,
                   "train/tokens_per_second": tokens / max(seconds, 1e-9),
                   "time/step_s": seconds / self.steps,
                   "time/data_s": data_seconds / self.steps}
        if flops:
            metrics["perf/tflops_per_gpu"] = flops / max(seconds, 1e-9) / world_size / 1e12
            if self.peak_tflops:
                metrics["perf/mfu"] = metrics["perf/tflops_per_gpu"] / self.peak_tflops
        if memory:
            metrics["memory/allocated_gib"], metrics["memory/reserved_gib"] = memory
        self.reset()
        return metrics
