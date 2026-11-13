"""Token-weighted NLL, perplexity and bits per byte over an exact, unpadded validation subset.

Call it with the underlying model, not the DDP wrapper, since ranks may hold unequal numbers of batches. `variants`
(Transformer.eval_variants) adds eval/nll@{suffix} for forward calls with other options, such as more passes.
eval/bpb divides the NLL of the non-special targets, in bits, by their UTF-8 bytes; unlike NLL and perplexity it
compares across tokenizers.
"""

import math

import torch
from torch import distributed as dist



def evaluate(model, loader, *, device, precision="float32", variants: dict[str, dict] | None = None,
             token_bytes: torch.Tensor | None = None) -> dict[str, float]:
    """eval/nll, eval/ppl and token counts of the default call, and eval/nll@{suffix} of each variant;
    `token_bytes` (lorm.data.token_byte_lengths) adds eval/bpb."""
    metrics = _evaluate(model, loader, device, precision, {}, None if token_bytes is None else token_bytes.to(device))
    evaluated = {(): metrics["eval/nll"]}
    for suffix, options in (variants or {}).items():
        key = tuple(sorted(options.items()))
        if key not in evaluated:
            evaluated[key] = _evaluate(model, loader, device, precision, options)["eval/nll"]
        metrics[f"eval/nll@{suffix}"] = evaluated[key]
    return metrics


@torch.no_grad()
def _evaluate(model, loader, device, precision, options: dict, token_bytes=None) -> dict[str, float]:
    # Imported here: lorm.train imports this module, so importing lorm.eval first would close the cycle.
    from ..train.loss import causal_loss

    was_training = model.training
    model.eval()
    totals = torch.zeros(5, dtype=torch.float64, device=device)
    try:
        for batch in loader:
            inputs = batch["input_ids"].to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == "bfloat16"):
                logits = model(inputs, **options)
            labels = batch["labels"].to(device, non_blocking=True)
            losses = causal_loss(logits, labels, reduction="none").double()
            targets = labels[:, 1:].reshape(-1)
            totals[0] += losses.sum()
            totals[1] += (targets != -100).sum()
            totals[2] += inputs.numel()
            if token_bytes is not None:
                lengths = token_bytes[targets.clamp(min=0)] * (targets != -100)
                totals[3] += losses[lengths > 0].sum()
                totals[4] += lengths.sum()
        if dist.is_initialized():
            dist.all_reduce(totals)
        loss, targets, inputs, text_loss, text_bytes = totals.tolist()
        if targets == 0 or not math.isfinite(loss):
            raise ValueError("Evaluation requires a nonempty set of finite token losses.")
        nll = loss / targets
        metrics = {"eval/nll": nll, "eval/ppl": math.exp(nll), "eval/target_tokens": int(targets),
                   "eval/input_tokens": int(inputs)}
        if text_bytes:
            metrics |= {"eval/bpb": text_loss / (math.log(2) * text_bytes), "eval/target_bytes": int(text_bytes)}
        return metrics
    finally:
        model.train(was_training)
