"""Zero-shot HellaSwag, ARC-Easy, PIQA and LAMBADA of trained checkpoints.

Prompts and metrics follow lm-evaluation-harness (https://github.com/EleutherAI/lm-evaluation-harness):
a multiple-choice item picks the choice of highest summed log-probability (acc) or of highest
log-probability per UTF-8 byte of the choice (acc_norm); LAMBADA counts a last word as predicted
when every token of it is the top prediction (acc), and reports exp of the mean negative
log-probability of the last words (ppl). Every prompt begins with the token that begins a training
document (BOS where the tokenizer has one, as in the Dolma 3 shards, else end-of-text). As in the
harness, the continuation's tokens are those that encoding context + continuation adds to the
context's, so tokenizers that merge across the boundary or add a prefix space score the same text.

`--passes` sets the passes of a model that takes them, such as a feedback Transformer (default: its configured
passes); other models ignore it.

DIR holds hellaswag.jsonl (validation), arc_easy.jsonl (test), piqa.jsonl (validation) and lambada.jsonl
(lambada_openai, English), one row per line as on the Hugging Face Hub (lorm/eval/fetch_benchmarks.py). Each RUN is a
run directory with config.yaml and checkpoint.pt; import the work that registers its model first, or use --work:

    python -m lorm.eval.downstream RUN [RUN ...] --data-dir DIR --out benchmarks.json [--passes 4] [--work MODULE]
"""

import argparse
from importlib import import_module
import json
import math
import re
from pathlib import Path

import torch
from torch.nn import functional as F
import yaml

from ..model import ModelConfig, build_model

TASKS = ("hellaswag", "arc_easy", "piqa", "lambada")


def hellaswag_text(text: str) -> str:
    text = text.strip().replace(" [title]", ". ")
    return re.sub(r"\[.*?\]", "", text).replace("  ", " ")


def items(task: str, row: dict) -> tuple[list[tuple[str, str]], int]:
    """(context, continuation) pairs and the index of the correct one."""
    if task == "hellaswag":
        context = hellaswag_text(row["activity_label"] + ": " + row["ctx_a"] + " " + row["ctx_b"].capitalize())
        return [(context, " " + hellaswag_text(ending)) for ending in row["endings"]], int(row["label"])
    if task == "arc_easy":
        context = f"Question: {row['question']}\nAnswer:"
        choices = row["choices"]
        return [(context, " " + text) for text in choices["text"]], choices["label"].index(row["answerKey"])
    if task == "piqa":
        context = f"Question: {row['goal']}\nAnswer:"
        return [(context, " " + row["sol1"]), (context, " " + row["sol2"])], row["label"]
    context, word = row["text"].rsplit(" ", 1)
    return [(context, " " + word)], 0


def encode(tokenizer, context: str, continuation: str) -> tuple[list[int], list[int]]:
    """The context's tokens after the document-start token, and the continuation's."""
    start = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    head = tokenizer.encode(context, add_special_tokens=False)
    whole = tokenizer.encode(context + continuation, add_special_tokens=False)
    if whole[:len(head)] != head:
        whole = head + tokenizer.encode(continuation, add_special_tokens=False)
    return [start] + head, whole[len(head):]


@torch.no_grad()
def score(model, pairs: list[tuple[list[int], list[int]]], length: int, batch: int, device,
          **options) -> list[tuple[float, bool]]:
    """Summed log-probability of each continuation and whether every token of it is the top prediction."""
    results = [None] * len(pairs)
    order = sorted(range(len(pairs)), key=lambda i: -len(pairs[i][0]) - len(pairs[i][1]))
    for start in range(0, len(order), batch):
        chunk = order[start:start + batch]
        # Contexts longer than the model's window keep their end.
        sequences = [(pairs[i][0] + pairs[i][1])[-length:] for i in chunk]
        width = -(-max(map(len, sequences)) // 64) * 64  # few distinct shapes
        tokens = torch.zeros(len(chunk), min(width, length), dtype=torch.long)
        for row, sequence in enumerate(sequences):
            tokens[row, :len(sequence)] = torch.tensor(sequence)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(tokens.to(device), **options).float()
        logprobs = F.log_softmax(logits, -1)
        for row, i in enumerate(chunk):
            count, end = len(pairs[i][1]), len(sequences[row])
            targets = tokens[row, end - count:end].to(device)
            predicted = logprobs[row, end - count - 1:end - 1]
            results[i] = (predicted.gather(-1, targets[:, None]).sum().item(),
                          bool((predicted.argmax(-1) == targets).all()))
    return results


def evaluate(model, tokenizer, rows: dict[str, list[dict]], length: int, batch: int, device,
             **options) -> dict[str, dict]:
    metrics = {}
    for task, task_rows in rows.items():
        examples = [items(task, row) for row in task_rows]
        pairs = [encode(tokenizer, context, continuation) for choices, _ in examples for context, continuation in choices]
        scored = iter(score(model, pairs, length, batch, device, **options))
        if task == "lambada":
            results = [next(scored) for _ in examples]
            nll = -sum(logp for logp, _ in results) / len(results)
            metrics[task] = {"acc": sum(hit for _, hit in results) / len(results), "ppl": math.exp(nll)}
            continue
        hits = norm_hits = 0
        for choices, answer in examples:
            logps = [next(scored)[0] for _ in choices]
            per_byte = [logp / len(continuation.encode()) for logp, (_, continuation) in zip(logps, choices)]
            hits += max(range(len(logps)), key=logps.__getitem__) == answer
            norm_hits += max(range(len(per_byte)), key=per_byte.__getitem__) == answer
        metrics[task] = {"acc": hits / len(examples), "acc_norm": norm_hits / len(examples)}
    return metrics


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--limit", type=int, help="first N rows per task, for a quick check")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--passes", type=int, help="passes of models that take them")
    parser.add_argument("--work", help="module that registers the runs' models, e.g. works.latent_feedback")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.work:
        import_module(args.work)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = {task: [json.loads(line) for line in (args.data_dir / f"{task}.jsonl").read_text().splitlines()][:args.limit]
            for task in args.tasks.split(",")}
    results = json.loads(args.out.read_text()) if args.out.exists() else {}
    for run in args.runs:
        config = yaml.safe_load((run / "config.yaml").read_text())
        model = build_model(ModelConfig.from_dict(config["model"]))
        state = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        model.to(device).eval()
        takes_passes = "passes" in (config["model"].get("instance") or {})
        options = {"passes": args.passes} if takes_passes and args.passes else {}
        settings = config["tokenizer"].copy()
        tokenizer = AutoTokenizer.from_pretrained(settings.pop("name"), **settings)
        metrics = evaluate(model, tokenizer, rows, config["model"]["max_seq_len"], args.batch, device, **options)
        results[run.name] = {"step": state["progress"]["step"], **options, **metrics}
        print(run.name, json.dumps(results[run.name]), flush=True)
        args.out.write_text(json.dumps(results, indent=1))
        del model, state
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
