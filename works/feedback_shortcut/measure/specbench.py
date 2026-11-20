"""Spec-Bench (Xia et al., 2024) wall-clock timing of a run at batch size 1 (Tables 5, 6 and 8): exact decoding and
pipelined correction at each depth D on the 480 questions, in one GPU job so that their times compare. The arms take
turns question by question (in reverse order on every other question), so that a change in the GPU's load while the
job runs falls on all of them alike.

As Spec-Bench's evaluation/eval.py with base models: each question's turns are joined with the model's earlier replies
by blank lines (no chat template), a reply runs to at most 256 new tokens and stops at EOS, the first question runs
three times as warm-up, and each turn's wall time spans the whole request: prefill (4 parallel passes, which exact
decoding shares) and decoding with the host's bookkeeping. A request runs on an engine whose buffer, in steps of 256
positions, holds its prompt, its reply and a window of overshoot; a prompt too long for the context keeps BOS and its
last tokens. Every engine's graphs are captured before the first timed request.

With --temperature, every arm samples (speculative sampling for the drafts), and the share of identical turns in
`score` means nothing. `score` reads the answer files: Spec-Bench's speedup (the mean over questions of tokens per second, over exact
decoding's) per task and overall, the tokens committed per call, the positions verified per committed token (rho) and
the share of turns identical to exact decoding's.

    python -m works.feedback_shortcut.measure.specbench run RUN --stand-in state_alignment --depths exact,0,1,2,3 \
        --questions data/spec_bench/question.jsonl --out results/specbench/RUN
    python -m works.feedback_shortcut.measure.specbench score results/specbench/RUN [...] --out specbench.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from lorm.eval.run import load_run
from .. import model as _  # noqa: F401  (registration)
from .engine import STAND_INS, Engine
from .quality import PRECISIONS, TOKENIZER

LENGTHS = tuple(range(256, 2049, 256))
# Every arm picks its buffer as if its window were the widest (16), so the arms of a request attend over one buffer.
SLACK = 17
MT_BENCH = ("writing", "roleplay", "reasoning", "math", "coding", "extraction", "stem", "humanities")
TASKS = ("mt_bench", "translation", "summarization", "qa", "math_reasoning", "rag")


class Decoder:
    """One engine per buffer length for one depth (None: exact decoding)."""

    def __init__(self, model, stand_in: str, depth: int | None, prefill_passes: int, context: int,
                 temperature: float = 0.0) -> None:
        self.context = min(context, LENGTHS[-1])
        self.engines = {n: Engine(model, stand_in, depth, n, prefill_passes=prefill_passes, temperature=temperature)
                        for n in LENGTHS if n <= self.context}
        for engine in self.engines.values():
            engine.capture()
            engine.generate([1, *range(1000, 1015)], 8)

    def generate(self, prompt: list[int], new: int, end: int):
        budget = self.context - new - SLACK
        if len(prompt) > budget:
            prompt = prompt[:1] + prompt[len(prompt) - budget + 1:]
        engine = self.engines[next(n for n in LENGTHS if len(prompt) + new + SLACK <= n)]
        out = engine.generate(prompt, new, end=end)
        # Tokens each call kept: the first comes from prefill, and commits past the end are cut.
        left, kept = len(out.tokens) - 1, []
        for commit in out.commits:
            kept.append(max(0, min(commit, left)))
            left -= kept[-1]
        return out.tokens, kept, out.widths


def answer(decoder, tokenizer, question: dict, new: int) -> dict:
    messages, turn = [], dict(turns=[], new_tokens=[], wall_time=[], accept_lengths=[], verified_positions=[])
    for text in question["turns"]:
        messages.append(text)
        ids = tokenizer("\n\n".join(messages)).input_ids
        torch.cuda.synchronize()
        start = time.time()
        tokens, kept, widths = decoder.generate(ids, new, tokenizer.eos_token_id)
        torch.cuda.synchronize()
        turn["wall_time"].append(time.time() - start)
        reply = tokenizer.decode(tokens, spaces_between_special_tokens=False)
        for special in tokenizer.all_special_tokens:
            reply = reply.replace(special, "")
        reply = reply.strip()
        messages.append(reply)
        turn["turns"].append(reply)
        turn["new_tokens"].append(len(tokens))
        turn["accept_lengths"] += kept
        turn["verified_positions"] += widths
    return turn


def run(args):
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])
    questions = [json.loads(line) for line in args.questions.open()]
    model = load_run(args.run, device).to(PRECISIONS[args.precision])
    args.out.mkdir(parents=True, exist_ok=True)
    names = args.depths.split(",")
    paths = {name: args.out / f"{name}.jsonl" for name in names}
    # A requeued job resumes after the last question every arm finished.
    rows = {name: path.read_text().splitlines() if path.exists() else [] for name, path in paths.items()}
    done = min(len(r) for r in rows.values())
    if done == len(questions):
        return
    with torch.no_grad():
        decoders = {name: Decoder(model, args.stand_in, None if name == "exact" else int(name), args.prefill_passes,
                                  model.config.max_seq_len, args.temperature) for name in names}
        torch.manual_seed(args.seed)
        for decoder in decoders.values():
            for _ in range(3):
                answer(decoder, tokenizer, questions[0], args.new)
        files = {name: path.open("w") for name, path in paths.items()}
        for name, f in files.items():
            f.writelines(line + "\n" for line in rows[name][:done])
        for i in range(done, len(questions)):
            question = questions[i]
            for name in names if i % 2 == 0 else names[::-1]:
                row = dict(question_id=question["question_id"], category=question["category"],
                           **answer(decoders[name], tokenizer, question, args.new))
                files[name].write(json.dumps(row) + "\n")
                files[name].flush()
        for f in files.values():
            f.close()
        print(args.run.name, "done", flush=True)


def speedup(rows: list[dict], base: list[dict]) -> float:
    rate = lambda rs: np.mean([sum(r["new_tokens"]) / sum(r["wall_time"]) for r in rs])  # noqa: E731
    return float(rate(rows) / rate(base))


def score(directory: Path) -> dict:
    read = lambda p: [json.loads(line) for line in p.read_text().splitlines()]  # noqa: E731
    exact = read(directory / "exact.jsonl")
    out = {}
    for path in sorted(directory.glob("*.jsonl")):
        if path.stem == "exact":
            continue
        rows = read(path)
        row = {}
        for task in (*TASKS, "overall"):
            pick = lambda rs: [r for r in rs if task == "overall" or r["category"] == task  # noqa: E731
                               or task == "mt_bench" and r["category"] in MT_BENCH]
            row[task] = speedup(pick(rows), pick(exact))
        kept = sum((r["accept_lengths"] for r in rows), [])
        verified = sum((r["verified_positions"] for r in rows), [])
        same = [a == b for r, e in zip(rows, exact) for a, b in zip(r["turns"], e["turns"])]
        out[path.stem] = dict(speedup=row, tokens_per_call=float(np.mean(kept)), rho=sum(verified) / sum(kept),
                              identical=sum(same) / len(same))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    timing = commands.add_parser("run")
    timing.add_argument("run", type=Path)
    timing.add_argument("--stand-in", default="mtp_state", choices=STAND_INS, help="mtp_state for a Transformer")
    timing.add_argument("--depths", default="exact,0,1,2,3")
    timing.add_argument("--questions", type=Path, required=True, help="Spec-Bench's data/spec_bench/question.jsonl")
    timing.add_argument("--out", type=Path, required=True)
    timing.add_argument("--new", type=int, default=256)
    timing.add_argument("--prefill-passes", type=int, default=4)
    timing.add_argument("--precision", default="float16", choices=PRECISIONS)
    timing.add_argument("--temperature", type=float, default=0.0, help="0 decodes greedily; else speculative sampling")
    timing.add_argument("--seed", type=int, default=0, help="seeds the sampling")
    scoring = commands.add_parser("score")
    scoring.add_argument("directories", nargs="+", type=Path)
    scoring.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    else:
        result = {d.name: score(d) for d in args.directories}
        for name, arms in result.items():
            for depth, r in arms.items():
                print(f"{name} D={depth}: speedup {r['speedup']['overall']:.2f}, tokens per call {r['tokens_per_call']:.2f}, "
                      f"rho {r['rho']:.2f}, identical {r['identical']:.1%}")
        if args.out:
            args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
