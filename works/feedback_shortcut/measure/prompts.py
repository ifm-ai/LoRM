"""The prompt sets of the batched benchmark (batched.py), in Spec-Bench's question format: all 480 Spec-Bench questions,
then the SPEED-Bench qualitative prompts (Abramovich et al., 2026) that are not in Spec-Bench, first those stored in
the dataset and then those it leaves as placeholders, which NVIDIA Model-Optimizer's examples/specdec_bench
prepare_data.py resolves by downloading their sources (gated sources stay unresolved and are dropped). Our models are
base models with a 2,048-token context, so a request is dropped, not truncated, when BOS, its turns, a reply of 256
tokens after every turn, the separators and one window of overshoot do not fit.

    python -m works.feedback_shortcut.measure.prompts --spec-bench question.jsonl --speed test.parquet \
        --resolved prepared.parquet --out data/prompts
"""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from .quality import TOKENIZER

SETS = ("spec_bench", "speed_unique", "speed_resolved")
CONTEXT, NEW, WINDOW, MARGIN = 2048, 256, 16, 16
PLACEHOLDER = "FULL BENCHMARK DATA SHOULD BE FETCHED FROM THE SOURCE USING SPECDEC_BENCH"


def fits(tokenizer, turns):
    # Each reply in the history adds two blank-line separators (at most 3 tokens each).
    length = len(tokenizer.encode("\n\n".join(turns))) + NEW * len(turns) + 6 * (len(turns) - 1)
    return length + WINDOW + MARGIN <= CONTEXT


def first_turns(directory: Path):
    """The first turns of every set in order, encoded as a base model's prompt (BOS, then the turn as plain text), and
    the end token."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])
    return [tokenizer.encode(json.loads(line)["turns"][0]) for name in SETS
            for line in (directory / f"{name}.jsonl").read_text().splitlines()], tokenizer.eos_token_id


def main():
    import pyarrow.parquet as pq
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--spec-bench", type=Path, required=True, help="Spec-Bench's data/spec_bench/question.jsonl")
    parser.add_argument("--speed", type=Path, required=True, help="SPEED-Bench's qualitative test.parquet")
    parser.add_argument("--resolved", type=Path, required=True, help="prepare_data.py's output for the same split")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])
    spec = [json.loads(line) for line in args.spec_bench.open()]
    seen = {row["turns"][0].strip() for row in spec}
    rows = lambda path: [dict(question_id=r["question_id"], category=r["category"],  # noqa: E731
                              turns=[str(t) for t in r["turns"]]) for r in pq.read_table(path).to_pylist()]
    speed = rows(args.speed)
    placeholders = {r["question_id"] for r in speed if any(t.startswith(PLACEHOLDER) for t in r["turns"])}
    sets = {"spec_bench": [r for r in spec if fits(tokenizer, r["turns"])],
            "speed_unique": [r for r in speed if r["question_id"] not in placeholders
                             and r["turns"][0].strip() not in seen and fits(tokenizer, r["turns"])],
            "speed_resolved": [r for r in rows(args.resolved) if r["question_id"] in placeholders
                               and not any(t.startswith(PLACEHOLDER) for t in r["turns"])
                               and r["turns"][0].strip() not in seen and fits(tokenizer, r["turns"])]}
    args.out.mkdir(parents=True, exist_ok=True)
    for name, kept in sets.items():
        (args.out / f"{name}.jsonl").write_text("".join(json.dumps(dict(question_id=r["question_id"],
                                                        category=r["category"], turns=r["turns"])) + "\n" for r in kept))
        print(name, len(kept))


if __name__ == "__main__":
    main()
