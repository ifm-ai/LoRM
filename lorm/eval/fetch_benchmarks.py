"""Download the zero-shot benchmarks at pinned revisions as one JSONL each, for lorm.eval.downstream.

    python -m lorm.eval.fetch_benchmarks DIR      (needs the prep extra for pyarrow)
"""

import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

# task: (dataset, revision, file); the splits lm-evaluation-harness scores.
BENCHMARKS = {
    "hellaswag": ("Rowan/hellaswag", "218ec52e09a7e7462a5400043bb9a69a41d06b76", "data/validation-00000-of-00001.parquet"),
    "arc_easy": ("allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453", "ARC-Easy/test-00000-of-00001.parquet"),
    "piqa": ("ybisk/piqa", "142c51238b3ca2bc61e9a075913871b8b600e8e1", "plain_text/validation/0000.parquet"),
    "lambada": ("EleutherAI/lambada_openai", "900124bf3b8235c6daf21033af9948b3f07346c4", "data/lambada_test_en.jsonl"),
}


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for task, (repo, revision, file) in BENCHMARKS.items():
        path = hf_hub_download(repo, file, repo_type="dataset", revision=revision)
        if path.endswith(".jsonl"):
            rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
        else:
            import pyarrow.parquet as pq
            rows = pq.read_table(path).to_pylist()
        (out / f"{task}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        print(task, len(rows), "rows", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
