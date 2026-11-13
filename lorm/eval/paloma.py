"""Seven Paloma validation subsets (Magnusson et al., 2023) as evaluation shards, raw or decontaminated.

    tokenize OUT      download allenai/paloma at a pinned revision, tokenize every document with the Llama 2 tokenizer
                      (no special tokens) -> OUT/work/<subset>.npz
    raw OUT           each document as BOS + tokens + EOS -> OUT/raw/<subset>.bin, the training shards' format
    decontaminate OUT --train DIR [--workers N]
                      drop every segment of at most 2048 tokens of which 70% or more of the 8-grams occur in the
                      training shards in DIR (PaLM's rule) -> OUT/decontaminated/<subset>.bin and contamination.csv

The paper's main numbers use the raw shards; the decontaminated ones are its appendix check.
"""

import argparse
import csv
import gzip
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO, REVISION = "allenai/paloma", "65cd6fc59dba021b21db414fa5e8d7765ffbe5e6"
TOKENIZER = ("meta-llama/Llama-2-7b-hf", "01c7f73d771dfac7d292323805ebc428287df4f9")
SUBSETS = ["c4_100_domains", "falcon-refinedweb", "dolma-v1_5", "redpajama", "m2d2_s2orc_unsplit",
           "m2d2_wikipedia_unsplit", "dolma_100_programing_languages"]
N, PRIME = 8, np.uint64(0x9E3779B97F4A7C15)


def write_shard(path: Path, documents: list[np.ndarray]) -> int:
    """BOS + tokens + EOS per document after a 256 x int32 header (magic 20240520, version 1, token count)."""
    stream = np.concatenate([np.concatenate([[1], d, [2]]).astype(np.uint16) for d in documents])
    header = np.zeros(256, np.int32)
    header[:3] = [20240520, 1, len(stream)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as target:
        target.write(header.tobytes())
        target.write(stream.tobytes())
    return len(stream)


def documents(out: Path, subset: str) -> list[np.ndarray]:
    work = np.load(out / "work" / f"{subset}.npz")
    return [work["tokens"][a:b] for a, b in zip(work["offsets"][:-1], work["offsets"][1:])]


def ngrams(tokens: np.ndarray) -> np.ndarray:
    """Rolling 64-bit hashes of every 8-gram."""
    t = tokens.astype(np.uint64)
    if len(t) < N:
        return np.empty(0, np.uint64)
    h = np.zeros(len(t) - N + 1, np.uint64)
    with np.errstate(over="ignore"):
        for k in range(N):
            h = h * PRIME + t[k:len(t) - N + 1 + k]
    return h


def tokenize(out: Path) -> None:
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])
    root = Path(snapshot_download(REPO, repo_type="dataset", revision=REVISION,
                                  allow_patterns=[f"{s}/val/*" for s in SUBSETS]))
    (out / "work").mkdir(parents=True, exist_ok=True)
    for subset in SUBSETS:
        texts = []
        for path in sorted((root / subset / "val").iterdir()):
            with gzip.open(path, "rt") as source:
                texts += [json.loads(line)["text"] for line in source]
        ids = tokenizer(texts, add_special_tokens=False)["input_ids"]
        np.savez(out / "work" / f"{subset}.npz", tokens=np.concatenate([np.array(x, np.uint16) for x in ids]),
                 offsets=np.cumsum([0] + [len(x) for x in ids]))
        print(subset, len(texts), "documents", flush=True)


def raw(out: Path) -> None:
    for subset in SUBSETS:
        print(subset, write_shard(out / "raw" / f"{subset}.bin", documents(out, subset)), "tokens", flush=True)


def _found(arguments) -> np.ndarray:
    path, table = arguments
    hashes = ngrams(np.fromfile(path, dtype=np.uint16, offset=1024))
    index = np.minimum(np.searchsorted(table, hashes), len(table) - 1)
    return np.unique(hashes[table[index] == hashes])


def decontaminate(out: Path, train: Path, workers: int) -> None:
    segments = {}
    for subset in SUBSETS:
        # Segments of at most 2048 tokens: a whole-document fraction would hide copied sections of long documents.
        segments[subset] = [d[s:s + 2048] for d in documents(out, subset) for s in range(0, len(d), 2048)]
    table = np.unique(np.concatenate([ngrams(s) for parts in segments.values() for s in parts]))
    shards = sorted(train.glob("*_train_*.bin"))
    with Pool(workers) as pool:
        found = np.unique(np.concatenate(pool.map(_found, [(p, table) for p in shards], chunksize=4)))
    rows = []
    for subset, parts in segments.items():
        overlap = np.array([np.isin(ngrams(s), found).mean() if len(s) >= N else 0.0 for s in parts])
        kept = [s for s, o in zip(parts, overlap) if o < 0.7]
        tokens = write_shard(out / "decontaminated" / f"{subset}.bin", kept)
        rows.append(dict(subset=subset, segments=len(parts), contaminated=int((overlap >= 0.7).sum()), kept_tokens=tokens))
        print(rows[-1], flush=True)
    with (out / "contamination.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("step", choices=["tokenize", "raw", "decontaminate"])
    parser.add_argument("out", type=Path)
    parser.add_argument("--train", type=Path, help="decontaminate: the directory of training shards")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.step == "decontaminate":
        decontaminate(args.out, args.train, args.workers)
    else:
        {"tokenize": tokenize, "raw": raw}[args.step](args.out)


if __name__ == "__main__":
    main()
