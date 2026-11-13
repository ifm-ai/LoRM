"""Download and tokenize the Dolma 3 mix that lorm/data/dolma3.py reads, at pinned revisions.

    python -m lorm.data.prepare_dolma3 download RAW_DIR [--files N]
    python -m lorm.data.prepare_dolma3 tokenize RAW_DIR OUT_DIR [--workers N]
    python -m lorm.data.prepare_dolma3 verify OUT_DIR

The Llama-2 tokenizer is gated on the Hub: accept its license and log in before tokenizing.
Each document becomes BOS + tokens + EOS (1 and 2), concatenated.
The raw files are bucketed by source and topic, so they are shuffled with a fixed seed first: a
prefix of the training shards is then a random sample of files rather than of topics. The first
files of the shuffled order, up to VAL_TOKENS, form the validation shard; manifest.json records
the order, so later files stay unseen by a run that reads a prefix of the training shards. Shards
hold SHARD_TOKENS uint16 tokens after a 256 x int32 header [20240520, 1, count, 0, ...], the
modded-nanogpt format lorm/data/tokens.py reads.

`download --files N` fetches only the first N files of that order and writes the whole order to
RAW_DIR/order.json; `tokenize` then reads those files in that order. Since the shuffle depends only
on the number of files, the shards it writes are the same bytes as the first shards of a full run:
a run that trains on a few billion tokens downloads a few gigabytes instead of the 110 GB mix.
"""

import argparse
import io
import json
import random
from multiprocessing import Pool
from pathlib import Path, PurePosixPath

import numpy as np

from .dolma3 import SOURCE

TOKENIZER = ("meta-llama/Llama-2-7b-hf", "01c7f73d771dfac7d292323805ebc428287df4f9")
BOS, EOS, MAGIC = 1, 2, 20240520
SHARD_TOKENS, VAL_TOKENS, SEED = 100_000_000, 20_000_000, 0
FILES_PER_BATCH, DOCUMENTS_PER_ENCODE = 120, 256
ORDER = "order.json"
_tokenizer = None


def _init(path: str) -> None:
    from tokenizers import Tokenizer

    global _tokenizer
    _tokenizer = Tokenizer.from_file(path)
    _tokenizer.encode_special_tokens = True  # a literal "<s>" in the text must not become a document boundary


def tokenize_file(path: Path) -> tuple[str, int, np.ndarray]:
    """All documents of one .jsonl.zst file as one uint16 array of BOS + tokens + EOS each."""
    import zstandard

    pieces, documents, batch = [], 0, []

    def flush():
        for encoding in _tokenizer.encode_batch(batch, add_special_tokens=False):
            pieces.append(np.array([BOS, *encoding.ids, EOS], dtype=np.uint16))
        batch.clear()

    with open(path, "rb") as handle:
        reader = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(handle, read_across_frames=True),
                                  encoding="utf-8")
        for line in reader:
            batch.append(json.loads(line)["text"])
            documents += 1
            if len(batch) == DOCUMENTS_PER_ENCODE:
                flush()
    flush()
    return str(path), documents, np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.uint16)


class ShardWriter:
    def __init__(self, out_dir: Path, split: str) -> None:
        self.out_dir, self.split, self.buffer, self.size, self.counts = out_dir, split, [], 0, []

    def add(self, tokens: np.ndarray) -> None:
        while len(tokens):
            take = min(SHARD_TOKENS - self.size, len(tokens))
            self.buffer.append(tokens[:take])
            self.size += take
            tokens = tokens[take:]
            if self.size == SHARD_TOKENS:
                self.flush()

    def flush(self) -> None:
        if not self.size:
            return
        header = np.zeros(256, dtype=np.int32)
        header[:3] = (MAGIC, 1, self.size)
        with open(self.out_dir / f"dolma3_{self.split}_{len(self.counts):06d}.bin", "wb") as out:
            out.write(header.tobytes())
            for piece in self.buffer:
                out.write(piece.astype("<u2").tobytes())
        self.counts.append(self.size)
        self.buffer, self.size = [], 0


def shuffled(names: list[str]) -> list[str]:
    """The fixed order of the raw files (paths relative to the raw directory): sorted as paths, then
    shuffled with SEED."""
    names = sorted(names, key=PurePosixPath)
    random.Random(SEED).shuffle(names)
    return names


def raw_files(names) -> list[str]:
    return [name for name in names if len(PurePosixPath(name).parts) == 3 and name.startswith("data/")
            and name.endswith(".jsonl.zst")]


def ordered_files(raw_dir: Path) -> list[Path]:
    """The raw files to tokenize, in the fixed order: all of them, or the prefix order.json names."""
    if not (raw_dir / ORDER).exists():
        return [raw_dir / name for name in shuffled([str(p.relative_to(raw_dir)) for p in raw_dir.glob("data/*/*.jsonl.zst")])]
    names = json.loads((raw_dir / ORDER).read_text())
    present = [name for name in names if (raw_dir / name).exists()]
    if not present or present != names[:len(present)]:
        raise ValueError(f"{raw_dir} must hold a prefix of the files {ORDER} lists.")
    return [raw_dir / name for name in present]


def tokenize(raw_dir: Path, tokenizer: Path, out_dir: Path, workers: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = ordered_files(raw_dir)
    writers = {split: ShardWriter(out_dir, split) for split in ("val", "train")}
    order, documents, split = [], 0, "val"
    with Pool(workers, initializer=_init, initargs=(str(tokenizer),)) as pool:
        # Ordered batches bound the memory: imap would buffer every finished file behind a slow one.
        results = (result for start in range(0, len(files), FILES_PER_BATCH)
                   for result in pool.map(tokenize_file, files[start:start + FILES_PER_BATCH], chunksize=1))
        for path, count, tokens in results:
            writers[split].add(tokens)
            order.append({"file": str(Path(path).relative_to(raw_dir)), "split": split, "docs": count,
                          "tokens": int(len(tokens))})
            documents += count
            if split == "val" and writers["val"].size + sum(writers["val"].counts) >= VAL_TOKENS:
                writers["val"].flush()
                split = "train"
    for writer in writers.values():
        writer.flush()
    manifest = {"source": "@".join(SOURCE), "tokenizer": "@".join(TOKENIZER), "document": "BOS + tokens + EOS",
                "seed": SEED, "files": len(files), "docs": documents,
                "shards": {name: writer.counts for name, writer in writers.items()}, "order": order}
    (out_dir / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def verify(out_dir: Path) -> None:
    """Every shard's header against its payload, and the header counts against the manifest."""
    manifest = json.loads((out_dir / "manifest.json").read_text())
    for split, counts in manifest["shards"].items():
        paths = sorted(out_dir.glob(f"dolma3_{split}_*.bin"))
        if len(paths) != len(counts):
            raise ValueError(f"{split}: {len(paths)} shards on disk, {len(counts)} in the manifest.")
        for path, count in zip(paths, counts):
            header = np.fromfile(path, dtype=np.int32, count=256)
            payload = (path.stat().st_size - 1024) // 2
            if tuple(header[:3]) != (MAGIC, 1, count) or payload != count:
                raise ValueError(f"{path.name}: header {header[:3].tolist()}, {payload} tokens, manifest {count}.")
    print({split: sum(counts) for split, counts in manifest["shards"].items()}, "tokens verified")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    step = commands.add_parser("download")
    step.add_argument("raw_dir", type=Path)
    step.add_argument("--files", type=int, help="only the first N files of the fixed order")
    step = commands.add_parser("tokenize")
    step.add_argument("raw_dir", type=Path)
    step.add_argument("out_dir", type=Path)
    step.add_argument("--workers", type=int, default=32)
    commands.add_parser("verify").add_argument("out_dir", type=Path)
    args = parser.parse_args()
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    if args.command == "download":
        patterns = None
        if args.files is not None:
            names = shuffled(raw_files(HfApi().list_repo_files(SOURCE[0], repo_type="dataset", revision=SOURCE[1])))
            args.raw_dir.mkdir(parents=True, exist_ok=True)
            (args.raw_dir / ORDER).write_text(json.dumps(names))
            patterns = names[:args.files]
        snapshot_download(SOURCE[0], repo_type="dataset", revision=SOURCE[1], local_dir=args.raw_dir, max_workers=16,
                          allow_patterns=patterns)
    elif args.command == "tokenize":
        tokenizer = hf_hub_download(TOKENIZER[0], "tokenizer.json", revision=TOKENIZER[1])
        manifest = tokenize(args.raw_dir, Path(tokenizer), args.out_dir, args.workers)
        print({split: (len(counts), sum(counts)) for split, counts in manifest["shards"].items()})
    else:
        verify(args.out_dir)


if __name__ == "__main__":
    main()
