"""Build D3, the training data of Latent Feedback Is a Compute and Data Amplifier: the Dolma 3 mix deduplicated, stratified and decontaminated.

    python -m works.latent_feedback.data SRC OUT [--workers N]

SRC holds the whole mix as tokenized by `python -m lorm.data.prepare_dolma3 tokenize`; only its train shards are read.
1. Deduplicate: each document (BOS to EOS) is hashed with blake2b (8 bytes); the first copy in source order is kept.
   The mix repeats documents (Olmo 3's quality upsampling), so about 60% of its tokens are extra copies.
2. Stratify: each raw file lies in a source-topic bucket (data/<bucket>/). The distinct documents of each bucket are
   permuted with SEED; document i of bucket b gets the key (b's tokens before i + half of i's) / b's token share, and
   all documents are sorted by key. Every prefix then follows the mixture of the whole, and a smaller unique-data
   pool is a prefix of a larger one. The first documents, up to VAL_TOKENS, form the validation shard.
3. Decontaminate: a train document is dropped if it contains a paragraph (newline-separated, at least 13 \\w-words) of
   any Paloma source, val or test split. Matching is exact on Llama 2 tokens, in both forms a paragraph takes: with the
   tokenizer's dummy prefix (a document's first paragraph) and without (one after a newline, token 13). The
   validation shard is left unchanged, and the order of the kept documents too.
OUT gets the shards (prepare_dolma3's format, 100M tokens each) and manifest.json. The distinct documents are held in
memory, 2 bytes per token: about 150 GB for the whole mix.
"""

import argparse
import gzip
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from lorm.data.dolma3 import SOURCE
from lorm.data.prepare_dolma3 import BOS, EOS, TOKENIZER, ShardWriter

VAL_TOKENS, SEED, NL, MIN_WORDS, CHUNK = 100_000_000, 0, 13, 13, 1 << 24
PALOMA = ("allenai/paloma", "65cd6fc59dba021b21db414fa5e8d7765ffbe5e6")  # lorm.eval.paloma's revision
PALOMA_SOURCES = ["4chan_meta_sep", "c4_100_domains", "c4_en", "dolma-v1_5", "dolma_100_programing_languages",
                  "dolma_100_subreddits", "falcon-refinedweb", "gab", "m2d2_s2orc_unsplit", "m2d2_wikipedia_unsplit",
                  "manosphere_meta_sep", "mc4", "ptb", "redpajama", "twitterAAE_HELM_fixed", "wikitext_103"]
M1, M2, M3 = np.uint64(0x9E3779B97F4A7C15), np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB)


def read(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype="<u2", offset=1024)


def doc_hash(tokens: np.ndarray) -> int:
    return int.from_bytes(hashlib.blake2b(memoryview(tokens), digest_size=8).digest(), "little")


def mix(z: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = z + M1
        z = (z ^ (z >> np.uint64(30))) * M2
        z = (z ^ (z >> np.uint64(27))) * M3
        return z ^ (z >> np.uint64(31))


def segments(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hash and start of every run of tokens between BOS, EOS and newline tokens."""
    cut = (t == BOS) | (t == EOS) | (t == NL)
    starts = np.flatnonzero(~cut & np.concatenate([[True], cut[:-1]]))
    if not len(starts):
        return np.empty(0, np.uint64), starts
    idx = np.flatnonzero(~cut)
    at = np.searchsorted(idx, starts)
    lengths = np.diff(np.append(at, len(idx)))
    pos = idx - np.repeat(starts, lengths)
    with np.errstate(over="ignore"):
        h = np.add.reduceat(mix(t[idx].astype(np.uint64) | (pos.astype(np.uint64) << np.uint64(16))), at)
        return mix(h ^ lengths.astype(np.uint64)), starts


def paragraph_hashes(tokenizer, texts: list[str]) -> tuple[np.ndarray, int]:
    """Sorted hashes of the distinct paragraphs of at least MIN_WORDS words in texts, both forms, and their count."""
    paras = list(dict.fromkeys(p for text in texts for p in text.split("\n") if len(re.findall(r"\w+", p)) >= MIN_WORDS))
    first = tokenizer(paras, add_special_tokens=False)["input_ids"]
    inner = [ids[ids.index(NL) + 1:] for ids in tokenizer(["\n" + p for p in paras], add_special_tokens=False)["input_ids"]]
    return np.unique(segments(np.array([x for ids in first + inner for x in ids + [NL]], np.uint16))[0]), len(paras)


def paloma_table(tokenizer) -> tuple[np.ndarray, dict]:
    """Paragraph hashes of all Paloma sources, val and test, and the paragraph count of each source."""
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(PALOMA[0], repo_type="dataset", revision=PALOMA[1],
                                  allow_patterns=[f"{s}/*" for s in PALOMA_SOURCES]))
    tables, counts = [], {}
    for source in PALOMA_SOURCES:
        texts = []
        for path in sorted(root.glob(f"{source}/*/*.gz")):
            with gzip.open(path, "rt") as f:
                texts += [json.loads(line)["text"] for line in f]
        table, counts[source] = paragraph_hashes(tokenizer, texts)
        tables.append(table)
        print(source, counts[source], "paragraphs", flush=True)
    return np.unique(np.concatenate(tables)), counts


def index(src: Path, workers: int) -> dict:
    """Start, length, hash and bucket of every train document of src in source order, and the documents that span
    a shard boundary (no document is longer than a shard)."""
    manifest = json.loads((src / "manifest.json").read_text())
    files = [o for o in manifest["order"] if o["split"] == "train"]
    paths = [src / f"dolma3_train_{i:06d}.bin" for i in range(len(manifest["shards"]["train"]))]
    offsets = np.concatenate([[0], np.cumsum(manifest["shards"]["train"])])

    def scan(i):
        t = read(paths[i])
        b = np.flatnonzero(t == BOS)
        assert len(b) and (t[b[b > 0] - 1] == EOS).all(), f"{paths[i]}: a document does not end with EOS"
        hashes = np.array([doc_hash(t[x:y]) for x, y in zip(b[:-1], b[1:])], dtype=np.uint64)
        return b + offsets[i], hashes, t[:b[0]].copy(), t[b[-1]:].copy()

    with ThreadPoolExecutor(workers) as pool:
        shards = list(pool.map(scan, range(len(paths))))
    assert not len(shards[0][2]), "the first shard must start with a document"
    tails = [s[3] if i + 1 == len(shards) else np.concatenate([s[3], shards[i + 1][2]]) for i, s in enumerate(shards)]
    first_doc = np.concatenate([[0], np.cumsum([len(s[0]) for s in shards])])
    starts = np.concatenate([s[0] for s in shards])
    hashes = np.concatenate([np.append(s[1], np.uint64(doc_hash(tail))) for s, tail in zip(shards, tails)])
    docs_per_file = [o["docs"] for o in files]
    assert len(starts) == sum(docs_per_file), "the manifest's documents do not match the shards"
    names = sorted({o["file"].split("/")[1] for o in files})
    bucket = np.repeat([names.index(o["file"].split("/")[1]) for o in files], docs_per_file)
    return dict(paths=paths, offsets=offsets, first_doc=first_doc, starts=starts, hashes=hashes, bucket=bucket,
                names=names, lengths=np.diff(np.append(starts, offsets[-1])),
                boundary={int(first_doc[i + 1]) - 1: tail for i, tail in enumerate(tails)})


def stratified(hashes: np.ndarray, lengths: np.ndarray, bucket: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The first copy of each distinct document in the stratified order, split into (validation, train) indices."""
    distinct = np.sort(np.unique(hashes, return_index=True)[1])
    db = bucket[distinct]
    share = np.bincount(db, weights=lengths[distinct])
    share /= share.sum()
    by_bucket = distinct[np.argsort(db, kind="stable")]
    edges = np.concatenate([[0], np.cumsum(np.bincount(db))])
    rng = np.random.default_rng(SEED)
    ids, keys = np.empty_like(distinct), np.empty(len(distinct))
    for b, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if hi > lo:
            d = ids[lo:hi] = by_bucket[lo:hi][rng.permutation(hi - lo)]
            keys[lo:hi] = (np.cumsum(lengths[d]) - lengths[d] / 2) / share[b]
    order = ids[np.argsort(keys, kind="stable")]
    val = int(np.searchsorted(np.cumsum(lengths[order]), VAL_TOKENS, side="right"))
    return order[:val], order[val:]


def contaminated(stream: np.ndarray, starts: np.ndarray, table: np.ndarray, workers: int) -> np.ndarray:
    """Whether each document of stream (beginning at starts) contains a paragraph of table."""
    edges = np.append(np.unique(starts[np.searchsorted(starts, np.arange(0, len(stream), CHUNK), side="right") - 1]),
                      len(stream))

    def scan(k):
        h, at = segments(stream[edges[k]:edges[k + 1]])
        i = np.minimum(np.searchsorted(table, h), len(table) - 1)
        return edges[k] + at[table[i] == h]

    with ThreadPoolExecutor(workers) as pool:
        hits = np.concatenate(list(pool.map(scan, range(len(edges) - 1))))
    flags = np.zeros(len(starts), bool)
    flags[np.searchsorted(starts, hits, side="right") - 1] = True
    return flags


def build(src: Path, out: Path, table: np.ndarray, paragraphs: dict, workers: int) -> dict:
    ix = index(src, workers)
    starts, lengths = ix["starts"], ix["lengths"]
    val, train = stratified(ix["hashes"], lengths, ix["bucket"])
    docs = np.concatenate([val, train])
    out_start = np.full(len(starts), -1)
    out_start[docs] = np.cumsum(lengths[docs]) - lengths[docs]
    stream = np.empty(int(lengths[docs].sum()), np.uint16)

    def fill(i):
        t = read(ix["paths"][i])
        lo, hi = ix["first_doc"][i], ix["first_doc"][i + 1]
        for d in lo + np.flatnonzero(out_start[lo:hi] >= 0):
            a = starts[d] - ix["offsets"][i]
            stream[out_start[d]:out_start[d] + lengths[d]] = ix["boundary"][d] if d == hi - 1 else t[a:a + lengths[d]]

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(fill, range(len(ix["paths"]))))
    flags = contaminated(stream, out_start[docs], table, workers)
    out.mkdir(parents=True, exist_ok=True)
    writers = {split: ShardWriter(out, split) for split in ("val", "train")}
    val_tokens = int(lengths[val].sum())
    writers["val"].add(stream[:val_tokens])
    removed, position = docs[len(val):][flags[len(val):]], val_tokens
    for d in removed:
        writers["train"].add(stream[position:out_start[d]])
        position = out_start[d] + lengths[d]
    writers["train"].add(stream[position:])
    for writer in writers.values():
        writer.flush()
    manifest = {
        "source": "@".join(SOURCE), "tokenizer": "@".join(TOKENIZER), "document": "BOS + tokens + EOS",
        "built_from": "the train shards of `python -m lorm.data.prepare_dolma3 tokenize` by "
                      "`python -m works.latent_feedback.data`",
        "seed": SEED, "input": {"documents": len(starts), "tokens": int(lengths.sum())},
        "distinct": {"documents": len(docs), "tokens": len(stream)},
        "decontamination": {"paloma": "@".join(PALOMA), "splits": ["val", "test"],
                            "min_words": MIN_WORDS, "paragraphs": paragraphs, "removed_documents": len(removed),
                            "removed_tokens": int(lengths[removed].sum()),
                            "val_documents_with_match": int(flags[:len(val)].sum())},
        "docs": {"val": len(val), "train": len(train) - len(removed)},
        "shards": {split: writer.counts for split, writer in writers.items()},
        "buckets": dict(zip(ix["names"], (np.bincount(ix["bucket"][docs], weights=lengths[docs],
                                                       minlength=len(ix["names"])) / len(stream)).tolist()))}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("src", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    table, paragraphs = paloma_table(AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1]))
    manifest = build(args.src, args.out, table, paragraphs, args.workers)
    print({key: manifest[key] for key in ("docs", "distinct", "decontamination")})


if __name__ == "__main__":
    main()
