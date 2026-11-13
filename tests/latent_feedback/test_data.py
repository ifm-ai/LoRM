"""D3 on a tiny synthetic source: three raw files in two buckets, written in 64-token shards so that documents span
shard boundaries, with one repeated document and one that holds a Paloma paragraph."""

import json

import numpy as np
import pytest

import lorm.data.prepare_dolma3 as prepare
import works.latent_feedback.data as d3

PARAGRAPH = "a b c d e f g h i j k l m"


def tokenize(texts, add_special_tokens=False):
    """Characters as tokens after a dummy prefix 3; a newline is token 13, as in Llama 2."""
    return {"input_ids": [[3] + [13 if c == "\n" else 20 + ord(c) % 200 for c in text] for text in texts]}


def documents(*paths):
    t = np.concatenate([d3.read(path) for path in paths]).tolist()
    starts = [i for i, x in enumerate(t) if x == prepare.BOS] + [len(t)]
    return [tuple(t[a:b]) for a, b in zip(starts[:-1], starts[1:])]


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "SHARD_TOKENS", 64)
    monkeypatch.setattr(d3, "VAL_TOKENS", 50)
    rng = np.random.default_rng(1)
    doc = lambda: (1, *rng.integers(20, 220, rng.integers(3, 20)).tolist(), 2)
    files = {"data/a/0.jsonl.zst": [doc() for _ in range(8)], "data/b/0.jsonl.zst": [doc() for _ in range(6)],
             "data/a/1.jsonl.zst": [doc() for _ in range(8)]}
    repeated = files["data/a/0.jsonl.zst"][2]
    contaminated = (1, 40, 41, 13, *tokenize([PARAGRAPH])["input_ids"][0][1:], 13, 42, 2)
    files["data/b/0.jsonl.zst"][1:1] = [repeated, contaminated]
    files["data/a/1.jsonl.zst"].append(repeated)
    writer = prepare.ShardWriter(tmp_path, "train")
    for docs in files.values():
        writer.add(np.array([x for doc in docs for x in doc], np.uint16))
    writer.flush()
    order = [{"file": name, "split": "train", "docs": len(docs), "tokens": sum(map(len, docs))}
             for name, docs in files.items()]
    (tmp_path / "manifest.json").write_text(json.dumps({"order": order, "shards": {"train": writer.counts}}))
    return tmp_path, [doc for docs in files.values() for doc in docs], repeated, contaminated


def test_d3(source, tmp_path):
    src, docs, repeated, contaminated = source
    ix = d3.index(src, workers=2)
    assert [tuple(d) for d in np.split(np.concatenate([d3.read(p) for p in ix["paths"]]), ix["starts"][1:])] == docs
    val, train = d3.stratified(ix["hashes"], ix["lengths"], ix["bucket"])
    first = [i for i, doc in enumerate(docs) if doc not in docs[:i]]
    assert sorted([*val, *train]) == first  # one copy of each document, the first

    table, count = d3.paragraph_hashes(tokenize, ["short line\n" + PARAGRAPH + "\nmore"])
    manifest = d3.build(src, tmp_path / "out", table, {"test": count}, workers=2)
    out_val = documents(tmp_path / "out" / "dolma3_val_000000.bin")
    out_train = documents(*sorted((tmp_path / "out").glob("dolma3_train_*.bin")))
    assert out_val == [docs[i] for i in val]
    assert sum(map(len, out_val)) <= d3.VAL_TOKENS < sum(map(len, out_val)) + len(docs[train[0]])
    assert out_train == [docs[i] for i in train if docs[i] != contaminated]
    assert contaminated in [docs[i] for i in train] and manifest["decontamination"]["removed_documents"] == 1
    assert [*out_val, *out_train].count(repeated) == 1
    assert manifest["shards"]["train"][:-1] == [64] * (len(manifest["shards"]["train"]) - 1)
