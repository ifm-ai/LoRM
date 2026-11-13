"""A tiny training experiment: two train shards and one val shard of 17-token IDs, and a stand-in tokenizer."""

import numpy as np
import pytest

from lorm.data import TokenDataset
import lorm.train.main as training


class TinyTokenizer:
    all_special_ids = [0]

    def __len__(self):
        return 17

    def convert_ids_to_tokens(self, ids):
        return [f"t{i}" for i in ids]


@pytest.fixture
def config(tmp_path, monkeypatch):
    for name, values in [("a.bin", np.arange(31) % 17), ("b.bin", np.arange(31, 65) % 17),
                         ("val.bin", np.arange(35) % 17)]:
        (tmp_path / name).write_bytes(np.array(values, dtype="<u2").tobytes())
    sources = lambda *args, **kwargs: (TokenDataset([tmp_path / "a.bin", tmp_path / "b.bin"], 6),
                                       TokenDataset([tmp_path / "val.bin"], 6))
    monkeypatch.setattr(training, "load_data", sources)
    monkeypatch.setattr(training.AutoTokenizer, "from_pretrained", lambda *a, **k: TinyTokenizer())
    return {
        "seed": 91, "output_dir": str(tmp_path / "run"),
        "tokenizer": {"name": "local-test"},
        "data": {"name": "dolma3-mix-150b-llama2", "cache_dir": str(tmp_path), "sequence_length": 6},
        "model": {"vocab_size": 17, "dim": 16, "n_layers": 2, "n_heads": 2, "n_kv_heads": 1, "hidden_dim": 32,
                  "max_seq_len": 6, "attention_backend": "sdpa"},
        "optimizer": {"lr": 0.003, "weight_decay": 0.1},
        "train": {"max_steps": 6, "batch_size": 2, "gradient_accumulation": 2, "device": "cpu", "precision": "float32",
                  "warmup_steps": 1, "decay_steps": 2, "log_every": 1, "eval_every": 2, "checkpoint_every": 2},
        "eval": {"batch_size": 2, "num_sequences": 5},
        "wandb": {"mode": "disabled"},
    }
