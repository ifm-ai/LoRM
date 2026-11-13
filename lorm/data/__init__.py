"""Training and validation data: memory-mapped token shards (tokens.py) of the corpora in DATASETS."""

import re

import torch
from transformers import PreTrainedTokenizerBase

from .dolma3 import SOURCE, load_dolma3_llama2
from .tokens import TokenDataset

# data.name -> (source dataset, pinned revision, loader).
DATASETS = {"dolma3-mix-150b-llama2": (*SOURCE, load_dolma3_llama2),
            # Deduplicated, stratified and decontaminated (works/latent_feedback/data.py), in the same shard format.
            "dolma3-mix-150b-llama2-d3": (*SOURCE, load_dolma3_llama2)}


def load_data(tokenizer: PreTrainedTokenizerBase, *, name: str, num_sequences: int | None = None,
              **options) -> tuple[TokenDataset, TokenDataset]:
    """Training and validation datasets of DATASETS[name]. num_sequences keeps the first training sequences
    (TokenDataset.prefix), the tokens a run sees once per epoch; num_shards limits the training split only."""
    load = DATASETS[name][2]
    training = load(tokenizer, split="train", name=name, **options)
    if num_sequences is not None:
        training = training.prefix(num_sequences)
    validation = {key: value for key, value in options.items() if key != "num_shards"}
    return training, load(tokenizer, split="val", name=name, **validation)


def token_byte_lengths(tokenizer: PreTrainedTokenizerBase) -> torch.Tensor:
    """UTF-8 bytes of text per token ID, for bits per byte; special tokens such as EOT hold none.

    A byte-level BPE vocabulary spells every byte of a token as one character, so the
    lengths are exact even for tokens that end inside a multi-byte character. A SentencePiece
    vocabulary (Llama-2) writes spaces as U+2581 and bytes it cannot spell as <0xNN>, one byte each;
    the space it prefixes to a document counts as one byte.
    """
    special = set(tokenizer.all_special_ids)
    tokens = tokenizer.convert_ids_to_tokens(range(len(tokenizer)))
    if any("\u2581" in token for token in tokens):
        return torch.tensor([0 if index in special else 1 if re.fullmatch(r"<0x[0-9A-F]{2}>", token)
                             else len(token.replace("\u2581", " ").encode()) for index, token in enumerate(tokens)])
    return torch.tensor([0 if index in special else len(token) for index, token in enumerate(tokens)])
