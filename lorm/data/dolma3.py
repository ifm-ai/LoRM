"""Dolma 3 mix shards tokenized with the Llama 2 tokenizer (lorm/data/prepare_dolma3.py); tokens are never re-encoded.

Each document is BOS + tokens + EOS. The source files were shuffled before tokenizing, so a prefix of the training
shards is a random sample of the mix. `cache_dir` holds the shards, named dolma3_{train,val}_NNNNNN.bin in the
modded-nanogpt format.
"""

from pathlib import Path

from transformers import PreTrainedTokenizerBase

from .tokens import TokenDataset

SOURCE = ("allenai/dolma3_mix-150B-1025", "afa92bfb22366821c5e6cd427cdd036b34b713ef")


def load_dolma3_llama2(tokenizer: PreTrainedTokenizerBase, *, cache_dir: str | Path, sequence_length: int,
                       split: str = "train", num_shards: int | None = None, name: str = "dolma3-mix-150b-llama2",
                       revision: str | None = None) -> TokenDataset:
    """The first num_shards shards of `split` (all when None); revision must be the tokenized source's."""
    if len(tokenizer) != 32000 or tokenizer.bos_token_id != 1 or tokenizer.eos_token_id != 2:
        raise ValueError("These shards hold Llama 2 tokens: vocab 32000, BOS 1, EOS 2.")
    if revision not in (None, SOURCE[1]):
        raise ValueError(f"The shards were tokenized from {SOURCE[0]} at {SOURCE[1]}.")
    paths = sorted(Path(cache_dir).glob(f"dolma3_{split}_*.bin"))
    if not paths or (num_shards is not None and num_shards > len(paths)):
        raise ValueError(f"{cache_dir} does not hold the requested {split} shards.")
    return TokenDataset(paths[:num_shards], sequence_length, offset=1024)
