"""Read an ordered stream of pretokenized binary shards without changing tokens.

Shards contain little-endian uint16 or uint32 IDs after an optional byte offset.
Sequences can cross shards; only the stream's final incomplete sequence is dropped.
Labels equal input_ids; lorm.train.loss.causal_loss shifts them. The dataset has no cursor:
checkpoint StatefulDataLoader.state_dict(), then load_state_dict() before iterating
the restored loader, with unchanged files and loader settings. Memory maps open lazily.
`prefix` keeps the stream's first sequences, so runs that repeat data repeat the same tokens.
"""

from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from itertools import accumulate
from operator import index as integer_index
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TokenDataset(Dataset):
    def __init__(
        self,
        files: Sequence[str | Path],
        sequence_length: int,
        *,
        dtype: str = "uint16",
        offset: int = 0,
    ) -> None:
        if isinstance(files, (str, Path)) or not files:
            raise ValueError("Provide a nonempty ordered sequence of token files.")
        if type(sequence_length) is not int or sequence_length < 2:
            raise ValueError("sequence_length must be an integer of at least 2.")
        if dtype not in ("uint16", "uint32"):
            raise ValueError("dtype must be uint16 or uint32 (little-endian).")
        self.dtype = np.dtype("<u2" if dtype == "uint16" else "<u4")
        if type(offset) is not int or offset < 0 or offset % self.dtype.itemsize:
            raise ValueError("offset must be a nonnegative, token-aligned byte count.")
        self.files = tuple(Path(path).resolve() for path in files)
        if len(set(self.files)) != len(self.files):
            raise ValueError("Token files must not be repeated.")
        self.sequence_length = sequence_length
        self.offset = offset
        counts = []
        for path in self.files:
            size = path.stat().st_size - offset
            if size <= 0 or size % self.dtype.itemsize:
                raise ValueError(f"Invalid token payload size: {path}")
            counts.append(size // self.dtype.itemsize)
        self._ends = tuple(accumulate(counts))
        self._arrays = {}
        if len(self) == 0:
            raise ValueError("Token files cannot fill one sequence.")

    def __len__(self) -> int:
        return self._ends[-1] // self.sequence_length

    @property
    def num_tokens(self) -> int:
        return self._ends[-1]

    def prefix(self, num_sequences: int) -> "TokenDataset":
        """The first `num_sequences` sequences of the stream, independent of any seed."""
        if type(num_sequences) is not int or not 0 < num_sequences <= len(self):
            raise ValueError(f"num_sequences must be a positive integer of at most {len(self)}.")
        end = num_sequences * self.sequence_length
        kept = bisect_left(self._ends, end) + 1
        view = copy(self)
        view.files, view._ends, view._arrays = self.files[:kept], self._ends[:kept - 1] + (end,), {}
        return view

    def warm(self, part: int = 0, parts: int = 1, *, threads: int = 4) -> tuple[Path, ...]:
        """Read every `parts`-th file from `part` once, sequentially, into the page cache; return them.

        Memory-mapped reads of an uncached shard on a network filesystem can wait for each page, while one
        sequential read fills the cache much faster. Ranks on one node warm disjoint parts.
        """
        files = self.files[part::parts]
        def read(path):
            with open(path, "rb", buffering=0) as file:
                while file.read(1 << 24):
                    pass
        with ThreadPoolExecutor(threads) as pool:
            list(pool.map(read, files))
        return files

    def read(self, start: int, count: int) -> torch.Tensor:
        """Tokens start..start + count of the stream, across shard boundaries."""
        if not 0 <= start <= start + count <= self._ends[-1]:
            raise IndexError(start)
        shard = bisect_right(self._ends, start)
        tokens = np.empty(count, dtype=np.int64)
        written = 0
        while written < count:
            begin = self._ends[shard - 1] if shard else 0
            if shard not in self._arrays:
                self._arrays[shard] = np.memmap(
                    self.files[shard], mode="r", dtype=self.dtype, offset=self.offset,
                    shape=(self._ends[shard] - begin,),
                )
            size = min(count - written, self._ends[shard] - start)
            tokens[written : written + size] = self._arrays[shard][start - begin : start - begin + size]
            written += size
            start += size
            shard += 1
        return torch.from_numpy(tokens)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        index = integer_index(index)
        if not 0 <= index < len(self):
            raise IndexError(index)
        input_ids = self.read(index * self.sequence_length, self.sequence_length)
        return {"input_ids": input_ids, "labels": input_ids.clone()}
