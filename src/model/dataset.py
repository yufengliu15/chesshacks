"""PyTorch dataset utilities for Bagz chess records."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset

# Handle different import paths (local vs Modal environment)
try:
    from ..bagz import BagFileReader
except (ImportError, ValueError):
    from bagz import BagFileReader

from . import tokenizer
from . import moves
from .coders import decode_action_value


BOARD_VOCAB_SIZE = tokenizer.VOCAB_SIZE
SEQUENCE_LENGTH = tokenizer.SEQUENCE_LENGTH + 1  # board + action token
VOCAB_SIZE = BOARD_VOCAB_SIZE + moves.NUM_ACTIONS


def fen_to_tokens(fen: str) -> np.ndarray:
    board_tokens = tokenizer.tokenize(fen)
    if board_tokens.shape[0] != tokenizer.SEQUENCE_LENGTH:
        raise ValueError("Unexpected tokenized board length.")
    return board_tokens


def action_to_token(move: str) -> int | None:
    action_id = moves.MOVE_TO_ACTION.get(move)
    if action_id is None:
        return None
    return BOARD_VOCAB_SIZE + action_id


class ActionValueIterableDataset(IterableDataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Streams action-value records directly from Bagz files."""

    def __init__(
        self,
        bag_paths: Sequence[Path],
        *,
        max_records: int | None = None,
        min_win_prob: float | None = None,
        max_win_prob: float | None = None,
        shuffle_files: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self._bag_paths = list(bag_paths)
        self._max_records = max_records
        self._min_win_prob = min_win_prob
        self._max_win_prob = max_win_prob
        self._shuffle_files = shuffle_files
        self._seed = seed

    def _iter_files(self) -> Iterable[Path]:
        paths = list(self._bag_paths)
        if self._shuffle_files:
            rng = np.random.default_rng(self._seed)
            rng.shuffle(paths)
        return paths

    def _yield_samples(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        produced = 0
        for path in self._iter_files():
            reader = BagFileReader(str(path))
            for record in reader:
                fen, move, win_prob = decode_action_value(record)
                if (
                    self._min_win_prob is not None
                    and win_prob < self._min_win_prob
                ):
                    continue
                if (
                    self._max_win_prob is not None
                    and win_prob > self._max_win_prob
                ):
                    continue

                action_token = action_to_token(move)
                if action_token is None:
                    continue

                board_tokens = fen_to_tokens(fen)
                sequence = np.concatenate(
                    [board_tokens, np.asarray([action_token], dtype=np.int64)]
                ).astype(np.int64)
                target = np.asarray(win_prob, dtype=np.float32)

                produced += 1
                yield (
                    torch.from_numpy(sequence),
                    torch.from_numpy(target),
                )
                if self._max_records is not None and produced >= self._max_records:
                    return

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        return self._yield_samples()

