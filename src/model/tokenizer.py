"""Tokenization utilities for chess FEN strings."""

from __future__ import annotations

import numpy as np


CHARACTERS = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "p",
    "n",
    "r",
    "k",
    "q",
    "P",
    "B",
    "N",
    "R",
    "Q",
    "K",
    "w",
    ".",
]

VOCAB_SIZE = len(CHARACTERS)
CHAR_TO_INDEX = {letter: index for index, letter in enumerate(CHARACTERS)}
_SPACES = frozenset({"1", "2", "3", "4", "5", "6", "7", "8"})
SEQUENCE_LENGTH = 77


def tokenize(fen: str) -> np.ndarray:
    """Returns an array of integer tokens representing the `fen` string."""
    board, side, castling, en_passant, halfmoves_last, fullmoves = fen.split(" ")
    board = side + board.replace("/", "")

    indices: list[int] = []

    for char in board:
        if char in _SPACES:
            indices.extend(int(char) * [CHAR_TO_INDEX["."]])
        else:
            indices.append(CHAR_TO_INDEX[char])

    if castling == "-":
        indices.extend(4 * [CHAR_TO_INDEX["."]])
    else:
        for char in castling:
            indices.append(CHAR_TO_INDEX[char])
        if len(castling) < 4:
            indices.extend((4 - len(castling)) * [CHAR_TO_INDEX["."]])

    if en_passant == "-":
        indices.extend(2 * [CHAR_TO_INDEX["."]])
    else:
        indices.extend(CHAR_TO_INDEX[char] for char in en_passant)

    halfmoves_last += "." * (3 - len(halfmoves_last))
    indices.extend(CHAR_TO_INDEX[x] for x in halfmoves_last)

    fullmoves += "." * (3 - len(fullmoves))
    indices.extend(CHAR_TO_INDEX[x] for x in fullmoves)

    if len(indices) != SEQUENCE_LENGTH:
        raise ValueError(
            f"Tokenization produced {len(indices)} tokens, "
            f"expected {SEQUENCE_LENGTH}."
        )

    return np.asarray(indices, dtype=np.int64)

