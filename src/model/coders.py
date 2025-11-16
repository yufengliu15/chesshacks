"""Lightweight decoders for Bagz chess records."""

from __future__ import annotations

import struct
from typing import Tuple


def _read_varint(buffer: bytes, start: int) -> tuple[int, int]:
    """Reads a little-endian base-128 varint."""
    shift = 0
    value = 0
    index = start

    while True:
        byte = buffer[index]
        index += 1
        value |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            break
        shift += 7

    return value, index


def decode_action_value(record: bytes) -> Tuple[str, str, float]:
    """Decodes a Bagz record into (fen, move, win_probability)."""
    offset = 0
    fen_len, offset = _read_varint(record, offset)
    fen = record[offset : offset + fen_len].decode("utf-8")
    offset += fen_len

    move_len, offset = _read_varint(record, offset)
    move = record[offset : offset + move_len].decode("utf-8")
    offset += move_len

    win_prob = struct.unpack(">d", record[offset : offset + 8])[0]
    return fen, move, float(win_prob)

