"""Helpers for mapping chess moves to action ids."""

from __future__ import annotations

import chess

from typing import Dict, Tuple

_FILES = ["a", "b", "c", "d", "e", "f", "g", "h"]


def _compute_move_tables() -> Tuple[Dict[str, int], Dict[int, str]]:
    all_moves: list[str] = []
    board = chess.BaseBoard.empty()

    for square in range(64):
        attacks: list[int] = []
        board.set_piece_at(square, chess.Piece.from_symbol("Q"))
        attacks.extend(board.attacks(square))

        board.set_piece_at(square, chess.Piece.from_symbol("N"))
        attacks.extend(board.attacks(square))
        board.remove_piece_at(square)

        for target in attacks:
            all_moves.append(chess.square_name(square) + chess.square_name(target))

    promotion_moves: list[str] = []
    for rank, next_rank in [("2", "1"), ("7", "8")]:
        for file_index, file_letter in enumerate(_FILES):
            base = f"{file_letter}{rank}"
            straight = f"{base}{file_letter}{next_rank}"
            promotion_moves.extend(straight + piece for piece in "qrbn")

            if file_letter > "a":
                left = f"{base}{_FILES[file_index - 1]}{next_rank}"
                promotion_moves.extend(left + piece for piece in "qrbn")
            if file_letter < "h":
                right = f"{base}{_FILES[file_index + 1]}{next_rank}"
                promotion_moves.extend(right + piece for piece in "qrbn")

    all_moves.extend(promotion_moves)

    move_to_action: Dict[str, int] = {}
    action_to_move: Dict[int, str] = {}

    for action, move in enumerate(all_moves):
        move_to_action[move] = action
        action_to_move[action] = move

    return move_to_action, action_to_move


MOVE_TO_ACTION, ACTION_TO_MOVE = _compute_move_tables()
NUM_ACTIONS = len(MOVE_TO_ACTION)

