"""Material evaluation utilities for chess positions."""

from __future__ import annotations

import chess
import numpy as np

# Standard piece values (in centipawns)
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 310,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,  # King has no material value (cannot be captured)
}

# Normalization constant (queen value) for neural network input
MATERIAL_SCALE = 900.0

# Checkmate is worth more than all pieces combined
# Total material on board: 8 pawns + 2 knights + 2 bishops + 2 rooks + 1 queen = ~3900 centipawns per side
# Checkmate bonus should exceed total possible material advantage
CHECKMATE_VALUE = 10000  # 10,000 centipawns = ~11 queens
CHECKMATE_NORMALIZED = CHECKMATE_VALUE / MATERIAL_SCALE  # ~11.0 in normalized units


def get_material_count(board: chess.Board) -> dict[chess.PieceType, tuple[int, int]]:
    """Count pieces for each side.
    
    Args:
        board: Chess board position
    
    Returns:
        Dict mapping piece type to (white_count, black_count)
    """
    counts = {}
    for piece_type in PIECE_VALUES.keys():
        white_count = len(board.pieces(piece_type, chess.WHITE))
        black_count = len(board.pieces(piece_type, chess.BLACK))
        counts[piece_type] = (white_count, black_count)
    
    return counts


def calculate_material(board: chess.Board, color: chess.Color) -> int:
    """Calculate total material value for one side.
    
    Args:
        board: Chess board position
        color: Color to calculate material for
    
    Returns:
        Total material in centipawns
    """
    material = 0
    for piece_type, value in PIECE_VALUES.items():
        count = len(board.pieces(piece_type, color))
        material += count * value
    
    return material


def calculate_material_balance(board: chess.Board, perspective: chess.Color = chess.WHITE) -> int:
    """Calculate material balance from perspective of given color.
    
    Includes checkmate detection - checkmate is valued as more important than all pieces.
    
    Args:
        board: Chess board position
        perspective: Color perspective (positive = advantage for this color)
    
    Returns:
        Material balance in centipawns (positive = advantage, negative = disadvantage)
        Checkmate positions return ±CHECKMATE_VALUE
    """
    # Check for checkmate first - this overrides material calculation
    if board.is_checkmate():
        # Determine who is checkmated (it's the side to move)
        checkmated_color = board.turn
        
        # If we're checkmated, return large negative value
        # If opponent is checkmated, return large positive value
        if checkmated_color == perspective:
            return -CHECKMATE_VALUE
        else:
            return CHECKMATE_VALUE
    
    # Normal material calculation
    white_material = calculate_material(board, chess.WHITE)
    black_material = calculate_material(board, chess.BLACK)
    
    balance = white_material - black_material
    
    # Flip if from black's perspective
    if perspective == chess.BLACK:
        balance = -balance
    
    return balance


def get_normalized_material_balance(board: chess.Board, perspective: chess.Color = chess.WHITE) -> float:
    """Get material balance normalized for neural network input.
    
    Checkmate positions return ±CHECKMATE_NORMALIZED (~±11.0), which is larger than
    any possible material advantage, teaching the model that checkmate > material.
    
    Args:
        board: Chess board position
        perspective: Color perspective
    
    Returns:
        Normalized material balance
        - Regular positions: typically in [-4, 4] range (material advantage in queen units)
        - Checkmate: ±CHECKMATE_NORMALIZED (~±11.0)
    """
    balance = calculate_material_balance(board, perspective)
    
    # Don't clip checkmate values - let them be large to emphasize importance
    if abs(balance) >= CHECKMATE_VALUE:
        return balance / MATERIAL_SCALE  # ~±11.0 for checkmate
    else:
        # Clip normal material to ±5 queens (max realistic advantage)
        return np.clip(balance / MATERIAL_SCALE, -5.0, 5.0)


def get_material_features(board: chess.Board, perspective: chess.Color = chess.WHITE) -> np.ndarray:
    """Get detailed material features for neural network input.
    
    Args:
        board: Chess board position
        perspective: Color perspective
    
    Returns:
        Feature vector [7 values]:
            - Total material balance (normalized)
            - Pawn balance (normalized)
            - Knight balance (normalized)
            - Bishop balance (normalized) 
            - Rook balance (normalized)
            - Queen balance (normalized)
            - Piece count balance (just count of pieces, not value)
    """
    counts = get_material_count(board)
    
    features = []
    
    # Overall material balance
    total_balance = get_normalized_material_balance(board, perspective)
    features.append(total_balance)
    
    # Individual piece balances
    for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        white_count, black_count = counts[piece_type]
        balance = white_count - black_count
        
        # Flip if black's perspective
        if perspective == chess.BLACK:
            balance = -balance
        
        # Normalize by typical max count (e.g., 8 pawns, 2 rooks, etc.)
        if piece_type == chess.PAWN:
            max_count = 8
        elif piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK]:
            max_count = 2
        else:  # Queen
            max_count = 1
        
        normalized = np.clip(balance / max_count, -2.0, 2.0)
        features.append(normalized)
    
    # Total piece count balance (not value, just count)
    white_pieces = sum(counts[pt][0] for pt in PIECE_VALUES.keys() if pt != chess.KING)
    black_pieces = sum(counts[pt][1] for pt in PIECE_VALUES.keys() if pt != chess.KING)
    piece_count_balance = white_pieces - black_pieces
    if perspective == chess.BLACK:
        piece_count_balance = -piece_count_balance
    
    features.append(np.clip(piece_count_balance / 16.0, -1.0, 1.0))
    
    return np.array(features, dtype=np.float32)


def format_material_info(board: chess.Board) -> str:
    """Format material information as a human-readable string.
    
    Args:
        board: Chess board position
    
    Returns:
        Formatted string with material counts and balance
    """
    # Check for checkmate first
    if board.is_checkmate():
        checkmated = "White" if board.turn == chess.WHITE else "Black"
        winner = "Black" if board.turn == chess.WHITE else "White"
        return f"🏆 CHECKMATE - {winner} wins! ({checkmated} is checkmated)\n  Evaluation: {CHECKMATE_VALUE:+d} centipawns (infinite advantage)"
    
    counts = get_material_count(board)
    white_material = calculate_material(board, chess.WHITE)
    black_material = calculate_material(board, chess.BLACK)
    balance = white_material - black_material
    
    lines = []
    lines.append("Material Count:")
    
    for piece_type, (w_count, b_count) in counts.items():
        if piece_type == chess.KING:
            continue
        piece_name = chess.piece_name(piece_type).capitalize()
        value = PIECE_VALUES[piece_type]
        lines.append(f"  {piece_name:6s}: White {w_count} ({w_count * value:4d}) | Black {b_count} ({b_count * value:4d})")
    
    lines.append(f"\nTotal Material:")
    lines.append(f"  White: {white_material} centipawns")
    lines.append(f"  Black: {black_material} centipawns")
    lines.append(f"  Balance: {balance:+d} (White's favor)" if balance >= 0 else f"  Balance: {balance:+d} (Black's favor)")
    
    if abs(balance) >= 900:
        advantage = abs(balance) // 100 / 10
        color = "White" if balance > 0 else "Black"
        lines.append(f"  → {color} is up ~{advantage:.1f} pawns equivalent")
    
    return "\n".join(lines)

