"""Self-play game generation for reinforcement learning.

Generates chess games where the model plays against Stockfish, collecting
training data in the form of (position, move, game_outcome) for the model's moves only.
"""

from __future__ import annotations

import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import chess
import modal
import numpy as np
import torch

# Modal app
app = modal.App("chess-selfplay")

# Modal volumes
checkpoints_vol = modal.Volume.from_name("chess-checkpoints", create_if_missing=True)
selfplay_data_vol = modal.Volume.from_name("chess-selfplay-data", create_if_missing=True)

# Get the src directory path
src_dir = Path(__file__).parent

# Container image
image = (
    modal.Image.debian_slim()
    .apt_install("stockfish")  # Install Stockfish engine
    .pip_install([
        "torch>=2.0.0",
        "numpy",
        "python-chess",
        "tqdm",
        "zstandard",  # Required by bagz
    ])
    .env({"CACHE_BUST": "2024-11-15-v7"})  # Force rebuild when model changes
    .add_local_file(str(src_dir / "bagz.py"), remote_path="/root/bagz.py")
    .add_local_dir(str(src_dir / "model"), remote_path="/root/model")
)


@dataclass
class SelfPlayConfig:
    """Configuration for self-play game generation."""
    temperature: float = 0.1  # Temperature for move selection (lower = more deterministic)
    max_moves: int = 50  # Maximum moves per game (shortened to encourage decisive play)
    num_games: int = 100  # Number of games to generate
    exploration_moves: int = 30  # Use stochastic policy for first N moves
    checkpoint_path: str = "/checkpoints/base_model.pt"
    output_dir: str = "/selfplay_data"
    device: str = "cuda"
    
    # Draw outcome values (0.0 = bad as loss, 1.0 = good as win)
    # Lower = harsher penalty, higher = lighter penalty
    natural_draw_value: float = 0.15  # Value for natural draws (stalemate, 50-move rule)
    repetition_draw_value: float = 0.05  # Value for repetition draws (worst - almost same as loss)
    max_moves_draw_value: float = 0.10  # Value for hitting move limit (indecisive play)
    normal_draw_value: float = 0.10  # Value for other draws
    
    checkmate_bonus: float = 1.0  # Value for checkmate wins (always 1.0)
    
    # Stockfish settings
    stockfish_path: str = "/usr/games/stockfish"  # Path to stockfish binary
    stockfish_depth: int = 4  # Stockfish search depth (opponent - higher = stronger)
    stockfish_time_limit: float = 0.1  # Time limit per move in seconds
    model_plays_white: bool = True  # alternate for variety
    
    # Move-by-move evaluation settings
    use_move_evaluation: bool = True  # Use second Stockfish to evaluate each move
    eval_stockfish_depth: int = 6  # Depth for move evaluation (higher = more accurate)
    eval_weight: float = 0.7  # Weight for move eval vs game outcome (0.7 = 70% eval, 30% outcome)
    eval_scale: float = 300.0  # Centipawns to normalize (±300cp = ±1.0 in model terms)
    
    # MCTS settings
    use_mcts: bool = False  # Whether to use MCTS for model's move selection
    mcts_simulations: int = 100  # Number of MCTS simulations (if enabled)
    mcts_c_puct: float = 1.5  # Exploration constant for MCTS
    mcts_max_depth: int = 5  # Max depth per simulation (0 = unlimited)


def _write_varint(value: int) -> bytes:
    """Writes a little-endian base-128 varint."""
    result = []
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            byte |= 0x80
        result.append(byte)
        if not value:
            break
    return bytes(result)


def encode_action_value(fen: str, move: str, win_prob: float) -> bytes:
    """Encodes a (fen, move, win_probability) tuple into a Bagz record."""
    fen_bytes = fen.encode("utf-8")
    move_bytes = move.encode("utf-8")
    win_prob_bytes = struct.pack(">d", win_prob)
    
    return (
        _write_varint(len(fen_bytes)) + fen_bytes +
        _write_varint(len(move_bytes)) + move_bytes +
        win_prob_bytes
    )


class SimpleBagWriter:
    """Simple bag file writer for self-play data."""
    
    def __init__(self, filename: str):
        self.filename = filename
        self._records_file = open(filename + ".tmp", "wb")
        self._limits_file = open(filename + ".limits.tmp", "wb")
        self._count = 0
    
    def write(self, data: bytes) -> None:
        """Write a record."""
        if data:
            self._records_file.write(data)
        self._limits_file.write(struct.pack('<q', self._records_file.tell()))
        self._count += 1
    
    def close(self) -> None:
        """Close and finalize the bag file."""
        import shutil
        
        # Close write handles
        self._limits_file.close()
        
        # Reopen limits file in read mode and append to records
        with open(self.filename + ".limits.tmp", "rb") as limits_read:
            shutil.copyfileobj(limits_read, self._records_file)
        
        self._records_file.close()
        
        # Remove temporary limits file
        os.unlink(self.filename + ".limits.tmp")
        
        # Rename records file to final name
        os.rename(self.filename + ".tmp", self.filename)
        
        print(f"✅ Wrote {self._count} records to {self.filename}")


class SelfPlayEngine:
    """Engine for self-play using the transformer model."""
    
    def __init__(self, model, device: str = "cuda", use_mcts: bool = False, 
                 mcts_simulations: int = 100, mcts_c_puct: float = 1.5, 
                 mcts_max_depth: int = 5):
        self.model = model
        self.device = torch.device(device)
        self.model.eval()
        self.use_mcts = use_mcts
        
        # Initialize MCTS if requested
        if use_mcts:
            from model.mcts import MCTS, MCTSConfig
            mcts_config = MCTSConfig(
                num_simulations=mcts_simulations,
                c_puct=mcts_c_puct,
                temperature=0.3,  # Some randomness for diversity
                max_depth=mcts_max_depth if mcts_max_depth > 0 else None,
                dirichlet_alpha=0.3,
                dirichlet_epsilon=0.25,
            )
            self.mcts = MCTS(model, device, mcts_config)
            print(f"🌳 MCTS enabled for self-play: {mcts_simulations} sims, "
                  f"c_puct={mcts_c_puct}, max_depth={mcts_max_depth}")
        else:
            self.mcts = None
    
    def _fen_to_tokens(self, fen: str) -> torch.Tensor:
        """Convert FEN to token tensor."""
        from model.dataset import fen_to_tokens
        tokens = fen_to_tokens(fen)
        return torch.from_numpy(tokens.astype("int64"))
    
    def _action_to_token(self, move: str) -> Optional[int]:
        """Convert move to action token."""
        from model.dataset import action_to_token
        return action_to_token(move)
    
    def choose_move(
        self, 
        board: chess.Board, 
        temperature: float = 1.0
    ) -> tuple[chess.Move | None, dict[str, float]]:
        """Choose a move using the model with optional temperature sampling.
        
        If MCTS is enabled, uses MCTS for move selection.
        Otherwise, uses direct value evaluation.
        """
        if self.use_mcts and self.mcts is not None:
            # Use MCTS for stronger play
            move, mcts_info = self.mcts.search(board, add_noise=True)  # Add noise for exploration
            
            # Extract move probabilities from visit counts
            move_probs = {}
            for key, value in mcts_info.items():
                if key.endswith('_visits'):
                    move_uci = key[:-7]  # Remove '_visits' suffix
                    move_probs[move_uci] = value
            
            return move, move_probs
        
        # Original direct evaluation approach
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None, {}
        
        # Get FEN tokens
        fen_tokens = self._fen_to_tokens(board.fen())
        
        # Score all legal moves
        sequences = []
        kept_moves = []
        for move in legal_moves:
            move_token = self._action_to_token(move.uci())
            if move_token is None:
                continue
            
            seq = torch.cat([
                fen_tokens,
                torch.tensor([move_token], dtype=torch.int64)
            ])
            sequences.append(seq)
            kept_moves.append(move)
        
        if not sequences:
            return legal_moves[0], {"random": 1.0}
        
        # Batch inference
        batch = torch.stack(sequences).to(self.device)
        with torch.no_grad():
            logits = self.model(batch)
            # Convert to win probabilities
            win_probs = torch.sigmoid(logits).cpu().numpy()
        
        # Select move based on temperature
        if temperature < 0.01:
            # Greedy selection
            best_idx = np.argmax(win_probs)
            chosen_move = kept_moves[best_idx]
        else:
            # Temperature-based sampling
            # Use softmax on win probabilities scaled by temperature
            scaled_probs = win_probs / temperature
            exp_probs = np.exp(scaled_probs - np.max(scaled_probs))  # numerical stability
            probs = exp_probs / exp_probs.sum()
            
            chosen_idx = np.random.choice(len(kept_moves), p=probs)
            chosen_move = kept_moves[chosen_idx]
        
        # Return move and probability distribution
        move_probs = {move.uci(): float(prob) for move, prob in zip(kept_moves, win_probs)}
        return chosen_move, move_probs


def centipawns_to_value(cp_eval: float, model_plays_white: bool, scale: float = 300.0) -> float:
    """Convert centipawn evaluation to a value in [0, 1].
    
    Args:
        cp_eval: Evaluation in centipawns from white's perspective
        model_plays_white: Whether model is playing white
        scale: Centipawns for normalization (default 300 = 3 pawns)
    
    Returns:
        Value in [0, 1] from model's perspective
        1.0 = winning position, 0.5 = equal, 0.0 = losing
    """
    # Convert to model's perspective
    if not model_plays_white:
        cp_eval = -cp_eval
    
    # Normalize using sigmoid function
    # ±scale centipawns maps to approximately [0.1, 0.9]
    value = 1.0 / (1.0 + np.exp(-cp_eval / scale))
    
    return np.clip(value, 0.0, 1.0)


def play_game(engine: SelfPlayEngine, config: SelfPlayConfig) -> tuple[list, str, str]:
    """Play a game between the model and Stockfish, return move history + outcome.
    
    Only the model's moves are recorded for training data.
    
    Returns:
        moves: List of (fen, move_uci, move_probs) tuples (model's moves only)
        outcome: "1-0" (white wins), "0-1" (black wins), or "1/2-1/2" (draw)
        game_type: "checkmate", "stalemate", "repetition", "fifty_move", "max_moves", or "normal"
    """
    import chess.engine
    
    board = chess.Board()
    moves = []  # Only stores model's moves (now with evaluations)
    game_type = "normal"
    
    # Initialize Stockfish engines
    # Engine 1: Opponent
    stockfish_opponent = chess.engine.SimpleEngine.popen_uci(config.stockfish_path)
    
    # Engine 2: Move evaluator (optional)
    stockfish_eval = None
    if config.use_move_evaluation:
        stockfish_eval = chess.engine.SimpleEngine.popen_uci(config.stockfish_path)
    
    try:
        for move_num in range(config.max_moves):
            # Determine whose turn it is
            is_model_turn = (board.turn == chess.WHITE) == config.model_plays_white
            
            # Store position before move
            fen_before = board.fen()
            
            if is_model_turn:
                # Model's turn
                # Use temperature for exploration in early game
                if move_num < config.exploration_moves:
                    temperature = config.temperature
                else:
                    temperature = 0.01  # Near-greedy in late game
                
                # Choose move
                move, move_probs = engine.choose_move(board, temperature=temperature)
                
                if move is None:
                    # Game over (checkmate or stalemate)
                    break
                
                # Make the move
                board.push(move)
                
                # Evaluate the position after model's move (if enabled)
                move_eval_pawns = 0.0
                if stockfish_eval is not None:
                    try:
                        eval_info = stockfish_eval.analyse(
                            board,
                            chess.engine.Limit(depth=config.eval_stockfish_depth)
                        )
                        score = eval_info['score'].white()
                        if score.is_mate():
                            # Mate score
                            mate_in = score.mate()
                            move_eval_pawns = 100.0 if mate_in > 0 else -100.0
                        else:
                            # Centipawn score - convert to pawns
                            move_eval_pawns = score.score() / 100.0
                    except Exception as e:
                        print(f"Warning: Stockfish eval failed: {e}")
                        move_eval_pawns = 0.0
                
                # Undo the move (we'll push it again below)
                board.pop()
                
                # Store move data with evaluation
                moves.append((fen_before, move.uci(), move_probs, move_eval_pawns))
            else:
                # Stockfish opponent's turn
                result = stockfish_opponent.play(
                    board,
                    chess.engine.Limit(
                        depth=config.stockfish_depth,
                        time=config.stockfish_time_limit
                    )
                )
                move = result.move
                
                if move is None:
                    # Game over
                    break
            
            # Make move
            board.push(move)
            
            # Check for game over
            if board.is_game_over():
                # Determine type of game over
                if board.is_checkmate():
                    game_type = "checkmate"
                elif board.is_stalemate():
                    game_type = "stalemate"
                break
            
            # Check for draw by repetition
            if board.is_repetition(3):
                game_type = "repetition"
                break
            
            # Check for fifty-move rule
            if board.can_claim_fifty_moves():
                game_type = "fifty_move"
                break
        
        # If we hit max moves without conclusion
        if move_num >= config.max_moves - 1 and not board.is_game_over():
            game_type = "max_moves"
        
        # Determine outcome
        result = board.result(claim_draw=True)
        
    finally:
        # Always close Stockfish engines
        stockfish_opponent.quit()
        if stockfish_eval is not None:
            stockfish_eval.quit()
    
    return moves, result, game_type


def assign_outcomes(moves: list, result: str, game_type: str, config: SelfPlayConfig) -> list[tuple[str, str, float]]:
    """Assign win probabilities to moves based on game outcome and move evaluations.
    
    NOTE: This function only receives the MODEL'S moves (not Stockfish's moves).
    We need to assign values from the model's perspective.
    
    Args:
        moves: List of (fen, move_uci, move_probs, eval) tuples (MODEL'S MOVES ONLY)
               eval is Stockfish evaluation in pawns from white's perspective
        result: "1-0", "0-1", or "1/2-1/2"
        game_type: "checkmate", "stalemate", "repetition", "fifty_move", "max_moves", or "normal"
        config: Self-play configuration (includes model_plays_white, use_move_evaluation)
    
    Returns:
        List of (fen, move_uci, win_prob) tuples
    
    Note:
        All values are clamped to [0, 1] to prevent loss explosion in BCEWithLogitsLoss.
        If use_move_evaluation is True, blends Stockfish eval with game outcome.
    """
    outcomes = []
    
    # Determine if model won, lost, or drew
    model_plays_white = config.model_plays_white
    
    if result == "1-0":
        # White won
        model_won = model_plays_white
    elif result == "0-1":
        # Black won
        model_won = not model_plays_white
    else:
        # Draw
        model_won = None
    
    # Calculate game outcome value
    if model_won is True:
        outcome_value = 1.0
    elif model_won is False:
        outcome_value = 0.0
    else:
        # Draw - assign value based on draw type
        if game_type == "repetition":
            outcome_value = config.repetition_draw_value
        elif game_type == "max_moves":
            outcome_value = config.max_moves_draw_value
        elif game_type in ["stalemate", "fifty_move"]:
            outcome_value = config.natural_draw_value
        else:
            outcome_value = config.normal_draw_value
    
    # Assign values to each move
    for move_data in moves:
        if len(move_data) == 4:
            # New format: (fen, move_uci, move_probs, eval)
            fen, move_uci, _, stockfish_eval_pawns = move_data
            
            if config.use_move_evaluation and stockfish_eval_pawns != 0.0:
                # Convert Stockfish eval to value [0, 1]
                eval_value = centipawns_to_value(
                    stockfish_eval_pawns * 100,  # Convert pawns to centipawns
                    model_plays_white,
                    config.eval_scale
                )
                
                # Blend eval with game outcome
                # eval_weight controls the blend (0.7 = 70% eval, 30% outcome)
                final_value = (config.eval_weight * eval_value + 
                             (1 - config.eval_weight) * outcome_value)
            else:
                # No evaluation available, use game outcome only
                final_value = outcome_value
        else:
            # Old format: (fen, move_uci, move_probs) - backward compatibility
            fen, move_uci, _ = move_data
            final_value = outcome_value
        
        # CRITICAL: Clamp value to [0, 1] to prevent loss explosion
        final_value = max(0.0, min(1.0, final_value))
        
        outcomes.append((fen, move_uci, final_value))
    
    return outcomes


@app.function(
    image=image,
    volumes={
        "/checkpoints": checkpoints_vol,
        "/selfplay_data": selfplay_data_vol,
    },
    gpu="H100",
    timeout=3600 * 4,  # 4 hours (longer for MCTS)
    memory=32000,
)
def generate_selfplay_games(
    num_games: int = 100,
    checkpoint_step: int = 22000,
    temperature: float = 0.3,
    exploration_moves: int = 30,
    output_shard_idx: int = 0,
    stockfish_depth: int = 4,
    stockfish_time_limit: float = 0.1,
    use_mcts: bool = False,
    mcts_simulations: int = 100,
    mcts_c_puct: float = 1.5,
    mcts_max_depth: int = 5,
) -> dict:
    """Generate games where the model plays against Stockfish.
    
    Args:
        num_games: Number of games to generate
        checkpoint_step: Which checkpoint to use
        temperature: Temperature for move selection
        exploration_moves: Number of moves to use stochastic policy
        output_shard_idx: Shard index for output file
        stockfish_depth: Stockfish search depth (higher = stronger)
        stockfish_time_limit: Time limit per Stockfish move in seconds
        use_mcts: Whether to use MCTS for model's moves
        mcts_simulations: Number of MCTS simulations (if enabled)
        mcts_c_puct: Exploration constant for MCTS
        mcts_max_depth: Max depth per simulation (0 = unlimited)
    
    Returns:
        Statistics about the generated games
    """
    # Setup Python path for imports
    import sys
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")
    
    print(f"🎮 Starting self-play generation: {num_games} games")
    
    # Load model
    checkpoint_path = f"/checkpoints/checkpoint_step_{checkpoint_step}.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Import model
    from model.transformer_model import ActionValueTransformer, TransformerConfig
    
    # Load checkpoint (always load to CPU first to avoid device issues)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get model config
    if "config" in checkpoint and isinstance(checkpoint["config"], dict):
        if "model" in checkpoint["config"]:
            model_cfg = checkpoint["config"]["model"]
        else:
            model_cfg = checkpoint["config"]
        model_config = TransformerConfig(**model_cfg)
    else:
        raise ValueError("Checkpoint missing config")
    
    # Create and load model
    model = ActionValueTransformer(model_config).to(device)
    
    # Get state dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        raise ValueError("Checkpoint missing model state")
    
    # Handle backward compatibility: remap 'head' to 'value_head'
    # Old checkpoints have 'head.X.weight', new model expects 'value_head.X.weight'
    # We keep BOTH to ensure compatibility with the alias self.head = self.value_head
    if any(k.startswith("head.") for k in state_dict.keys()):
        print("🔄 Converting old checkpoint format (head → value_head)")
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k] = v  # Keep original
            if k.startswith("head."):
                new_key = k.replace("head.", "value_head.", 1)
                new_state_dict[new_key] = v  # Also add value_head version
        state_dict = new_state_dict
    
    # Load with strict=False to allow missing keys from the alias
    model.load_state_dict(state_dict, strict=False)
    
    print(f"✅ Loaded model from step {checkpoint_step}")
    
    # Create self-play engine (with optional MCTS)
    engine = SelfPlayEngine(
        model, 
        device=str(device),
        use_mcts=use_mcts,
        mcts_simulations=mcts_simulations,
        mcts_c_puct=mcts_c_puct,
        mcts_max_depth=mcts_max_depth,
    )
    
    # Generate games and collect data
    all_training_data = []
    game_results = {
        "model_wins": 0, 
        "stockfish_wins": 0, 
        "draws": 0,
        "checkmates": 0,
        "repetitions": 0,
        "max_moves": 0,
        "model_as_white": 0,
        "model_as_black": 0,
    }
    game_lengths = []
    
    print(f"🎲 Generating {num_games} games vs Stockfish (depth {stockfish_depth})...")
    
    for game_idx in range(num_games):
        try:
            # Alternate colors for variety: model plays white on even games, black on odd
            model_plays_white = (game_idx % 2 == 0)
            
            # Create config for this game
            config = SelfPlayConfig(
                temperature=temperature,
                exploration_moves=exploration_moves,
                num_games=num_games,
                stockfish_depth=stockfish_depth,
                stockfish_time_limit=stockfish_time_limit,
                model_plays_white=model_plays_white,
                use_mcts=use_mcts,
                mcts_simulations=mcts_simulations,
                mcts_c_puct=mcts_c_puct,
                mcts_max_depth=mcts_max_depth,
            )
            
            moves, result, game_type = play_game(engine, config)
            training_data = assign_outcomes(moves, result, game_type, config)
            all_training_data.extend(training_data)
            
            # Update statistics
            game_lengths.append(len(moves))
            
            # Track color statistics
            if model_plays_white:
                game_results["model_as_white"] += 1
            else:
                game_results["model_as_black"] += 1
            
            # Track results from model's perspective
            if result == "1-0":
                if model_plays_white:
                    game_results["model_wins"] += 1
                else:
                    game_results["stockfish_wins"] += 1
            elif result == "0-1":
                if model_plays_white:
                    game_results["stockfish_wins"] += 1
                else:
                    game_results["model_wins"] += 1
            else:
                game_results["draws"] += 1
            
            # Track game types
            if game_type == "checkmate":
                game_results["checkmates"] += 1
            elif game_type == "repetition":
                game_results["repetitions"] += 1
            elif game_type == "max_moves":
                game_results["max_moves"] += 1
            
            if (game_idx + 1) % 10 == 0:
                win_rate = game_results["model_wins"] / (game_idx + 1) * 100
                print(f"  Generated {game_idx + 1}/{num_games} games "
                      f"(model win rate: {win_rate:.1f}%, "
                      f"avg length: {np.mean(game_lengths):.1f} moves, "
                      f"checkmates: {game_results['checkmates']}, "
                      f"repetitions: {game_results['repetitions']})")
        
        except Exception as e:
            print(f"⚠️  Error in game {game_idx}: {e}")
            continue
    
    # Write to bag file
    output_dir = Path("/selfplay_data")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    output_file = output_dir / f"selfplay-{output_shard_idx:05d}_step_{checkpoint_step}_data.bag"
    
    print(f"💾 Writing {len(all_training_data)} training examples to {output_file.name}...")
    
    writer = SimpleBagWriter(str(output_file))
    for fen, move, win_prob in all_training_data:
        record = encode_action_value(fen, move, win_prob)
        writer.write(record)
    writer.close()
    
    # Commit to volume
    selfplay_data_vol.commit()
    print(f"✅ Committed to Modal volume")
    
    # Statistics
    total_decisive = game_results["model_wins"] + game_results["stockfish_wins"]
    model_win_rate = (game_results["model_wins"] / num_games * 100) if num_games > 0 else 0
    
    stats = {
        "num_games": num_games,
        "num_training_examples": len(all_training_data),
        "avg_game_length": float(np.mean(game_lengths)) if game_lengths else 0,
        "model_wins": game_results["model_wins"],
        "stockfish_wins": game_results["stockfish_wins"],
        "draws": game_results["draws"],
        "model_win_rate": model_win_rate,
        "checkmates": game_results["checkmates"],
        "repetitions": game_results["repetitions"],
        "max_moves": game_results["max_moves"],
        "checkpoint_step": checkpoint_step,
        "output_file": output_file.name,
    }
    
    print(f"\n📊 Games vs Stockfish (depth {stockfish_depth}):")
    print(f"   Games played: {stats['num_games']}")
    print(f"   Training examples: {stats['num_training_examples']}")
    print(f"   Avg game length: {stats['avg_game_length']:.1f} moves")
    print(f"   Results: {game_results['model_wins']}W-{game_results['draws']}D-{game_results['stockfish_wins']}L")
    print(f"   Model win rate: {model_win_rate:.1f}%")
    print(f"   Game types: {game_results['checkmates']} checkmates, {game_results['repetitions']} repetitions, {game_results['max_moves']} hit max moves")
    
    return stats


@app.function(
    image=image,
    volumes={"/selfplay_data": selfplay_data_vol},
)
def list_selfplay_data() -> list[str]:
    """List available self-play data files."""
    data_dir = Path("/selfplay_data")
    if not data_dir.exists():
        return []
    
    files = []
    for file in data_dir.glob("selfplay-*.bag"):
        size_mb = file.stat().st_size / 1024 / 1024
        files.append(f"{file.name} ({size_mb:.2f} MB)")
    
    return sorted(files)


@app.local_entrypoint()
def main(
    num_games: int = 100,
    checkpoint_step: int = 22000,
    temperature: float = 0.3,
    num_shards: int = 1,
    stockfish_depth: int = 4,
    stockfish_time_limit: float = 0.1,
    use_mcts: bool = False,
    mcts_simulations: int = 100,
    mcts_c_puct: float = 1.5,
    mcts_max_depth: int = 5,
):
    """Generate games where model plays against Stockfish in parallel.
    
    Args:
        num_games: Total number of games to generate
        checkpoint_step: Which checkpoint to use
        temperature: Temperature for move selection
        num_shards: Number of parallel workers (shards)
        stockfish_depth: Stockfish search depth (higher = stronger)
        stockfish_time_limit: Time limit per Stockfish move in seconds
        use_mcts: Whether to use MCTS for model's moves
        mcts_simulations: Number of MCTS simulations (if enabled)
        mcts_c_puct: Exploration constant for MCTS
        mcts_max_depth: Max depth per simulation (0 = unlimited)
    """
    print("🎮 Chess Training: Model vs Stockfish")
    print("=" * 50)
    print(f"Generating {num_games} games using checkpoint step {checkpoint_step}")
    print(f"Temperature: {temperature}")
    print(f"Stockfish depth: {stockfish_depth}")
    print(f"Stockfish time limit: {stockfish_time_limit}s")
    print(f"Parallel shards: {num_shards}")
    if use_mcts:
        print(f"🌳 MCTS: Enabled ({mcts_simulations} sims, c_puct={mcts_c_puct}, max_depth={mcts_max_depth})")
    else:
        print(f"⚡ MCTS: Disabled (direct evaluation)")
    print()
    
    # Distribute games across shards
    games_per_shard = num_games // num_shards
    
    # Launch parallel self-play workers
    results = []
    for shard_idx in range(num_shards):
        result = generate_selfplay_games.remote(
            num_games=games_per_shard,
            checkpoint_step=checkpoint_step,
            temperature=temperature,
            output_shard_idx=shard_idx,
            stockfish_depth=stockfish_depth,
            stockfish_time_limit=stockfish_time_limit,
            use_mcts=use_mcts,
            mcts_simulations=mcts_simulations,
            mcts_c_puct=mcts_c_puct,
            mcts_max_depth=mcts_max_depth,
        )
        results.append(result)
    
    # Print results
    print("\n✅ Game generation completed!")
    total_model_wins = 0
    total_draws = 0
    total_stockfish_wins = 0
    total_examples = 0
    
    for idx, stats in enumerate(results):
        print(f"\nShard {idx}:")
        print(f"  Games: {stats['num_games']}")
        print(f"  Examples: {stats['num_training_examples']}")
        print(f"  Avg length: {stats['avg_game_length']:.1f}")
        print(f"  Results: {stats['model_wins']}W-{stats['draws']}D-{stats['stockfish_wins']}L")
        print(f"  Win rate: {stats['model_win_rate']:.1f}%")
        print(f"  Types: {stats['checkmates']} checkmates, {stats['repetitions']} repetitions")
        
        total_model_wins += stats['model_wins']
        total_draws += stats['draws']
        total_stockfish_wins += stats['stockfish_wins']
        total_examples += stats['num_training_examples']
    
    # Overall statistics
    total_games = total_model_wins + total_draws + total_stockfish_wins
    overall_win_rate = (total_model_wins / total_games * 100) if total_games > 0 else 0
    
    print(f"\n{'='*50}")
    print(f"📊 Overall Results:")
    print(f"  Total games: {total_games}")
    print(f"  Total training examples: {total_examples}")
    print(f"  Overall: {total_model_wins}W-{total_draws}D-{total_stockfish_wins}L")
    print(f"  Overall win rate: {overall_win_rate:.1f}%")
    
    # List all data files
    files = list_selfplay_data.remote()
    print(f"\n📁 Self-play data files:")
    for f in files:
        print(f"  - {f}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "list":
        files = list_selfplay_data.remote()
        print("Self-play data files:")
        for f in files:
            print(f"  - {f}")
    else:
        # Parse command line args
        num_games = int(sys.argv[1]) if len(sys.argv) > 1 else 100
        checkpoint_step = int(sys.argv[2]) if len(sys.argv) > 2 else 22000
        temperature = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
        num_shards = int(sys.argv[4]) if len(sys.argv) > 4 else 1
        
        main(num_games, checkpoint_step, temperature, num_shards)

