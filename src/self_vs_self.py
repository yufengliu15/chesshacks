"""Self-play training: Model vs slightly mutated version of itself.

The model plays against a copy with slightly randomized weights.
The winner's weights become the new checkpoint.
This is an evolutionary/genetic algorithm approach to training.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import chess
import chess.engine
import modal
import numpy as np
import torch
import torch.nn as nn

# Modal app
app = modal.App("chess-self-vs-self")

# Modal volumes
checkpoints_vol = modal.Volume.from_name("chess-checkpoints", create_if_missing=True)

# Get the src directory path
src_dir = Path(__file__).parent

# Container image
image = (
    modal.Image.debian_slim()
    .apt_install("stockfish")
    .pip_install([
        "torch>=2.0.0",
        "numpy",
        "python-chess",
    ])
    .env({"CACHE_BUST": "2024-11-16-self-vs-self-v12"})
    .add_local_file(str(src_dir / "bagz.py"), remote_path="/root/src/bagz.py")
    .add_local_dir(str(src_dir / "model"), remote_path="/root/model")
)


@dataclass
class SelfVsSelfConfig:
    """Configuration for self-vs-self training."""
    num_games: int = 20  # Games per iteration
    max_moves: int = 50  # Max moves per game
    mutation_std: float = 0.01  # Standard deviation for weight perturbation
    mutation_probability: float = 0.1  # Probability each weight gets mutated
    checkpoint_path: str = "/checkpoints/base_model.pt"
    output_checkpoint_path: str = "/checkpoints/evolved_model.pt"
    device: str = "cuda"
    
    # MCTS settings
    use_mcts: bool = True
    mcts_simulations: int = 10
    mcts_c_puct: float = 1.5
    mcts_max_depth: int = 5
    
    # Stockfish evaluation settings
    use_stockfish_eval: bool = True  # Use Stockfish to evaluate moves
    stockfish_path: str = "/usr/games/stockfish"  # Path to Stockfish
    eval_stockfish_depth: int = 6  # Depth for move evaluation
    eval_weight: float = 0.7  # Weight for move eval vs game outcome
    eval_scale: float = 300.0  # Centipawns to normalize
    
    # Training settings
    train_on_data: bool = True  # Train winner on collected data
    training_epochs: int = 3  # Epochs to train on collected data
    training_batch_size: int = 64  # Batch size for training
    training_lr: float = 1e-4  # Learning rate for training
    
    # Checkpoint settings
    step_increment: int = 1  # How much to increment step number after each iteration


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load model from checkpoint."""
    import sys
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")
    
    # Import directly to avoid dataset.py imports
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "model.transformer_model",
        "/root/model/transformer_model.py"
    )
    transformer_module = importlib.util.module_from_spec(spec)
    # Register in sys.modules before execution (required for dataclasses in Python 3.13)
    sys.modules['model.transformer_model'] = transformer_module
    spec.loader.exec_module(transformer_module)
    
    ActionValueTransformer = transformer_module.ActionValueTransformer
    TransformerConfig = transformer_module.TransformerConfig
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Try to get config from checkpoint (support both 'config' and 'model_config')
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        # Handle nested config format
        if isinstance(config_dict, dict) and 'model' in config_dict:
            config = TransformerConfig(**config_dict['model'])
        else:
            config = TransformerConfig(**config_dict)
    elif 'model_config' in checkpoint:
        config_dict = checkpoint['model_config']
        config = TransformerConfig(**config_dict)
    else:
        # Infer config from state dict
        vocab_size = state_dict['embedding.weight'].shape[0]
        d_model = state_dict['embedding.weight'].shape[1]
        
        # Try to infer other params
        max_seq_len = state_dict.get('pos_encoder.pe', torch.zeros(1, 78, d_model)).shape[1]
        
        # Count transformer layers
        num_layers = 0
        while f'transformer.layers.{num_layers}.self_attn.in_proj_weight' in state_dict:
            num_layers += 1
        
        if num_layers == 0:
            num_layers = 4  # default
        
        # Infer num_heads and dim_feedforward
        if f'transformer.layers.0.self_attn.in_proj_weight' in state_dict:
            # in_proj_weight shape is (3*d_model, d_model) for Q, K, V
            num_heads = 8  # default assumption
        else:
            num_heads = 8
        
        if f'transformer.layers.0.linear1.weight' in state_dict:
            dim_feedforward = state_dict['transformer.layers.0.linear1.weight'].shape[0]
        else:
            dim_feedforward = 256
        
        config = TransformerConfig(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
        )
    
    model = ActionValueTransformer(config)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    return model, config


def mutate_weights(model: nn.Module, std: float = 0.01, probability: float = 0.1) -> nn.Module:
    """Create a mutated copy of the model.
    
    Args:
        model: Original model
        std: Standard deviation of Gaussian noise
        probability: Probability each parameter gets mutated
    
    Returns:
        New model with slightly perturbed weights
    """
    # Create a copy
    mutated_model = type(model)(model.config)
    mutated_model.load_state_dict(model.state_dict())
    
    # Mutate parameters
    with torch.no_grad():
        for param in mutated_model.parameters():
            if param.requires_grad:
                # Create mutation mask
                mask = torch.rand_like(param) < probability
                # Add Gaussian noise where mask is True
                noise = torch.randn_like(param) * std
                param.data += mask * noise
    
    return mutated_model


def centipawns_to_value(cp_eval: float, model_perspective_is_white: bool, scale: float = 300.0) -> float:
    """Convert centipawn evaluation to a value in [0, 1]."""
    # Convert to model's perspective
    if not model_perspective_is_white:
        cp_eval = -cp_eval
    
    # Normalize using sigmoid
    value = 1.0 / (1.0 + np.exp(-cp_eval / scale))
    return np.clip(value, 0.0, 1.0)


def play_game(
    model1,
    model2,
    model1_plays_white: bool,
    max_moves: int = 50,
    device: str = "cuda",
    use_mcts: bool = True,
    mcts_config: Optional[dict] = None,
    stockfish_eval_engine = None,
    eval_stockfish_depth: int = 6,
) -> tuple[str, list, list]:
    """Play a game between two models.
    
    Args:
        model1: First model
        model2: Second model
        model1_plays_white: If True, model1 plays white
        max_moves: Maximum moves before declaring draw
        device: Device to run on
        use_mcts: Whether to use MCTS
        mcts_config: MCTS configuration
        stockfish_eval_engine: Stockfish engine for evaluation (optional)
        eval_stockfish_depth: Depth for Stockfish evaluation
    
    Returns:
        Tuple of (result, model1_moves, model2_moves)
        result: "1-0" (white wins), "0-1" (black wins), "1/2-1/2" (draw)
        model1_moves: List of (fen, move_uci, eval_pawns) for model1
        model2_moves: List of (fen, move_uci, eval_pawns) for model2
    """
    # Import directly to avoid dataset.py
    import importlib.util
    import sys
    
    spec_tokenizer = importlib.util.spec_from_file_location("model.tokenizer", "/root/model/tokenizer.py")
    tokenizer_module = importlib.util.module_from_spec(spec_tokenizer)
    sys.modules['model.tokenizer'] = tokenizer_module
    spec_tokenizer.loader.exec_module(tokenizer_module)
    tokenize = tokenizer_module.tokenize
    VOCAB_SIZE = tokenizer_module.VOCAB_SIZE
    
    spec_moves = importlib.util.spec_from_file_location("model.moves", "/root/model/moves.py")
    moves_module = importlib.util.module_from_spec(spec_moves)
    sys.modules['model.moves'] = moves_module
    spec_moves.loader.exec_module(moves_module)
    MOVE_TO_ACTION = moves_module.MOVE_TO_ACTION
    
    # Initialize MCTS if requested
    mcts1 = None
    mcts2 = None
    if use_mcts and mcts_config:
        # Import MCTS directly
        spec_mcts = importlib.util.spec_from_file_location("model.mcts", "/root/model/mcts.py")
        mcts_module = importlib.util.module_from_spec(spec_mcts)
        sys.modules['model.mcts'] = mcts_module
        spec_mcts.loader.exec_module(mcts_module)
        MCTS = mcts_module.MCTS
        MCTSConfig = mcts_module.MCTSConfig
        
        mcts_cfg = MCTSConfig(
            num_simulations=mcts_config.get('simulations', 100),
            c_puct=mcts_config.get('c_puct', 1.5),
            temperature=0.1,
            max_depth=mcts_config.get('max_depth', 5),
        )
        mcts1 = MCTS(model1, device, mcts_cfg)
        mcts2 = MCTS(model2, device, mcts_cfg)
    
    board = chess.Board()
    
    # Track moves for training data
    model1_moves = []  # (fen, move_uci, eval_pawns)
    model2_moves = []
    
    for move_num in range(max_moves):
        if board.is_game_over():
            break
        
        # Determine whose turn
        is_model1_turn = (board.turn == chess.WHITE) == model1_plays_white
        current_model = model1 if is_model1_turn else model2
        current_mcts = mcts1 if is_model1_turn else mcts2
        
        # Save FEN before move
        fen_before = board.fen()
        
        # Choose move
        if use_mcts and current_mcts:
            move, _ = current_mcts.search(board, add_noise=False)
        else:
            # Direct evaluation
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            
            # Evaluate all legal moves
            fen_tokens = tokenize(board.fen())
            fen_tensor = torch.from_numpy(fen_tokens.astype('int64'))
            
            sequences = []
            kept_moves = []
            
            for mv in legal_moves:
                action_id = MOVE_TO_ACTION.get(mv.uci())
                if action_id is None:
                    continue
                move_token = VOCAB_SIZE + action_id
                seq = torch.cat([
                    fen_tensor,
                    torch.tensor([move_token], dtype=torch.int64)
                ])
                sequences.append(seq)
                kept_moves.append(mv)
            
            if not sequences:
                move = legal_moves[0]
            else:
                batch = torch.stack(sequences).to(device)
                with torch.no_grad():
                    logits = current_model(batch)
                    probs = torch.sigmoid(logits).cpu().numpy()
                
                # Choose best move
                best_idx = np.argmax(probs)
                move = kept_moves[best_idx]
        
        if move is None:
            break
        
        # Make the move
        board.push(move)
        
        # Evaluate position after move (if Stockfish available)
        move_eval_pawns = 0.0
        if stockfish_eval_engine is not None:
            try:
                eval_info = stockfish_eval_engine.analyse(
                    board,
                    chess.engine.Limit(depth=eval_stockfish_depth)
                )
                score = eval_info['score'].white()
                if score.is_mate():
                    mate_in = score.mate()
                    move_eval_pawns = 100.0 if mate_in > 0 else -100.0
                else:
                    move_eval_pawns = score.score() / 100.0
            except Exception as e:
                move_eval_pawns = 0.0
        
        # Store move with evaluation
        if is_model1_turn:
            model1_moves.append((fen_before, move.uci(), move_eval_pawns))
        else:
            model2_moves.append((fen_before, move.uci(), move_eval_pawns))
        
        # Check for draw conditions
        if board.is_repetition(3) or board.can_claim_fifty_moves():
            break
    
    # Get result
    result = board.result(claim_draw=True)
    return result, model1_moves, model2_moves


def train_on_collected_data(
    model,
    training_data: list,
    config: SelfVsSelfConfig,
    device: str = "cuda",
) -> dict:
    """Train model on collected training data.
    
    Args:
        model: Model to train
        training_data: List of (fen, move_uci, target_value)
        config: Training configuration
        device: Device to train on
    
    Returns:
        Training statistics
    """
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import importlib.util
    import sys
    
    # Import tokenizer
    if 'model.tokenizer' not in sys.modules:
        spec_tokenizer = importlib.util.spec_from_file_location("model.tokenizer", "/root/model/tokenizer.py")
        tokenizer_module = importlib.util.module_from_spec(spec_tokenizer)
        sys.modules['model.tokenizer'] = tokenizer_module
        spec_tokenizer.loader.exec_module(tokenizer_module)
    else:
        tokenizer_module = sys.modules['model.tokenizer']
    
    tokenize = tokenizer_module.tokenize
    VOCAB_SIZE = tokenizer_module.VOCAB_SIZE
    
    # Import moves
    if 'model.moves' not in sys.modules:
        spec_moves = importlib.util.spec_from_file_location("model.moves", "/root/model/moves.py")
        moves_module = importlib.util.module_from_spec(spec_moves)
        sys.modules['model.moves'] = moves_module
        spec_moves.loader.exec_module(moves_module)
    else:
        moves_module = sys.modules['model.moves']
    
    MOVE_TO_ACTION = moves_module.MOVE_TO_ACTION
    
    # Create dataset
    class ChessDataset(Dataset):
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            fen, move_uci, target = self.data[idx]
            
            # Tokenize FEN
            fen_tokens = tokenize(fen)
            
            # Get action token
            action_id = MOVE_TO_ACTION.get(move_uci)
            if action_id is None:
                # Skip invalid moves
                return None
            
            action_token = VOCAB_SIZE + action_id
            
            # Combine tokens
            sequence = np.concatenate([fen_tokens, [action_token]])
            
            return {
                'sequence': torch.from_numpy(sequence.astype('int64')),
                'target': torch.tensor(target, dtype=torch.float32)
            }
    
    # Filter valid data
    dataset = ChessDataset(training_data)
    
    def collate_fn(batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        sequences = torch.stack([b['sequence'] for b in batch])
        targets = torch.stack([b['target'] for b in batch])
        return {'sequences': sequences, 'targets': targets}
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.training_batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # Setup training
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=config.training_lr)
    loss_fn = nn.BCEWithLogitsLoss()
    
    total_loss = 0.0
    total_batches = 0
    
    print(f"\n🎓 Training on {len(training_data)} examples for {config.training_epochs} epochs...")
    
    for epoch in range(config.training_epochs):
        epoch_loss = 0.0
        epoch_batches = 0
        
        for batch in dataloader:
            if batch is None:
                continue
            
            sequences = batch['sequences'].to(device)
            targets = batch['targets'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(sequences)
            
            # Compute loss
            loss = loss_fn(logits.squeeze(), targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_batches += 1
            total_loss += loss.item()
            total_batches += 1
        
        if epoch_batches > 0:
            avg_epoch_loss = epoch_loss / epoch_batches
            print(f"  Epoch {epoch + 1}/{config.training_epochs}: loss = {avg_epoch_loss:.4f}")
    
    model.eval()
    
    avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
    
    stats = {
        'avg_loss': avg_loss,
        'total_batches': total_batches,
        'epochs': config.training_epochs,
    }
    
    print(f"✅ Training complete: avg loss = {avg_loss:.4f}")
    
    return stats


def assign_training_data(
    model1_moves: list,
    model2_moves: list,
    result: str,
    model1_plays_white: bool,
    eval_weight: float = 0.7,
    eval_scale: float = 300.0,
) -> list:
    """Assign training targets to moves based on outcome and evaluations.
    
    Returns:
        List of (fen, move_uci, target_value) tuples for ALL moves
    """
    training_data = []
    
    # Determine outcomes
    if result == "1-0":
        white_outcome = 1.0
        black_outcome = 0.0
    elif result == "0-1":
        white_outcome = 0.0
        black_outcome = 1.0
    else:
        white_outcome = 0.1  # Harsh draw penalty
        black_outcome = 0.1
    
    # Model 1 outcome
    model1_outcome = white_outcome if model1_plays_white else black_outcome
    model1_is_white = model1_plays_white
    
    # Model 2 outcome
    model2_outcome = black_outcome if model1_plays_white else white_outcome
    model2_is_white = not model1_plays_white
    
    # Process model1 moves
    for fen, move_uci, eval_pawns in model1_moves:
        if eval_pawns != 0.0:
            # Convert eval to value
            eval_value = centipawns_to_value(
                eval_pawns * 100,  # Convert to centipawns
                model1_is_white,
                eval_scale
            )
            # Blend with outcome
            final_value = eval_weight * eval_value + (1 - eval_weight) * model1_outcome
        else:
            final_value = model1_outcome
        
        final_value = max(0.0, min(1.0, final_value))
        training_data.append((fen, move_uci, final_value))
    
    # Process model2 moves
    for fen, move_uci, eval_pawns in model2_moves:
        if eval_pawns != 0.0:
            eval_value = centipawns_to_value(
                eval_pawns * 100,
                model2_is_white,
                eval_scale
            )
            final_value = eval_weight * eval_value + (1 - eval_weight) * model2_outcome
        else:
            final_value = model2_outcome
        
        final_value = max(0.0, min(1.0, final_value))
        training_data.append((fen, move_uci, final_value))
    
    return training_data


@app.function(
    image=image,
    volumes={
        "/checkpoints": checkpoints_vol,
        "/selfplay_data": modal.Volume.from_name("chess-selfplay-data", create_if_missing=True),
    },
    gpu="H100",
    timeout=3600 * 2,  # 2 hours
    memory=32000,
)
def run_self_vs_self_iteration(
    checkpoint_step: int = 22000,
    num_games: int = 20,
    mutation_std: float = 0.01,
    mutation_probability: float = 0.1,
    use_mcts: bool = True,
    mcts_simulations: int = 10,
    mcts_c_puct: float = 1.5,
    mcts_max_depth: int = 5,
    use_stockfish_eval: bool = True,
    stockfish_path: str = "/usr/games/stockfish",
    eval_stockfish_depth: int = 6,
    eval_weight: float = 0.7,
    eval_scale: float = 300.0,
    train_on_data: bool = True,
    training_epochs: int = 3,
    training_batch_size: int = 64,
    training_lr: float = 1e-4,
    step_increment: int = 1,
    output_shard_idx: int = 0,
) -> dict:
    """Run one iteration of self-vs-self training.
    
    Args:
        checkpoint_step: Starting checkpoint
        num_games: Number of games to play
        mutation_std: Std dev for weight mutations
        mutation_probability: Probability each weight mutates
        use_mcts: Whether to use MCTS
        mcts_simulations: Number of MCTS simulations
        mcts_c_puct: MCTS exploration constant
        mcts_max_depth: MCTS max depth
        use_stockfish_eval: Use Stockfish to evaluate moves
        stockfish_path: Path to Stockfish binary
        eval_stockfish_depth: Depth for Stockfish evaluation
        eval_weight: Weight for eval vs outcome
        eval_scale: Centipawn scale for normalization
        train_on_data: Train winner on collected data
        training_epochs: Number of training epochs
        training_batch_size: Batch size for training
        training_lr: Learning rate for training
        step_increment: How much to increment checkpoint step
        output_shard_idx: Shard index for output file
    
    Returns:
        Statistics dictionary
    """
    import sys
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")
    if "/root/src" not in sys.path:
        sys.path.insert(0, "/root/src")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Reload volume to get latest state
    checkpoints_vol.reload()
    
    # Load base model
    checkpoint_path = f"/checkpoints/checkpoint_step_{checkpoint_step}.pt"
    print(f"📥 Loading checkpoint: {checkpoint_path}")
    
    # Verify checkpoint exists before trying to load
    import os
    if not os.path.exists(checkpoint_path):
        print(f"❌ ERROR: Checkpoint not found: {checkpoint_path}")
        print(f"Available checkpoints in /checkpoints:")
        try:
            files = sorted(os.listdir("/checkpoints"))
            for f in files[-10:]:  # Show last 10 files
                print(f"  - {f}")
        except Exception as e:
            print(f"  Could not list directory: {e}")
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    base_model, config = load_model(checkpoint_path, str(device))
    print(f"✅ Loaded base model from step {checkpoint_step}")
    
    # Create mutated version
    print(f"🧬 Creating mutated model (std={mutation_std}, prob={mutation_probability})")
    mutated_model = mutate_weights(base_model, std=mutation_std, probability=mutation_probability)
    mutated_model.to(device)
    mutated_model.eval()
    print(f"✅ Created mutated model")
    
    # MCTS config
    mcts_config = {
        'simulations': mcts_simulations,
        'c_puct': mcts_c_puct,
        'max_depth': mcts_max_depth,
    } if use_mcts else None
    
    if use_mcts:
        print(f"🌳 MCTS enabled: {mcts_simulations} sims, c_puct={mcts_c_puct}, max_depth={mcts_max_depth}")
    else:
        print(f"⚡ MCTS disabled (greedy selection)")
    
    # Initialize Stockfish evaluator if requested
    stockfish_eval = None
    if use_stockfish_eval:
        import chess.engine
        stockfish_eval = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        print(f"🎯 Stockfish evaluation enabled (depth {eval_stockfish_depth})")
    
    # Play games
    print(f"\n🎮 Playing {num_games} games (Base vs Mutated)...")
    
    base_wins = 0
    mutated_wins = 0
    draws = 0
    all_training_data = []
    
    start_time = time.time()
    
    try:
        for game_idx in range(num_games):
            # Alternate colors
            base_plays_white = (game_idx % 2 == 0)
            
            result, model1_moves, model2_moves = play_game(
                base_model,
                mutated_model,
                model1_plays_white=base_plays_white,
                max_moves=50,
                device=str(device),
                use_mcts=use_mcts,
                mcts_config=mcts_config,
                stockfish_eval_engine=stockfish_eval,
                eval_stockfish_depth=eval_stockfish_depth,
            )
        
            # Collect training data from this game
            game_training_data = assign_training_data(
                model1_moves,
                model2_moves,
                result,
                model1_plays_white=base_plays_white,
                eval_weight=eval_weight,
                eval_scale=eval_scale,
            )
            all_training_data.extend(game_training_data)
            
            # Track results
            if result == "1-0":
                if base_plays_white:
                    base_wins += 1
                else:
                    mutated_wins += 1
            elif result == "0-1":
                if base_plays_white:
                    mutated_wins += 1
                else:
                    base_wins += 1
            else:
                draws += 1
            
            if (game_idx + 1) % 5 == 0:
                print(f"  Game {game_idx + 1}/{num_games}: "
                      f"Base {base_wins}W-{draws}D-{mutated_wins}L (vs Mutated)")
    
    finally:
        # Clean up Stockfish
        if stockfish_eval is not None:
            stockfish_eval.quit()
    
    elapsed = time.time() - start_time
    
    # Determine winner
    if mutated_wins > base_wins:
        winner = "mutated"
        winner_model = mutated_model
        print(f"\n🏆 MUTATED model wins! ({mutated_wins}-{draws}-{base_wins})")
    elif base_wins > mutated_wins:
        winner = "base"
        winner_model = base_model
        print(f"\n🏆 BASE model wins! ({base_wins}-{draws}-{mutated_wins})")
    else:
        # Tie - keep base model
        winner = "base (tie)"
        winner_model = base_model
        print(f"\n🤝 TIE! Keeping base model ({base_wins}-{draws}-{mutated_wins})")
    
    # Train winner on collected data
    training_stats = {}
    if train_on_data and all_training_data:
        # Create config for training
        train_config = SelfVsSelfConfig(
            training_epochs=training_epochs,
            training_batch_size=training_batch_size,
            training_lr=training_lr,
        )
        
        training_stats = train_on_collected_data(
            winner_model,
            all_training_data,
            train_config,
            str(device)
        )
    
    # Save winner as new checkpoint
    new_step = checkpoint_step + step_increment
    new_checkpoint_path = f"/checkpoints/checkpoint_step_{new_step}.pt"
    
    print(f"\n💾 Saving winner to {new_checkpoint_path}...")
    
    checkpoint = {
        'step': new_step,
        'model_state_dict': winner_model.state_dict(),
        'config': config.to_dict(),  # Use 'config' not 'model_config' for compatibility with engine.py
        'training_method': 'self_vs_self',
        'parent_step': checkpoint_step,
        'mutation_std': mutation_std,
        'mutation_probability': mutation_probability,
        'base_wins': base_wins,
        'mutated_wins': mutated_wins,
        'draws': draws,
        'winner': winner,
    }
    
    torch.save(checkpoint, new_checkpoint_path)
    
    # Commit and reload to ensure checkpoint is available
    print(f"💾 Committing checkpoint to volume...")
    checkpoints_vol.commit()
    checkpoints_vol.reload()
    
    # Verify checkpoint exists
    import os
    if not os.path.exists(new_checkpoint_path):
        raise FileNotFoundError(f"Checkpoint was saved but not found after commit: {new_checkpoint_path}")
    
    print(f"✅ Saved checkpoint step {new_step}")
    
    # Save training data to bag file
    if all_training_data:
        from pathlib import Path
        import struct
        
        def _write_varint(value: int) -> bytes:
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
            fen_bytes = fen.encode("utf-8")
            move_bytes = move.encode("utf-8")
            win_prob_bytes = struct.pack(">d", win_prob)
            return (
                _write_varint(len(fen_bytes)) + fen_bytes +
                _write_varint(len(move_bytes)) + move_bytes +
                win_prob_bytes
            )
        
        output_dir = Path("/selfplay_data")
        output_dir.mkdir(exist_ok=True, parents=True)
        output_file = output_dir / f"self_vs_self-{output_shard_idx:05d}_step_{checkpoint_step}_data.bag"
        
        print(f"\n💾 Writing {len(all_training_data)} training examples to {output_file.name}...")
        
        # Simple bag writer
        with open(str(output_file) + ".tmp", "wb") as records_file:
            with open(str(output_file) + ".limits.tmp", "wb") as limits_file:
                for fen, move, win_prob in all_training_data:
                    record = encode_action_value(fen, move, win_prob)
                    if record:
                        records_file.write(record)
                    limits_file.write(struct.pack('<q', records_file.tell()))
        
        # Finalize bag file
        import shutil
        with open(str(output_file) + ".tmp", "ab") as records_file:
            with open(str(output_file) + ".limits.tmp", "rb") as limits_file:
                shutil.copyfileobj(limits_file, records_file)
        
        os.unlink(str(output_file) + ".limits.tmp")
        os.rename(str(output_file) + ".tmp", str(output_file))
        
        # Commit to volume
        modal.Volume.from_name("chess-selfplay-data").commit()
        
        print(f"✅ Saved training data ({len(all_training_data)} examples)")
    
    # Statistics
    stats = {
        'checkpoint_step': checkpoint_step,
        'new_checkpoint_step': new_step,
        'num_games': num_games,
        'base_wins': base_wins,
        'mutated_wins': mutated_wins,
        'draws': draws,
        'winner': winner,
        'mutation_std': mutation_std,
        'mutation_probability': mutation_probability,
        'elapsed_time': elapsed,
        'games_per_second': num_games / elapsed if elapsed > 0 else 0,
        'training_examples': len(all_training_data),
        'trained': train_on_data and bool(all_training_data),
        'training_loss': training_stats.get('avg_loss', 0.0) if training_stats else 0.0,
    }
    
    print(f"\n📊 Statistics:")
    print(f"   Games played: {num_games}")
    print(f"   Base wins: {base_wins}")
    print(f"   Mutated wins: {mutated_wins}")
    print(f"   Draws: {draws}")
    print(f"   Winner: {winner}")
    print(f"   Training examples: {len(all_training_data)}")
    print(f"   Time: {elapsed:.1f}s ({stats['games_per_second']:.2f} games/s)")
    print(f"   New checkpoint: step {new_step}")
    
    return stats


@app.local_entrypoint()
def main(
    starting_checkpoint: int = 22000,
    num_iterations: int = 10,
    games_per_iteration: int = 20,
    mutation_std: float = 0.01,
    mutation_probability: float = 0.1,
    use_mcts: bool = True,
    mcts_simulations: int = 10,
    mcts_c_puct: float = 1.5,
    mcts_max_depth: int = 5,
    use_stockfish_eval: bool = True,
    stockfish_path: str = "/usr/games/stockfish",
    eval_stockfish_depth: int = 6,
    eval_weight: float = 0.7,
    eval_scale: float = 300.0,
    train_on_data: bool = True,
    training_epochs: int = 3,
    training_batch_size: int = 64,
    training_lr: float = 1e-4,
    step_increment: int = 1,
):
    """Run multiple iterations of self-vs-self evolution.
    
    Args:
        starting_checkpoint: Initial checkpoint step
        num_iterations: Number of evolution iterations
        games_per_iteration: Games per iteration
        mutation_std: Standard deviation for mutations
        mutation_probability: Probability each weight mutates
        use_mcts: Use MCTS for move selection
        mcts_simulations: MCTS simulations per move
        mcts_c_puct: MCTS exploration constant
        mcts_max_depth: MCTS max depth
        use_stockfish_eval: Use Stockfish for move evaluation
        stockfish_path: Path to Stockfish binary
        eval_stockfish_depth: Stockfish evaluation depth
        eval_weight: Weight for evaluation vs outcome
        eval_scale: Centipawn scale for normalization
        train_on_data: Train winner on collected data
        training_epochs: Number of training epochs
        training_batch_size: Batch size for training
        training_lr: Learning rate for training
        step_increment: How much to increment checkpoint step after each iteration
    """
    print("=" * 70)
    print("🧬 SELF-VS-SELF EVOLUTIONARY TRAINING")
    print("=" * 70)
    print(f"Starting checkpoint: {starting_checkpoint}")
    print(f"Iterations: {num_iterations}")
    print(f"Games per iteration: {games_per_iteration}")
    print(f"Mutation: std={mutation_std}, prob={mutation_probability}")
    print(f"MCTS: {'Enabled' if use_mcts else 'Disabled'}")
    if use_mcts:
        print(f"  Simulations: {mcts_simulations}")
        print(f"  C_PUCT: {mcts_c_puct}")
        print(f"  Max depth: {mcts_max_depth}")
    print(f"Stockfish Eval: {'Enabled' if use_stockfish_eval else 'Disabled'}")
    if use_stockfish_eval:
        print(f"  Depth: {eval_stockfish_depth}")
        print(f"  Eval weight: {eval_weight}")
        print(f"  Scale: {eval_scale}cp")
    print(f"Training: {'Enabled' if train_on_data else 'Disabled'}")
    if train_on_data:
        print(f"  Epochs: {training_epochs}")
        print(f"  Batch size: {training_batch_size}")
        print(f"  Learning rate: {training_lr}")
    print("=" * 70)
    print()
    
    current_checkpoint = starting_checkpoint
    
    for iteration in range(num_iterations):
        print(f"\n{'='*70}")
        print(f"🔄 ITERATION {iteration + 1}/{num_iterations}")
        print(f"{'='*70}\n")
        
        stats = run_self_vs_self_iteration.remote(
            checkpoint_step=current_checkpoint,
            num_games=games_per_iteration,
            mutation_std=mutation_std,
            mutation_probability=mutation_probability,
            use_mcts=use_mcts,
            mcts_simulations=mcts_simulations,
            mcts_c_puct=mcts_c_puct,
            mcts_max_depth=mcts_max_depth,
            use_stockfish_eval=use_stockfish_eval,
            stockfish_path=stockfish_path,
            eval_stockfish_depth=eval_stockfish_depth,
            eval_weight=eval_weight,
            eval_scale=eval_scale,
            train_on_data=train_on_data,
            training_epochs=training_epochs,
            training_batch_size=training_batch_size,
            training_lr=training_lr,
            step_increment=step_increment,
            output_shard_idx=iteration,
        )
        
        current_checkpoint = stats['new_checkpoint_step']
        
        print(f"\n✅ Iteration {iteration + 1} complete")
        print(f"   Winner: {stats['winner']}")
        print(f"   Score: {stats['base_wins']}-{stats['draws']}-{stats['mutated_wins']}")
        print(f"   Training examples: {stats['training_examples']}")
        if stats.get('trained'):
            print(f"   Training loss: {stats.get('training_loss', 0.0):.4f}")
        print(f"   New checkpoint: step {current_checkpoint}")
    
    print(f"\n{'='*70}")
    print(f"🎉 TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Final checkpoint: step {current_checkpoint}")
    print(f"Total iterations: {num_iterations}")
    print(f"Starting checkpoint: {starting_checkpoint}")
    print(f"Improvement: +{current_checkpoint - starting_checkpoint} steps")
    print(f"{'='*70}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Self-vs-self evolutionary training")
    parser.add_argument("--starting-checkpoint", type=int, default=22000,
                       help="Starting checkpoint step")
    parser.add_argument("--num-iterations", type=int, default=10,
                       help="Number of evolution iterations")
    parser.add_argument("--games-per-iteration", type=int, default=20,
                       help="Games per iteration")
    parser.add_argument("--mutation-std", type=float, default=0.01,
                       help="Standard deviation for weight mutations")
    parser.add_argument("--mutation-probability", type=float, default=0.1,
                       help="Probability each weight gets mutated")
    parser.add_argument("--use-mcts", action="store_true", default=True,
                       help="Use MCTS for move selection")
    parser.add_argument("--no-mcts", action="store_false", dest="use_mcts",
                       help="Disable MCTS")
    parser.add_argument("--mcts-simulations", type=int, default=100,
                       help="MCTS simulations per move")
    parser.add_argument("--mcts-c-puct", type=float, default=1.5,
                       help="MCTS exploration constant")
    parser.add_argument("--mcts-max-depth", type=int, default=5,
                       help="MCTS max depth")
    parser.add_argument("--use-stockfish-eval", action="store_true", default=True,
                       help="Use Stockfish to evaluate moves")
    parser.add_argument("--no-stockfish-eval", action="store_false", dest="use_stockfish_eval",
                       help="Disable Stockfish evaluation")
    parser.add_argument("--stockfish-path", type=str, default="/usr/games/stockfish",
                       help="Path to Stockfish binary")
    parser.add_argument("--eval-stockfish-depth", type=int, default=6,
                       help="Stockfish depth for move evaluation")
    parser.add_argument("--eval-weight", type=float, default=0.7,
                       help="Weight for eval vs outcome (0.7 = 70%% eval, 30%% outcome)")
    parser.add_argument("--eval-scale", type=float, default=300.0,
                       help="Centipawn scale for normalization")
    parser.add_argument("--train-on-data", action="store_true", default=True,
                       help="Train winner on collected data")
    parser.add_argument("--no-train", action="store_false", dest="train_on_data",
                       help="Disable training on collected data")
    parser.add_argument("--training-epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--training-batch-size", type=int, default=64,
                       help="Batch size for training")
    parser.add_argument("--training-lr", type=float, default=1e-4,
                       help="Learning rate for training")
    parser.add_argument("--step-increment", type=int, default=1,
                       help="How much to increment checkpoint step after each iteration")
    
    args = parser.parse_args()
    
    main(
        starting_checkpoint=args.starting_checkpoint,
        num_iterations=args.num_iterations,
        games_per_iteration=args.games_per_iteration,
        mutation_std=args.mutation_std,
        mutation_probability=args.mutation_probability,
        use_mcts=args.use_mcts,
        mcts_simulations=args.mcts_simulations,
        mcts_c_puct=args.mcts_c_puct,
        mcts_max_depth=args.mcts_max_depth,
        use_stockfish_eval=args.use_stockfish_eval,
        stockfish_path=args.stockfish_path,
        eval_stockfish_depth=args.eval_stockfish_depth,
        eval_weight=args.eval_weight,
        eval_scale=args.eval_scale,
        train_on_data=args.train_on_data,
        training_epochs=args.training_epochs,
        training_batch_size=args.training_batch_size,
        training_lr=args.training_lr,
        step_increment=args.step_increment,
    )

