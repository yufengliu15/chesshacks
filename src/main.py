from __future__ import annotations

import os
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env from project root
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Loaded configuration from .env file")
    else:
        print(f"ℹ️  No .env file found, using environment variables or defaults")
except ImportError:
    print(f"ℹ️  python-dotenv not installed, using environment variables only")
    print(f"   Install with: pip install python-dotenv")

from .utils import chess_manager, GameContext
from .model import TransformerChessEngine
from .model.engine import EngineConfig


# Allow specifying checkpoint via environment variable or use latest
CHECKPOINT_PATH = os.environ.get("CHESS_CHECKPOINT")
DEVICE = os.environ.get("CHESS_DEVICE", "cpu")

# MCTS configuration from environment
USE_MCTS = os.environ.get("CHESS_USE_MCTS", "0").lower() in ("1", "true", "yes")
MCTS_SIMULATIONS = int(os.environ.get("CHESS_MCTS_SIMULATIONS", "800"))
MCTS_C_PUCT = float(os.environ.get("CHESS_MCTS_C_PUCT", "1.5"))
MCTS_TEMPERATURE = float(os.environ.get("CHESS_MCTS_TEMPERATURE", "0.0"))
MCTS_MAX_DEPTH = int(os.environ.get("CHESS_MCTS_MAX_DEPTH", "5")) or None  # 0 = unlimited

if CHECKPOINT_PATH:
    print(f"🎯 Using checkpoint from environment: {CHECKPOINT_PATH}")
else:
    print(f"🎯 Auto-loading latest checkpoint from model/checkpoint/")

if USE_MCTS:
    depth_str = f"max_depth={MCTS_MAX_DEPTH}" if MCTS_MAX_DEPTH else "unlimited depth"
    print(f"🌳 MCTS enabled: {MCTS_SIMULATIONS} simulations, c_puct={MCTS_C_PUCT}, temp={MCTS_TEMPERATURE}, {depth_str}")
else:
    print(f"⚡ MCTS disabled (using greedy selection)")

config = EngineConfig(
    device=DEVICE,
    checkpoint_path=CHECKPOINT_PATH,
    use_mcts=USE_MCTS,
    mcts_simulations=MCTS_SIMULATIONS,
    mcts_c_puct=MCTS_C_PUCT,
    mcts_temperature=MCTS_TEMPERATURE,
    mcts_max_depth=MCTS_MAX_DEPTH,
)

# Module-level engine - loaded once at import time
_ENGINE = TransformerChessEngine(config=config)


def reload_engine():
    """Reload the engine with the latest checkpoint.
    
    Call this after downloading a new checkpoint to update the model
    without restarting the server.
    
    Usage:
        from main import reload_engine
        reload_engine()
    """
    global _ENGINE
    print("🔄 Reloading chess engine...")
    
    # Determine checkpoint path
    checkpoint_path = os.environ.get("CHESS_CHECKPOINT")
    device = os.environ.get("CHESS_DEVICE", "cpu")
    
    if checkpoint_path:
        print(f"🎯 Loading checkpoint from environment: {checkpoint_path}")
        config = EngineConfig(device=device, checkpoint_path=checkpoint_path)
    else:
        print(f"🎯 Auto-loading latest checkpoint from model/checkpoint/")
        config = EngineConfig(device=device, checkpoint_path=None)
    
    _ENGINE = TransformerChessEngine(config=config)
    print("✅ Engine reloaded successfully!")
    return _ENGINE


@chess_manager.entrypoint
def transformer_policy(ctx: GameContext):
    """Selects a move using the transformer policy."""
    move, probabilities = _ENGINE.choose_move(ctx.board)
    ctx.logProbabilities(probabilities)
    if move is None:
        raise ValueError("No legal moves available.")
    return move


@chess_manager.reset
def reset_func(ctx: GameContext):
    """Reset hook (no-op for now)."""
    _ = ctx
