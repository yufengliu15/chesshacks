"""Runtime chess engine powered by the transformer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import chess
import torch

from .dataset import fen_to_tokens, action_to_token
from .transformer_model import ActionValueTransformer, TransformerConfig
from .mcts import MCTS, MCTSConfig


CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoint"
CHECKPOINT_PATH = CHECKPOINT_DIR / "action_value_transformer.pt"


@dataclass
class EngineConfig:
    device: str = "cpu"
    checkpoint_path: Path | str | None = None  # Allow None to auto-find latest
    use_mcts: bool = False  # Whether to use MCTS for move selection
    mcts_simulations: int = 800  # Number of MCTS simulations
    mcts_c_puct: float = 1.5  # Exploration constant for MCTS
    mcts_temperature: float = 0.1  # Temperature for MCTS move selection (0=greedy)
    mcts_max_depth: Optional[int] = None  # Max depth per simulation (None = unlimited)


class TransformerChessEngine:
    """Wraps the trained PyTorch model for move selection."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        
        # Validate and set device
        device_str = self.config.device.lower()
        if device_str == "cuda" and not torch.cuda.is_available():
            print("⚠️  CUDA requested but not available, falling back to CPU")
            device_str = "cpu"
        elif device_str == "mps" and not torch.backends.mps.is_available():
            print("⚠️  MPS requested but not available, falling back to CPU")
            device_str = "cpu"
        
        self.device = torch.device(device_str)
        print(f"🔧 Engine using device: {self.device}")
        self.model: ActionValueTransformer | None = None
        self.mcts: Optional[MCTS] = None
        self._load_model()
        
        # Initialize MCTS if configured
        if self.config.use_mcts and self.model is not None:
            print(f"🌳 Initializing MCTS: {self.config.mcts_simulations} simulations, "
                  f"c_puct={self.config.mcts_c_puct}, max_depth={self.config.mcts_max_depth}")
            mcts_config = MCTSConfig(
                num_simulations=self.config.mcts_simulations,
                c_puct=self.config.mcts_c_puct,
                temperature=self.config.mcts_temperature,
                max_depth=self.config.mcts_max_depth,
            )
            self.mcts = MCTS(self.model, str(self.device), mcts_config)
        else:
            if not self.config.use_mcts:
                print(f"⚡ MCTS disabled (use_mcts={self.config.use_mcts})")
            elif self.model is None:
                print(f"⚠️  MCTS requested but model failed to load")

    def _load_model(self) -> None:
        # Determine checkpoint path
        if self.config.checkpoint_path is None:
            # Auto-find latest checkpoint in checkpoint directory
            ckpt_path = self._find_latest_checkpoint()
            if ckpt_path is None:
                print("⚠️  No checkpoint found in checkpoint directory")
                return
        else:
            ckpt_path = Path(self.config.checkpoint_path)
            if not ckpt_path.exists():
                print(f"⚠️  Checkpoint not found: {ckpt_path}")
                return

        print(f"📥 Loading checkpoint from: {ckpt_path}")

        try:
            # Always load to CPU first to avoid device compatibility issues
            # Then move model to target device after loading
            checkpoint = torch.load(ckpt_path, map_location='cpu')

            # Handle different checkpoint formats
            if "config" in checkpoint and isinstance(checkpoint["config"], dict):
                # New format from train.py (has full CONFIG dict)
                if "model" in checkpoint["config"]:
                    # Extract just the model config
                    model_cfg = checkpoint["config"]["model"]
                    model_config = TransformerConfig(**model_cfg)
                else:
                    # Old format (config is the model config)
                    model_config = TransformerConfig(**checkpoint["config"])
            else:
                raise ValueError("Checkpoint missing 'config' field")

            self.model = ActionValueTransformer(model_config)

            # Get state dict (handle both 'model_state_dict' and 'model_state')
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
            self.model.load_state_dict(state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()

            # Print checkpoint info
            if "step" in checkpoint:
                print(f"✅ Loaded checkpoint from step {checkpoint['step']}")
            if "loss" in checkpoint:
                print(f"   Loss: {checkpoint['loss']:.4f}")
            
            # Print weight statistics to verify checkpoint identity
            if "weight_stats" in checkpoint:
                ws = checkpoint["weight_stats"]
                print(f"   Embedding: μ={ws.get('embedding_mean', 'N/A'):.6f}, "
                      f"σ={ws.get('embedding_std', 'N/A'):.6f}, "
                      f"Δ={ws.get('embedding_change', 'N/A'):.4f}")
            else:
                # Calculate stats from loaded weights
                emb_weights = self.model.embedding.weight.data
                print(f"   Embedding: μ={emb_weights.mean().item():.6f}, "
                      f"σ={emb_weights.std().item():.6f}")

        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            import traceback
            traceback.print_exc()
            self.model = None

    def _find_latest_checkpoint(self) -> Path | None:
        """Find the latest checkpoint in the checkpoint directory."""
        if not CHECKPOINT_DIR.exists():
            return None

        # Look for checkpoint files
        checkpoints = list(CHECKPOINT_DIR.glob("checkpoint_step_*.pt"))
        checkpoints.extend(CHECKPOINT_DIR.glob("final_checkpoint_step_*.pt"))

        if not checkpoints:
            # Fallback to default checkpoint if it exists
            if CHECKPOINT_PATH.exists():
                return CHECKPOINT_PATH
            return None

        # Sort by step number (extract from filename)
        def extract_step(path: Path) -> int:
            try:
                # Extract number from "checkpoint_step_1000.pt" or "final_checkpoint_step_1000.pt"
                parts = path.stem.split("_")
                for i, part in enumerate(parts):
                    if part == "step" and i + 1 < len(parts):
                        return int(parts[i + 1])
                return 0
            except:
                return 0

        latest = max(checkpoints, key=extract_step)
        return latest

    def available(self) -> bool:
        return self.model is not None

    def _score_moves(
        self, board: chess.Board, legal_moves: list[chess.Move]
    ) -> Dict[chess.Move, float]:
        assert self.model is not None
        fen_tokens = fen_to_tokens(board.fen())
        fen_tensor = torch.from_numpy(fen_tokens.astype("int64"))
        sequences = []
        kept_moves: list[chess.Move] = []
        for move in legal_moves:
            move_token = action_to_token(move.uci())
            if move_token is None:
                continue
            seq = torch.cat(
                [
                    fen_tensor,
                    torch.tensor([move_token], dtype=torch.int64),
                ]
            )
            sequences.append(seq)
            kept_moves.append(move)

        if not sequences:
            return {}

        batch = torch.stack(sequences).to(self.device)
        with torch.no_grad():
            logits = self.model(batch)
            probs = torch.sigmoid(logits).cpu().tolist()

        return {move: prob for move, prob in zip(kept_moves, probs)}

    def choose_move(
        self, board: chess.Board, use_mcts: Optional[bool] = None
    ) -> tuple[chess.Move | None, Dict[chess.Move, float]]:
        """Choose a move for the given position.
        
        Args:
            board: Current chess position
            use_mcts: Override config to use/not use MCTS (None = use config default)
        
        Returns:
            move: Selected move (or None if no legal moves)
            probabilities: Dict of move -> probability/score
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None, {}

        if not self.available():
            uniform = {move: 1.0 / len(legal_moves) for move in legal_moves}
            return legal_moves[0], uniform

        # Determine whether to use MCTS
        should_use_mcts = use_mcts if use_mcts is not None else self.config.use_mcts
        
        if should_use_mcts and self.mcts is not None:
            # Use MCTS for move selection
            best_move, mcts_info = self.mcts.search(board, add_noise=False)
            
            # Convert MCTS info to move probabilities
            # Extract visit counts from the info dict
            visit_counts = {}
            for key, value in mcts_info.items():
                if key.endswith('_visits'):
                    move_uci = key[:-7]  # Remove '_visits' suffix
                    try:
                        move = chess.Move.from_uci(move_uci)
                        visit_counts[move] = value
                    except:
                        pass
            
            return best_move, visit_counts
        else:
            # Use direct value-based selection (original method)
            move_scores = self._score_moves(board, legal_moves)
            if not move_scores:
                uniform = {move: 1.0 / len(legal_moves) for move in legal_moves}
                return legal_moves[0], uniform

            best_move = max(move_scores.items(), key=lambda item: item[1])[0]
            total = sum(move_scores.values())
            if total > 0.0:
                normalized = {move: score / total for move, score in move_scores.items()}
            else:
                normalized = move_scores
            return best_move, normalized

