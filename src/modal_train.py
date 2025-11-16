"""Modal-based training script for chess transformer model with HuggingFace dataset integration."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import modal
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset
import numpy as np

# Import the existing transformer model and dataset utilities
from model.transformer_model import ActionValueTransformer, TransformerConfig
from model.dataset import ActionValueIterableDataset, VOCAB_SIZE, SEQUENCE_LENGTH
from model import tokenizer
from model import moves
from model.coders import decode_action_value

# Modal setup
app = modal.App("chess-training")

# Modal volumes for persistent storage
checkpoints_vol = modal.Volume.from_name("chess-checkpoints", create_if_missing=True)
data_vol = modal.Volume.from_name("chess-data", create_if_missing=True)

# Container image with required dependencies
image = (
    modal.Image.debian_slim()
    .pip_install([
        "torch>=2.0.0",
        "tqdm",
        "numpy",
        "datasets",
        "huggingface-hub",
        "transformers",
        "bagz",
    ])
    .copy_local_dir(
        str(Path(__file__).parent),
        "/app/model"
    )
)

# Training configuration - using existing model parameters
CONFIG = {
    "model": {
        "vocab_size": VOCAB_SIZE,
        "max_seq_len": SEQUENCE_LENGTH,
        "d_model": 128,
        "num_layers": 4,
        "num_heads": 8,
        "dim_feedforward": 256,
        "dropout": 0.1,
    },
    "training": {
        "batch_size": 512,
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "warmup_steps": 1000,
        "max_steps": 6000,
        "gradient_clip": 1.0,
        "eval_interval": 500,
        "log_interval": 100,
        "checkpoint_interval": 1000,  # Save checkpoint every N steps
        "max_records": 100000,  # Records per epoch
    },
    "data": {
        "dataset_name": "chimcis/searcless-chess-10m",
        "max_length": SEQUENCE_LENGTH,
        "train_split": "train",
        "val_split": "validation",
    }
}


class HuggingFaceChessDataset:
    """Dataset wrapper for HuggingFace chess dataset that produces data in the same format as ActionValueIterableDataset."""
    
    def __init__(self, dataset, max_length: int = SEQUENCE_LENGTH):
        self.dataset = dataset
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataset)
    
    def __iter__(self):
        """Iterate through dataset yielding (sequence, target) pairs like ActionValueIterableDataset."""
        for item in self.dataset:
            try:
                # Handle different possible dataset formats
                if isinstance(item, dict):
                    # Try to extract FEN, move, and win probability
                    if 'fen' in item and 'move' in item:
                        fen = item['fen']
                        move = item['move']
                        win_prob = item.get('win_prob', item.get('value', item.get('score', 0.5)))
                    elif 'text' in item:
                        # Parse text format: "fen move win_prob"
                        text = item['text']
                        parts = text.strip().split()
                        if len(parts) >= 3:
                            fen, move, win_prob = parts[0], parts[1], float(parts[2])
                        else:
                            continue
                    else:
                        # Try to parse the whole item as a string
                        text = str(item)
                        parts = text.strip().split()
                        if len(parts) >= 3:
                            fen, move, win_prob = parts[0], parts[1], float(parts[2])
                        else:
                            continue
                else:
                    # Item is already a string
                    parts = str(item).strip().split()
                    if len(parts) >= 3:
                        fen, move, win_prob = parts[0], parts[1], float(parts[2])
                    else:
                        continue
                
                # Convert to the same format as ActionValueIterableDataset
                # Get action token
                action_token = self._action_to_token(move)
                if action_token is None:
                    continue
                
                # Tokenize FEN
                try:
                    board_tokens = self._fen_to_tokens(fen)
                except:
                    continue
                
                # Create sequence: board tokens + action token
                sequence = np.concatenate([board_tokens, np.array([action_token], dtype=np.int64)])
                target = np.array(win_prob, dtype=np.float32)
                
                yield (
                    torch.from_numpy(sequence),
                    torch.from_numpy(target),
                )
                
            except Exception as e:
                # Skip invalid records
                continue
    
    def _fen_to_tokens(self, fen: str) -> np.ndarray:
        """Convert FEN to tokens using the existing tokenizer."""
        board_tokens = tokenizer.tokenize(fen)
        if board_tokens.shape[0] != tokenizer.SEQUENCE_LENGTH:
            raise ValueError("Unexpected tokenized board length.")
        return board_tokens
    
    def _action_to_token(self, move: str) -> int | None:
        """Convert move to action token using existing move mapping."""
        action_id = moves.MOVE_TO_ACTION.get(move)
        if action_id is None:
            return None
        return tokenizer.VOCAB_SIZE + action_id  # Board vocab size + action id


def save_checkpoint_local(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    step: int,
    loss: float,
    config: Dict[str, Any],
    checkpoint_dir: str = "./checkpoints"
) -> str:
    """Save checkpoint locally."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pt")
    
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'config': config,
        'timestamp': time.time(),
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"💾 Local checkpoint saved: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(checkpoint_path: str, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: Optional[Any] = None) -> Tuple[int, float]:
    """Load checkpoint and return step and loss."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    step = checkpoint.get('step', 0)
    loss = checkpoint.get('loss', float('inf'))
    
    print(f"📂 Loaded checkpoint from step {step}, loss: {loss:.4f}")
    return step, loss


@app.function(
    image=image,
    volumes={
        "/checkpoints": checkpoints_vol,
        "/data": data_vol,
    },
    gpu="A100",  # Use A100 GPU for training
    timeout=3600 * 6,  # 6 hours timeout
    memory=32000,  # 32GB memory
)
def train_model(
    resume_from_step: int = 0,
    checkpoint_interval: int = 1000,
    max_steps: int = 50000,
    local_checkpoint_dir: str = "./checkpoints",
) -> Dict[str, Any]:
    """Main training function that runs on Modal."""
    
    print("🚀 Starting chess model training...")
    print(f"📊 Configuration: {CONFIG}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Import model classes in Modal environment
    sys.path.insert(0, "/app")
    try:
        from model.transformer_model import ActionValueTransformer, TransformerConfig
        from model.dataset import VOCAB_SIZE, SEQUENCE_LENGTH
        print("✅ Using existing transformer model")
        
        model_config = TransformerConfig(
            vocab_size=CONFIG["model"]["vocab_size"],
            max_seq_len=CONFIG["model"]["max_seq_len"],
            d_model=CONFIG["model"]["d_model"],
            num_layers=CONFIG["model"]["num_layers"],
            num_heads=CONFIG["model"]["num_heads"],
            dim_feedforward=CONFIG["model"]["dim_feedforward"],
            dropout=CONFIG["model"]["dropout"],
        )
        model = ActionValueTransformer(model_config).to(device)
    except ImportError as e:
        print(f"⚠️  Could not import existing model: {e}")
        return {"error": f"Could not import existing transformer model: {e}"}
    
    # Load dataset
    print("📥 Loading dataset...")
    try:
        dataset = load_dataset(CONFIG["data"]["dataset_name"], split=CONFIG["data"]["train_split"])
        print(f"✅ Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return {"error": f"Failed to load dataset: {e}"}
    
    # Create data loader using HuggingFace dataset
    train_dataset = HuggingFaceChessDataset(dataset, max_length=CONFIG["data"]["max_length"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["training"]["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False,
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📈 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["training"]["learning_rate"],
        weight_decay=CONFIG["training"]["weight_decay"]
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_steps,
        eta_min=CONFIG["training"]["learning_rate"] * 0.1
    )
    
    # Load checkpoint if resuming
    current_step = resume_from_step
    if resume_from_step > 0:
        checkpoint_path = f"/checkpoints/checkpoint_step_{resume_from_step}.pt"
        if os.path.exists(checkpoint_path):
            current_step, _ = load_checkpoint(checkpoint_path, model, optimizer, scheduler)
        else:
            print(f"⚠️  Checkpoint not found at step {resume_from_step}, starting fresh")
            current_step = 0
    
    # Training loop
    model.train()
    print("🎯 Starting training loop...")
    
    # Use BCEWithLogitsLoss for win probability prediction (same as existing train_engine.py)
    loss_fn = nn.BCEWithLogitsLoss()
    running_loss = 0.0
    log_interval = CONFIG["training"]["log_interval"]
    
    data_iter = iter(train_loader)
    
    while current_step < max_steps:
        try:
            # Get next batch
            try:
                batch = next(data_iter)
            except StopIteration:
                # Restart data iterator
                data_iter = iter(train_loader)
                batch = next(data_iter)
            
            # Unpack batch - expecting (sequences, targets)
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                sequences, targets = batch
            else:
                # Skip malformed batches
                continue
            
            # Move to device
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(sequences)
            
            # Compute loss
            loss = loss_fn(logits, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["training"]["gradient_clip"])
            
            # Optimizer step
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            running_loss += loss.item()
            current_step += 1
            
            # Logging
            if current_step % log_interval == 0:
                avg_loss = running_loss / log_interval
                lr = scheduler.get_last_lr()[0]
                # Calculate accuracy metrics like the existing trainer
                with torch.no_grad():
                    probs = torch.sigmoid(logits)
                    mae = torch.mean(torch.abs(probs - targets)).item()
                print(f"📊 Step {current_step}/{max_steps} | Loss: {avg_loss:.4f} | MAE: {mae:.4f} | LR: {lr:.6f}")
                running_loss = 0.0
            
            # Checkpointing
            if current_step % checkpoint_interval == 0:
                # Save to Modal volume
                modal_checkpoint_path = f"/checkpoints/checkpoint_step_{current_step}.pt"
                checkpoint = {
                    'step': current_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss.item(),
                    'config': CONFIG,
                    'timestamp': time.time(),
                }
                
                torch.save(checkpoint, modal_checkpoint_path)
                checkpoints_vol.commit()  # Persist to Modal
                print(f"💾 Modal checkpoint saved: {modal_checkpoint_path}")
                
                # Also save locally if path provided
                if local_checkpoint_dir:
                    save_checkpoint_local(
                        model, optimizer, scheduler, current_step, loss.item(), CONFIG, local_checkpoint_dir
                    )
        
        except Exception as e:
            print(f"❌ Error during training step {current_step}: {e}")
            continue  # Continue training instead of breaking
    
    # Final checkpoint
    final_checkpoint_path = f"/checkpoints/final_checkpoint_step_{current_step}.pt"
    final_checkpoint = {
        'step': current_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss.item() if 'loss' in locals() else 0.0,
        'config': CONFIG,
        'timestamp': time.time(),
    }
    
    torch.save(final_checkpoint, final_checkpoint_path)
    checkpoints_vol.commit()
    print(f"🏁 Final checkpoint saved: {final_checkpoint_path}")
    
    return {
        "final_step": current_step,
        "checkpoint_path": final_checkpoint_path,
        "status": "completed"
    }


@app.function(
    image=image,
    volumes={"/checkpoints": checkpoints_vol},
)
def list_checkpoints() -> list[str]:
    """List available checkpoints in Modal storage."""
    checkpoint_dir = "/checkpoints"
    if not os.path.exists(checkpoint_dir):
        return []
    
    checkpoints = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.pt'):
            checkpoints.append(file)
    
    return sorted(checkpoints)


@app.function(
    image=image,
    volumes={"/checkpoints": checkpoints_vol},
)
def download_checkpoint(checkpoint_name: str, local_path: str = "./") -> str:
    """Download a checkpoint from Modal storage to local machine."""
    modal_path = f"/checkpoints/{checkpoint_name}"
    
    if not os.path.exists(modal_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_name} not found in Modal storage")
    
    # Copy to local path would happen through Modal's file system
    # In practice, you'd use Modal's file download capabilities
    return f"Checkpoint {checkpoint_name} ready for download"


# CLI functions for easy usage
@app.local_entrypoint()
def main(
    resume_from_step: int = 0,
    checkpoint_interval: int = 1000,
    max_steps: int = 50000,
    local_checkpoint_dir: str = "./checkpoints",
):
    """Main entry point for training."""
    print("🌟 Chess Training with Modal")
    print("=" * 50)
    
    # Start training
    result = train_model.remote(
        resume_from_step=resume_from_step,
        checkpoint_interval=checkpoint_interval,
        max_steps=max_steps,
        local_checkpoint_dir=local_checkpoint_dir,
    )
    
    print(f"✅ Training completed: {result}")
    
    # List available checkpoints
    checkpoints = list_checkpoints.remote()
    print(f"📁 Available checkpoints: {checkpoints}")


if __name__ == "__main__":
    # You can also run specific functions
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "train":
            # Parse additional arguments
            resume_from = int(sys.argv[2]) if len(sys.argv) > 2 else 0
            max_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 50000
            checkpoint_interval = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
            
            result = train_model.remote(
                resume_from_step=resume_from,
                max_steps=max_steps,
                checkpoint_interval=checkpoint_interval,
            )
            print(result)
            
        elif command == "list":
            checkpoints = list_checkpoints.remote()
            print("Available checkpoints:")
            for cp in checkpoints:
                print(f"  - {cp}")
                
        elif command == "config":
            print("Current configuration:")
            import json
            print(json.dumps(CONFIG, indent=2))
            
        else:
            print("Available commands: train, list, config")
            print("Usage examples:")
            print("  python modal_train.py train                    # Start training from scratch")
            print("  python modal_train.py train 5000               # Resume from step 5000")
            print("  python modal_train.py train 0 100000 500       # Train for 100k steps, checkpoint every 500")
            print("  python modal_train.py list                     # List checkpoints")
            print("  python modal_train.py config                   # Show configuration")
    else:
        print("🚀 Starting training with default settings...")
        main()