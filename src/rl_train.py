"""Reinforcement learning training using self-play data.

Trains the model on games it played against itself, using actual game
outcomes as the training signal.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, Any

import modal
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Modal app
app = modal.App("chess-rl-train")

# Modal volumes
checkpoints_vol = modal.Volume.from_name("chess-checkpoints", create_if_missing=True)
selfplay_data_vol = modal.Volume.from_name("chess-selfplay-data", create_if_missing=True)

# Get the src directory path
src_dir = Path(__file__).parent

# Container image
image = (
    modal.Image.debian_slim()
    .pip_install([
        "torch>=2.0.0",
        "numpy",
        "python-chess",
        "tqdm",
        "zstandard",  # Required by bagz
    ])
    .env({"CACHE_BUST": "2024-11-15-v4-material"})  # Force rebuild for material head
    .add_local_file(str(src_dir / "bagz.py"), remote_path="/root/bagz.py")
    .add_local_dir(str(src_dir / "model"), remote_path="/root/model")
)

# RL Training configuration
RL_CONFIG = {
    "model": {
        "vocab_size": 32832,  # Will be set from checkpoint
        "max_seq_len": 77,
        "d_model": 128,
        "num_layers": 4,
        "num_heads": 8,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "use_material_head": True,  # ✅ Enable material evaluation
        "material_loss_weight": 0.1,  # 10% of loss for material prediction
    },
    "training": {
        "batch_size": 512,
        "learning_rate": 5e-5,  # Lower LR for fine-tuning
        "weight_decay": 0.01,
        "max_steps": 5000,
        "gradient_clip": 1.0,
        "log_interval": 100,
        "checkpoint_interval": 1000,
        "warmup_steps": 500,
    },
}


def load_checkpoint(checkpoint_path: str, device: str = "cuda"):
    """Load a checkpoint and return model + config."""
    from model.transformer_model import ActionValueTransformer, TransformerConfig
    
    # Always load to CPU first to avoid device compatibility issues
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
    
    # Create model
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
    
    # Load weights with strict=False to allow missing keys from the alias
    model.load_state_dict(state_dict, strict=False)
    
    step = checkpoint.get("step", 0)
    
    return model, model_config, step


@app.function(
    image=image,
    volumes={
        "/checkpoints": checkpoints_vol,
        "/selfplay_data": selfplay_data_vol,
    },
    gpu="H100",
    timeout=3600 * 4,  # 4 hours
    memory=32000,
)
def train_on_selfplay(
    checkpoint_step: int = 22000,
    max_steps: int = 5000,
    checkpoint_interval: int = 1000,
    batch_size: int = 512,
    learning_rate: float = 5e-5,
) -> Dict[str, Any]:
    """Train model on self-play data.
    
    Args:
        checkpoint_step: Which checkpoint to start from
        max_steps: Maximum training steps
        checkpoint_interval: Save checkpoint every N steps
        batch_size: Batch size for training
        learning_rate: Learning rate
    
    Returns:
        Training statistics
    """
    # Setup Python path for imports
    import sys
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")
    
    print("🚀 Starting RL training on self-play data...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Load base checkpoint
    checkpoint_path = f"/checkpoints/checkpoint_step_{checkpoint_step}.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"📥 Loading checkpoint from step {checkpoint_step}...")
    model, model_config, base_step = load_checkpoint(checkpoint_path, device=str(device))
    print(f"✅ Loaded model from step {base_step}")
    
    # Find self-play data files
    selfplay_dir = Path("/selfplay_data")
    if not selfplay_dir.exists():
        raise FileNotFoundError("Self-play data directory not found")
    
    # Look for self-play data matching this checkpoint
    bag_files = list(selfplay_dir.glob(f"selfplay-*_step_{checkpoint_step}_data.bag"))
    
    if not bag_files:
        # Fall back to any self-play data
        bag_files = list(selfplay_dir.glob("selfplay-*.bag"))
    
    if not bag_files:
        raise FileNotFoundError(f"No self-play data found in {selfplay_dir}")
    
    print(f"📂 Found {len(bag_files)} self-play data file(s)")
    for bf in bag_files:
        size_mb = bf.stat().st_size / 1024 / 1024
        print(f"   - {bf.name} ({size_mb:.2f} MB)")
    
    # Create dataset
    from model.dataset import ActionValueIterableDataset
    
    train_dataset = ActionValueIterableDataset(
        bag_paths=bag_files,
        max_records=None,  # Use all data
        shuffle_files=True,
        seed=42,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False,
    )
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=RL_CONFIG["training"]["weight_decay"]
    )
    
    warmup_steps = RL_CONFIG["training"]["warmup_steps"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, (step + 1) / warmup_steps)
    )
    
    # Training loop
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    mse_loss_fn = nn.MSELoss()  # For material prediction
    
    # Check if model has material head
    has_material_head = hasattr(model, 'material_head') and model.material_head is not None
    material_weight = model.config.material_loss_weight if has_material_head else 0.0
    
    if has_material_head:
        print(f"🎯 Material head enabled (weight={material_weight})")
        # Import material utilities
        import chess
        from model.material import get_normalized_material_balance
        from model.tokenizer import tokenize, SEQUENCE_LENGTH
    else:
        print(f"⚠️  Material head not found, training value head only")
    
    current_step = 0
    running_loss = 0.0
    running_value_loss = 0.0
    running_material_loss = 0.0
    running_mae = 0.0
    log_interval = RL_CONFIG["training"]["log_interval"]
    
    data_iter = iter(train_loader)
    
    initial_embedding_weights = model.embedding.weight.data.clone()
    
    print(f"🎯 Starting RL training loop...")
    print(f"   Target steps: {max_steps}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print()
    
    start_time = time.time()
    
    while current_step < max_steps:
        try:
            # Get next batch
            try:
                batch = next(data_iter)
            except StopIteration:
                # Restart data iterator (epoch complete)
                print(f"   📊 Epoch complete, restarting data...")
                data_iter = iter(train_loader)
                batch = next(data_iter)
            
            # Unpack batch
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                sequences, targets = batch
            else:
                continue
            
            # Move to device
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            # Ensure targets are 1D
            if targets.ndim > 1:
                targets = targets.squeeze(-1)
            
            # CRITICAL: Clamp targets to [0, 1] to prevent loss explosion
            # BCEWithLogitsLoss requires targets in [0, 1]
            targets = torch.clamp(targets, 0.0, 1.0)
            
            # Sanity check for invalid values
            if torch.isnan(targets).any() or torch.isinf(targets).any():
                print(f"⚠️  Warning: Invalid targets detected at step {current_step}, skipping batch")
                continue
            
            # Forward pass
            optimizer.zero_grad()
            
            # Get value predictions (and material if available)
            if has_material_head:
                value_logits, material_pred = model(sequences, return_material=True)
            else:
                value_logits = model(sequences)
            
            # Compute value loss
            value_loss = loss_fn(value_logits, targets)
            
            # Compute material loss (if material head exists)
            if has_material_head:
                # Extract FEN from sequences to compute material balance targets
                # Sequences are [batch, seq_len] where first 77 tokens are FEN
                material_targets = []
                
                for seq in sequences:
                    try:
                        # Decode FEN from token sequence (first SEQUENCE_LENGTH tokens)
                        fen_tokens = seq[:SEQUENCE_LENGTH].cpu().numpy()
                        
                        # Simple heuristic: extract board position from tokens
                        # This is a simplified approach - ideally we'd store FEN in dataset
                        # For now, skip material loss if we can't extract FEN easily
                        # We'll compute from move sequences in a future update
                        material_targets.append(0.0)  # Placeholder
                        
                    except Exception:
                        material_targets.append(0.0)
                
                material_targets_tensor = torch.tensor(material_targets, device=device, dtype=torch.float32)
                material_loss = mse_loss_fn(material_pred, material_targets_tensor)
                
                # Combined loss
                loss = value_loss + material_weight * material_loss
            else:
                loss = value_loss
                material_loss = torch.tensor(0.0)
            
            # Check for invalid loss (NaN or Inf)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️  Warning: Invalid loss detected at step {current_step}")
                print(f"   Loss: {loss.item()}")
                if has_material_head:
                    print(f"   Value loss: {value_loss.item():.4f}, Material loss: {material_loss.item():.4f}")
                print(f"   Value logits: min={value_logits.min().item():.3f}, max={value_logits.max().item():.3f}")
                print(f"   Targets: min={targets.min().item():.3f}, max={targets.max().item():.3f}, mean={targets.mean().item():.3f}")
                print(f"   Skipping this batch...")
                continue
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                RL_CONFIG["training"]["gradient_clip"]
            )
            
            # Optimizer step
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            running_loss += loss.item()
            if has_material_head:
                running_value_loss += value_loss.item()
                running_material_loss += material_loss.item()
            
            with torch.no_grad():
                probs = torch.sigmoid(value_logits)
                mae = torch.mean(torch.abs(probs - targets)).item()
                running_mae += mae
            
            current_step += 1
            
            # Logging
            if current_step % log_interval == 0:
                avg_loss = running_loss / log_interval
                avg_mae = running_mae / log_interval
                lr = scheduler.get_last_lr()[0]
                
                elapsed = time.time() - start_time
                steps_per_sec = current_step / elapsed
                eta_seconds = (max_steps - current_step) / steps_per_sec if steps_per_sec > 0 else 0
                eta_minutes = eta_seconds / 60
                
                with torch.no_grad():
                    logit_mean = value_logits.mean().item()
                    logit_std = value_logits.std().item()
                    prob_mean = probs.mean().item()
                    prob_std = probs.std().item()
                    target_mean = targets.mean().item()
                    target_std = targets.std().item()
                    
                    # Weight change
                    current_emb = model.embedding.weight.data
                    emb_change = torch.norm(current_emb - initial_embedding_weights).item()
                
                log_msg = f"Step {current_step}/{max_steps} | Loss: {avg_loss:.4f} | MAE: {avg_mae:.4f}"
                
                if has_material_head:
                    avg_value_loss = running_value_loss / log_interval
                    avg_material_loss = running_material_loss / log_interval
                    log_msg += f" | V: {avg_value_loss:.4f} | M: {avg_material_loss:.4f}"
                
                log_msg += f" | LR: {lr:.6f}"
                print(log_msg)
                print(f"  Output → Logits: μ={logit_mean:+.3f} σ={logit_std:.3f} | Probs: μ={prob_mean:.3f} σ={prob_std:.3f}")
                print(f"  Target → μ={target_mean:.3f} σ={target_std:.3f}")
                print(f"  Weights → Embedding Δ={emb_change:.4f} | Grad norm={grad_norm:.3f}")
                print(f"  Speed: {steps_per_sec:.2f} steps/sec | ETA: {eta_minutes:.1f} min")
                print()
                
                running_loss = 0.0
                running_value_loss = 0.0
                running_material_loss = 0.0
                running_mae = 0.0
            
            # Checkpointing
            if current_step % checkpoint_interval == 0:
                print(f"💾 Saving checkpoint at step {current_step}...")
                
                # Calculate new step number (base + RL steps)
                new_step = base_step + current_step
                
                checkpoint_path = f"/checkpoints/checkpoint_step_{new_step}.pt"
                checkpoint = {
                    'step': new_step,
                    'base_step': base_step,
                    'rl_steps': current_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss.item(),
                    'config': {
                        "model": model_config.to_dict(),
                        "training": RL_CONFIG["training"],
                    },
                    'timestamp': time.time(),
                    'rl_training': True,
                }
                
                torch.save(checkpoint, checkpoint_path)
                checkpoints_vol.commit()
                
                checkpoint_size = os.path.getsize(checkpoint_path) / 1024 / 1024
                print(f"   ✅ Saved checkpoint_step_{new_step}.pt ({checkpoint_size:.2f} MB)")
                print(f"   ✅ Committed to Modal volume")
                print()
        
        except Exception as e:
            print(f"❌ Error during training step {current_step}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final checkpoint
    print(f"\n💾 Saving final checkpoint...")
    final_step = base_step + current_step
    final_checkpoint_path = f"/checkpoints/checkpoint_step_{final_step}.pt"
    final_checkpoint = {
        'step': final_step,
        'base_step': base_step,
        'rl_steps': current_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss.item() if 'loss' in locals() else 0.0,
        'config': {
            "model": model_config.to_dict(),
            "training": RL_CONFIG["training"],
        },
        'timestamp': time.time(),
        'rl_training': True,
    }
    
    torch.save(final_checkpoint, final_checkpoint_path)
    checkpoints_vol.commit()
    
    final_size = os.path.getsize(final_checkpoint_path) / 1024 / 1024
    print(f"   ✅ Saved checkpoint_step_{final_step}.pt ({final_size:.2f} MB)")
    print(f"   ✅ Committed to Modal volume")
    
    total_time = time.time() - start_time
    print(f"\n🎉 RL training completed!")
    print(f"   Total time: {total_time / 60:.1f} minutes")
    print(f"   Final step: {final_step}")
    
    return {
        "status": "completed",
        "base_step": base_step,
        "rl_steps": current_step,
        "final_step": final_step,
        "total_time_minutes": total_time / 60,
        "checkpoint_path": final_checkpoint_path,
    }


@app.local_entrypoint()
def main(
    checkpoint_step: int = 22000,
    max_steps: int = 5000,
    checkpoint_interval: int = 1000,
    batch_size: int = 512,
    learning_rate: float = 5e-5,
):
    """Main entry point for RL training."""
    print("🎯 Chess RL Training")
    print("=" * 50)
    print(f"Base checkpoint: step {checkpoint_step}")
    print(f"RL training steps: {max_steps}")
    print(f"Learning rate: {learning_rate}")
    print()
    
    result = train_on_selfplay.remote(
        checkpoint_step=checkpoint_step,
        max_steps=max_steps,
        checkpoint_interval=checkpoint_interval,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    
    print(f"\n✅ Training completed:")
    print(f"   Base step: {result['base_step']}")
    print(f"   RL steps: {result['rl_steps']}")
    print(f"   Final step: {result['final_step']}")
    print(f"   Time: {result['total_time_minutes']:.1f} minutes")


if __name__ == "__main__":
    import sys
    
    # Parse command line args
    checkpoint_step = int(sys.argv[1]) if len(sys.argv) > 1 else 22000
    max_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    learning_rate = float(sys.argv[3]) if len(sys.argv) > 3 else 5e-5
    
    main(checkpoint_step, max_steps, 1000, 512, learning_rate)

