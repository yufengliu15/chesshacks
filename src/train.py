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
import numpy as np

# Add the src directory to path for imports
src_path = Path(__file__).parent  # This is already the src/ directory
sys.path.insert(0, str(src_path))

# Import the existing transformer model and dataset utilities
try:
    from model.transformer_model import ActionValueTransformer, TransformerConfig
    from model.dataset import ActionValueIterableDataset, VOCAB_SIZE, SEQUENCE_LENGTH
    LOCAL_MODEL_AVAILABLE = True
except ImportError:
    LOCAL_MODEL_AVAILABLE = False
    print("Warning: Local model imports failed. Using fallback transformer.")

# Modal setup
app = modal.App("chess-training")

# Modal volumes for persistent storage
checkpoints_vol = modal.Volume.from_name("chess-checkpoints", create_if_missing=True)
data_vol = modal.Volume.from_name("chess-data", create_if_missing=True)

# Get the src directory path
src_dir = Path(__file__).parent

# Container image with required dependencies
image = (
    modal.Image.debian_slim()
    .pip_install([
        "torch>=2.0.0",
        "tqdm",
        "numpy",
        "python-chess",
        "zstandard",  # For reading .bag files
        "huggingface-hub",  # For downloading dataset from HuggingFace
    ])
    .add_local_file(str(src_dir / "bagz.py"), "/root/bagz.py")
    .add_local_dir(str(src_dir / "model"), remote_path="/root/model")
)

# Training configuration - adjusted for the existing model
CONFIG = {
    "model": {
        "vocab_size": VOCAB_SIZE if LOCAL_MODEL_AVAILABLE else 32000,
        "max_seq_len": SEQUENCE_LENGTH if LOCAL_MODEL_AVAILABLE else 1024,
        "d_model": 128,
        "num_layers": 4,
        "num_heads": 8,
        "dim_feedforward": 256,
        "dropout": 0.1,
    },
    "training": {
        "batch_size": 512,
        "learning_rate": 1e-3,
        "weight_decay": 0.01,
        "warmup_steps": 1000,
        "max_steps": 6000,
        "gradient_clip": 1.0,
        "eval_interval": 500,
        "log_interval": 500,
        "checkpoint_interval": 500,  # Save checkpoint every N steps
        "max_records": 100000,  # Records per epoch
    },
    "data": {
        # Dataset configuration
        "hf_repo_id": "chimcis/searcless-chess-10m",  # HuggingFace dataset repository
        "data_dir": "/data",  # Directory to download/store .bag files in Modal
        "bag_prefix": "action_value",  # Prefix for bag files
        "bag_suffix": "_data.bag",  # Suffix for bag files
        "num_shards": 2148,  # Total number of shards in dataset
        "num_shards_to_use": 100,  # Number of shards to actually use (for faster training/testing)
        "max_records": 100000,  # Records per epoch
        "shuffle_files": True,
        "seed": 42,
    }
}


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
    return step, loss


def clear_old_checkpoints(checkpoint_dir: str = "/checkpoints") -> dict:
    """Delete all old checkpoints from the checkpoint directory."""
    if not os.path.exists(checkpoint_dir):
        return {'deleted_count': 0, 'total_size_mb': 0}
    
    deleted_count = 0
    total_size = 0
    
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.pt'):
            file_path = os.path.join(checkpoint_dir, file)
            file_size = os.path.getsize(file_path)
            total_size += file_size
            os.remove(file_path)
            deleted_count += 1
    
    # Commit changes to volume
    if deleted_count > 0:
        checkpoints_vol.commit()
    
    return {
        'deleted_count': deleted_count,
        'total_size_mb': total_size / 1024 / 1024
    }


@app.function(
    image=image,
    volumes={
        "/checkpoints": checkpoints_vol,
        "/data": data_vol,
    },
    gpu="H100",  # Use A100 GPU for training
    timeout=3600 * 6,  # 6 hours timeout
    memory=32000,  # 32GB memory
)
def train_model(
    resume_from_step: int = 0,
    checkpoint_interval: int = 1000,
    max_steps: int = 50000,
    local_checkpoint_dir: str = "./checkpoints",
    clear_checkpoints: bool = True,  # New parameter to control checkpoint deletion
) -> Dict[str, Any]:
    """Main training function that runs on Modal."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Clear old checkpoints if starting fresh (not resuming)
    if clear_checkpoints and resume_from_step == 0:
        print("🗑️  Clearing old checkpoints from previous training sessions...")
        result = clear_old_checkpoints()
        if result['deleted_count'] > 0:
            print(f"   ✅ Deleted {result['deleted_count']} old checkpoint(s)")
            print(f"   Freed: {result['total_size_mb']:.2f} MB")
        else:
            print("   No old checkpoints to delete.")
    elif resume_from_step > 0:
        print(f"📂 Resuming from step {resume_from_step}, keeping existing checkpoints...")
    
    # Import model classes in Modal environment
    try:
        from model.transformer_model import ActionValueTransformer, TransformerConfig
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
        import traceback
        traceback.print_exc()
        return {"error": f"Could not import existing transformer model: {e}"}
    
    # Download dataset from HuggingFace if needed
    print("📥 Ensuring dataset is available...")
    data_dir = Path(CONFIG["data"]["data_dir"])
    bag_prefix = CONFIG["data"]["bag_prefix"]
    bag_suffix = CONFIG["data"]["bag_suffix"]
    num_shards = CONFIG["data"]["num_shards"]
    num_shards_to_use = CONFIG["data"]["num_shards_to_use"]
    
    # Download bag files from HuggingFace
    from huggingface_hub import snapshot_download
    
    # Create list of files to download
    action_files = [
        f"{bag_prefix}-{idx:05d}-of-{num_shards:05d}{bag_suffix}"
        for idx in range(num_shards_to_use)
    ]
    
    print(f"📥 Downloading {len(action_files)} bag files from HuggingFace...")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    snapshot_download(
        repo_id=CONFIG["data"]["hf_repo_id"],
        repo_type="dataset",
        local_dir=str(data_dir),
        local_dir_use_symlinks=False,
        allow_patterns=action_files,
        resume_download=True,
    )
    print(f"✅ Dataset downloaded to {data_dir}")
    
    # Load local bag files
    print("📥 Loading local .bag files...")
    from model.dataset import ActionValueIterableDataset
    
    # Generate bag file paths and verify they exist
    # Format: action_value-00000-of-02148_data.bag, action_value-00001-of-02148_data.bag, etc.
    bag_files = []
    skipped_files = []
    
    for shard_idx in range(num_shards_to_use):
        filename = f"{bag_prefix}-{shard_idx:05d}-of-{num_shards:05d}{bag_suffix}"
        filepath = data_dir / filename
        
        # Check if file exists before adding it
        if filepath.exists():
            bag_files.append(filepath)
        else:
            skipped_files.append(filename)
            print(f"⚠️  Skipping missing file: {filename}")
    
    if not bag_files:
        print(f"❌ No bag files found in {data_dir}")
        return {"error": f"No bag files found in {data_dir}"}
    
    print(f"✅ Loaded {len(bag_files)} bag files (skipped {len(skipped_files)} missing)")
    print(f"   First file: {bag_files[0].name}")
    print(f"   Last file: {bag_files[-1].name}")
    
    # Create dataset from bag files
    train_dataset = ActionValueIterableDataset(
        bag_paths=bag_files,  # Already Path objects
        max_records=CONFIG["data"]["max_records"],
        shuffle_files=CONFIG["data"]["shuffle_files"],
        seed=CONFIG["data"]["seed"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["training"]["batch_size"],
        num_workers=0,  # IterableDataset works best with num_workers=0
        pin_memory=True if device.type == 'cuda' else False,
    )
    
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
            current_step = 0

    # Training loop
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    running_loss = 0.0
    log_interval = CONFIG["training"]["log_interval"]
    data_iter = iter(train_loader)
    
    # Store initial weights to verify they change during training
    initial_embedding_weights = model.embedding.weight.data.clone()
    initial_head_weights = list(model.head.parameters())[0].data.clone()
    print(f"📊 Initial embedding weights: mean={initial_embedding_weights.mean().item():.6f}, std={initial_embedding_weights.std().item():.6f}")

    while current_step < max_steps:
        try:
            # Get next batch
            try:
                batch = next(data_iter)
            except StopIteration:
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
            
            # Ensure targets are 1D [batch_size] not [batch_size, 1]
            # This prevents shape mismatch with BCEWithLogitsLoss
            if targets.ndim > 1:
                targets = targets.squeeze(-1)

            # Forward pass
            optimizer.zero_grad()
            logits = model(sequences)

            # Compute loss
            loss = loss_fn(logits, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["training"]["gradient_clip"])
            
            # Check for gradient issues every N steps
            if current_step % (log_interval * 10) == 0:
                # Check a sample of gradients
                if model.embedding.weight.grad is not None:
                    emb_grad_norm = model.embedding.weight.grad.norm().item()
                    emb_grad_mean = model.embedding.weight.grad.mean().item()
                    if emb_grad_norm < 1e-7:
                        print(f"  🚨 WARNING: Embedding gradients near zero! (norm={emb_grad_norm:.2e})")
                    elif emb_grad_norm > 100:
                        print(f"  ⚠️  WARNING: Very large embedding gradients! (norm={emb_grad_norm:.2e})")

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
                
                # Calculate MAE and output diversity metrics
                with torch.no_grad():
                    probs = torch.sigmoid(logits)
                    mae = torch.mean(torch.abs(probs - targets)).item()
                    
                    # Output diversity diagnostics - helps detect model collapse
                    logit_mean = logits.mean().item()
                    logit_std = logits.std().item()
                    prob_mean = probs.mean().item()
                    prob_std = probs.std().item()
                    prob_min = probs.min().item()
                    prob_max = probs.max().item()
                    
                    # Target diversity for comparison
                    target_mean = targets.mean().item()
                    target_std = targets.std().item()

                print(f"Step {current_step}/{max_steps} | Loss: {avg_loss:.4f} | MAE: {mae:.4f} | LR: {lr:.6f}")
                print(f"  Output → Logits: μ={logit_mean:+.3f} σ={logit_std:.3f} | Probs: μ={prob_mean:.3f} σ={prob_std:.3f} range=[{prob_min:.3f}, {prob_max:.3f}]")
                print(f"  Target → μ={target_mean:.3f} σ={target_std:.3f}")
                
                # Check if weights are actually changing (crucial diagnostic!)
                current_embedding_weights = model.embedding.weight.data
                current_head_weights = list(model.head.parameters())[0].data
                emb_weight_diff = torch.norm(current_embedding_weights - initial_embedding_weights).item()
                head_weight_diff = torch.norm(current_head_weights - initial_head_weights).item()
                
                print(f"  Weights → Embedding Δ={emb_weight_diff:.4f} | Head Δ={head_weight_diff:.4f}")
                
                # Critical warning if weights aren't changing!
                if emb_weight_diff < 1e-4 and head_weight_diff < 1e-4:
                    print(f"  🚨 CRITICAL: Weights are NOT changing! Training may be broken!")
                elif emb_weight_diff < 0.01:
                    print(f"  ⚠️  WARNING: Weights changing very slowly (Δ={emb_weight_diff:.4f})")
                
                # Warning if output diversity is too low (model collapse)
                if logit_std < 0.1:
                    print(f"  🚨 ALERT: Very low output diversity (σ={logit_std:.3f}) - model is collapsing to mean!")
                elif logit_std < 0.3:
                    print(f"  ⚠️  WARNING: Low output diversity (σ={logit_std:.3f}) - watch for collapse!")
                elif logit_std > 2.0:
                    print(f"  ⚠️  WARNING: Very high diversity (σ={logit_std:.3f}) - model may be unstable!")
                
                running_loss = 0.0

            # Checkpointing
            if current_step % checkpoint_interval == 0:
                print(f"\n💾 Saving checkpoint at step {current_step}...")
                
                # Verify weights have changed before saving
                current_emb_stats = model.embedding.weight.data
                emb_mean = current_emb_stats.mean().item()
                emb_std = current_emb_stats.std().item()
                emb_diff_from_init = torch.norm(current_emb_stats - initial_embedding_weights).item()
                
                print(f"   Current weights: Embedding μ={emb_mean:.6f} σ={emb_std:.6f} | Δ from init={emb_diff_from_init:.4f}")
                
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
                    # Add weight statistics for verification
                    'weight_stats': {
                        'embedding_mean': emb_mean,
                        'embedding_std': emb_std,
                        'embedding_change': emb_diff_from_init,
                    }
                }

                torch.save(checkpoint, modal_checkpoint_path)
                
                # Get checkpoint file size
                checkpoint_size = os.path.getsize(modal_checkpoint_path) / 1024 / 1024
                print(f"   ✅ Saved to Modal: checkpoint_step_{current_step}.pt ({checkpoint_size:.2f} MB)")
                
                # Sanity check: verify we can load it back
                verify_checkpoint = torch.load(modal_checkpoint_path, map_location='cpu')
                if verify_checkpoint['step'] != current_step:
                    print(f"   🚨 WARNING: Saved step {verify_checkpoint['step']} != current step {current_step}!")
                
                checkpoints_vol.commit()
                print(f"   ✅ Committed to Modal volume")

                # Also save locally if path provided
                if local_checkpoint_dir:
                    local_path = save_checkpoint_local(
                        model, optimizer, scheduler, current_step, loss.item(), CONFIG, local_checkpoint_dir
                    )
                    print(f"   ✅ Saved locally: {os.path.basename(local_path)}")
                
                print()  # Empty line for readability

        except Exception as e:
            print(f"❌ Error during training step {current_step}: {e}")
            print(f"❌ Error type: {type(e).__name__}")
            import traceback
            print("❌ Full traceback:")
            traceback.print_exc()
            continue  # Continue training instead of breaking
    
    # Final checkpoint
    print(f"\n💾 Saving final checkpoint at step {current_step}...")
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
    
    # Get checkpoint file size
    final_checkpoint_size = os.path.getsize(final_checkpoint_path) / 1024 / 1024
    print(f"   ✅ Saved to Modal: final_checkpoint_step_{current_step}.pt ({final_checkpoint_size:.2f} MB)")
    
    checkpoints_vol.commit()
    print(f"   ✅ Committed to Modal volume")
    print(f"\n🎉 Training completed!")

    return {
        "final_step": current_step,
        "checkpoint_path": final_checkpoint_path,
        "checkpoint_size_mb": final_checkpoint_size,
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
    clear_checkpoints: bool = True,
):
    """Main entry point for training.
    
    Args:
        resume_from_step: Step to resume from (0 for fresh start)
        checkpoint_interval: Save checkpoint every N steps
        max_steps: Maximum training steps
        local_checkpoint_dir: Local directory to save checkpoints
        clear_checkpoints: Delete old checkpoints when starting fresh (default: True)
    """
    print("🌟 Chess Training with Modal")
    print("=" * 50)
    
    # Start training
    result = train_model.remote(
        resume_from_step=resume_from_step,
        checkpoint_interval=checkpoint_interval,
        max_steps=max_steps,
        local_checkpoint_dir=local_checkpoint_dir,
        clear_checkpoints=clear_checkpoints,
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
                clear_checkpoints=True,  # Always clear when using CLI
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
            print("  python train.py train                    # Start fresh (clears old checkpoints)")
            print("  python train.py train 5000               # Resume from step 5000 (keeps checkpoints)")
            print("  python train.py train 0 100000 500       # Train for 100k steps, checkpoint every 500")
            print("  python train.py list                     # List checkpoints")
            print("  python train.py config                   # Show configuration")
            print("\nNote: Starting fresh training automatically deletes old checkpoints")
    else:
        print("🚀 Starting training with default settings...")
        main()