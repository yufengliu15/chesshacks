"""Diagnose checkpoint to check if model is learning properly."""

import torch
import numpy as np
from pathlib import Path

def load_checkpoint(checkpoint_path: str):
    """Load and inspect a checkpoint."""
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        print(f"❌ Checkpoint not found: {ckpt_path}")
        return None
    
    print(f"📥 Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    # Print basic info
    print("\n" + "="*60)
    print("CHECKPOINT INFO")
    print("="*60)
    
    if "step" in checkpoint:
        print(f"Step: {checkpoint['step']}")
    if "loss" in checkpoint:
        print(f"Loss: {checkpoint['loss']:.6f}")
    if "timestamp" in checkpoint:
        from datetime import datetime
        print(f"Timestamp: {datetime.fromtimestamp(checkpoint['timestamp'])}")
    
    # Check config
    if "config" in checkpoint:
        print("\nModel Config:")
        if "model" in checkpoint["config"]:
            for k, v in checkpoint["config"]["model"].items():
                print(f"  {k}: {v}")
    
    return checkpoint


def diagnose_model_weights(checkpoint):
    """Check if model weights look reasonable."""
    if "model_state_dict" not in checkpoint:
        print("\n❌ No model_state_dict found in checkpoint!")
        return
    
    state_dict = checkpoint["model_state_dict"]
    
    print("\n" + "="*60)
    print("WEIGHT DIAGNOSTICS")
    print("="*60)
    
    # Check embedding layer
    if "embedding.weight" in state_dict:
        emb_weights = state_dict["embedding.weight"]
        print(f"\n📊 Embedding Layer:")
        print(f"  Shape: {emb_weights.shape}")
        print(f"  Mean: {emb_weights.mean().item():.6f}")
        print(f"  Std: {emb_weights.std().item():.6f}")
        print(f"  Min: {emb_weights.min().item():.6f}")
        print(f"  Max: {emb_weights.max().item():.6f}")
        
        # Check if embeddings are diverse
        # Compare first 10 embeddings
        if emb_weights.shape[0] >= 10:
            sample_embs = emb_weights[:10]
            # Calculate pairwise distances
            dists = []
            for i in range(10):
                for j in range(i+1, 10):
                    dist = torch.norm(sample_embs[i] - sample_embs[j]).item()
                    dists.append(dist)
            avg_dist = np.mean(dists)
            print(f"  Avg pairwise distance (first 10): {avg_dist:.6f}")
            
            if avg_dist < 0.1:
                print("  ⚠️  WARNING: Embeddings are very similar! Model may not be learning.")
    
    # Check positional embeddings
    if "pos_embedding" in state_dict:
        pos_emb = state_dict["pos_embedding"]
        print(f"\n📊 Positional Embedding:")
        print(f"  Shape: {pos_emb.shape}")
        print(f"  Mean: {pos_emb.mean().item():.6f}")
        print(f"  Std: {pos_emb.std().item():.6f}")
    
    # Check output head
    head_layers = [k for k in state_dict.keys() if "head" in k]
    if head_layers:
        print(f"\n📊 Output Head:")
        for layer in head_layers:
            if "weight" in layer:
                weights = state_dict[layer]
                print(f"  {layer}:")
                print(f"    Shape: {weights.shape}")
                print(f"    Mean: {weights.mean().item():.6f}")
                print(f"    Std: {weights.std().item():.6f}")
            elif "bias" in layer:
                bias = state_dict[layer]
                print(f"  {layer}:")
                print(f"    Shape: {bias.shape}")
                print(f"    Mean: {bias.mean().item():.6f}")
                print(f"    Std: {bias.std().item():.6f}")
                
                # Check if bias is stuck at initialization
                if abs(bias.mean().item()) < 1e-6 and bias.std().item() < 1e-6:
                    print(f"    ⚠️  WARNING: Bias appears to be at initialization (all zeros)!")


def test_model_output_diversity(checkpoint):
    """Test if model produces diverse outputs for different inputs."""
    print("\n" + "="*60)
    print("OUTPUT DIVERSITY TEST")
    print("="*60)
    
    if "model_state_dict" not in checkpoint or "config" not in checkpoint:
        print("❌ Cannot test - missing model_state_dict or config")
        return
    
    # Reconstruct model
    from model.transformer_model import ActionValueTransformer, TransformerConfig
    
    if "model" in checkpoint["config"]:
        model_cfg = checkpoint["config"]["model"]
        model_config = TransformerConfig(**model_cfg)
    else:
        print("❌ Cannot reconstruct model config")
        return
    
    model = ActionValueTransformer(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Create 10 random input sequences
    vocab_size = model_config.vocab_size
    seq_len = model_config.max_seq_len
    
    print(f"\nGenerating {10} random inputs (vocab_size={vocab_size}, seq_len={seq_len})")
    
    inputs = []
    for i in range(10):
        # Create random sequence
        seq = torch.randint(0, vocab_size, (seq_len,))
        inputs.append(seq)
    
    batch = torch.stack(inputs)
    
    with torch.no_grad():
        logits = model(batch)
        probs = torch.sigmoid(logits)
    
    print(f"\nLogits: {logits.tolist()}")
    print(f"Probabilities: {probs.tolist()}")
    
    # Check diversity
    logit_std = logits.std().item()
    prob_std = probs.std().item()
    logit_range = (logits.max() - logits.min()).item()
    prob_range = (probs.max() - probs.min()).item()
    
    print(f"\n📊 Output Statistics:")
    print(f"  Logits - Mean: {logits.mean().item():.6f}, Std: {logit_std:.6f}, Range: {logit_range:.6f}")
    print(f"  Probs  - Mean: {probs.mean().item():.6f}, Std: {prob_std:.6f}, Range: {prob_range:.6f}")
    
    if logit_std < 0.1:
        print("\n⚠️  WARNING: Very low output diversity!")
        print("   All inputs produce nearly identical outputs.")
        print("   The model is likely not learning meaningful patterns.")
    elif logit_std < 0.5:
        print("\n⚠️  CAUTION: Low output diversity.")
        print("   Model may be collapsing to mean prediction.")
    else:
        print("\n✅ Output diversity looks reasonable.")


def check_optimizer_state(checkpoint):
    """Check if optimizer state looks correct."""
    print("\n" + "="*60)
    print("OPTIMIZER STATE CHECK")
    print("="*60)
    
    if "optimizer_state_dict" not in checkpoint:
        print("❌ No optimizer state found")
        return
    
    opt_state = checkpoint["optimizer_state_dict"]
    
    if "state" in opt_state and opt_state["state"]:
        # Check first parameter's optimizer state
        first_param_state = opt_state["state"][0] if 0 in opt_state["state"] else None
        
        if first_param_state:
            print("\nFirst parameter optimizer state:")
            for key, value in first_param_state.items():
                if torch.is_tensor(value):
                    print(f"  {key}: shape={value.shape}, mean={value.mean().item():.6f}")
                else:
                    print(f"  {key}: {value}")
        
        # Check if Adam moments are being updated
        if first_param_state and "exp_avg" in first_param_state:
            exp_avg = first_param_state["exp_avg"]
            exp_avg_sq = first_param_state.get("exp_avg_sq")
            
            if exp_avg.abs().mean().item() < 1e-8:
                print("\n⚠️  WARNING: exp_avg is near zero - gradients may not be flowing!")
            else:
                print(f"\n✅ exp_avg magnitude: {exp_avg.abs().mean().item():.6e}")
    
    # Check learning rate
    if "param_groups" in opt_state:
        for i, group in enumerate(opt_state["param_groups"]):
            if "lr" in group:
                print(f"\nParam group {i} learning rate: {group['lr']:.6e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python diagnose_checkpoint.py <checkpoint_path>")
        print("\nExample:")
        print("  python diagnose_checkpoint.py model/checkpoint/checkpoint_step_7000.pt")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    
    if checkpoint is None:
        sys.exit(1)
    
    # Run diagnostics
    diagnose_model_weights(checkpoint)
    test_model_output_diversity(checkpoint)
    check_optimizer_state(checkpoint)
    
    print("\n" + "="*60)
    print("DIAGNOSIS COMPLETE")
    print("="*60)

