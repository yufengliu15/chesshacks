"""Helper script to reload the chess engine with a new checkpoint.

Usage:
    # Reload with latest checkpoint
    python -c "from src.main import reload_engine; reload_engine()"
    
    # Or use this script
    python src/reload_model.py
    
    # Check which checkpoint is currently loaded
    python src/reload_model.py --check
"""

import sys
import os
from pathlib import Path

def check_loaded_model():
    """Check which model is currently loaded."""
    print("🔍 Checking currently loaded model...")
    
    try:
        from main import _ENGINE
        
        if _ENGINE.model is None:
            print("❌ No model is loaded!")
            return
        
        # Try to find the checkpoint directory
        from model.engine import CHECKPOINT_DIR
        
        if CHECKPOINT_DIR.exists():
            checkpoints = list(CHECKPOINT_DIR.glob("checkpoint_step_*.pt"))
            checkpoints.extend(CHECKPOINT_DIR.glob("final_checkpoint_step_*.pt"))
            
            if checkpoints:
                latest = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]) if p.stem.split('_')[-1].isdigit() else 0)
                print(f"📁 Latest checkpoint in directory: {latest.name}")
        
        print(f"✅ Model is loaded and ready")
        print(f"   Device: {_ENGINE.device}")
        
        # Print model weight statistics
        if _ENGINE.model:
            emb_weights = _ENGINE.model.embedding.weight.data
            print(f"   Current Embedding: μ={emb_weights.mean().item():.6f}, σ={emb_weights.std().item():.6f}")
        
    except Exception as e:
        print(f"❌ Error checking model: {e}")
        import traceback
        traceback.print_exc()


def reload_model():
    """Reload the model with the latest checkpoint."""
    print("🔄 Reloading model...")
    
    try:
        from main import reload_engine
        engine = reload_engine()
        
        if engine.model is None:
            print("❌ Failed to load model!")
            return False
        
        print("✅ Model reloaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error reloading model: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        check_loaded_model()
    else:
        # Default: reload
        success = reload_model()
        sys.exit(0 if success else 1)

