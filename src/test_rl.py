"""Test script for the RL training pipeline.

This script validates that all components work together correctly.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        from model.transformer_model import ActionValueTransformer, TransformerConfig
        print("✅ Model imports OK")
    except Exception as e:
        print(f"❌ Model import failed: {e}")
        return False
    
    try:
        from model.dataset import fen_to_tokens, action_to_token
        print("✅ Dataset imports OK")
    except Exception as e:
        print(f"❌ Dataset import failed: {e}")
        return False
    
    try:
        import selfplay
        print("✅ Self-play module OK")
    except Exception as e:
        print(f"❌ Self-play import failed: {e}")
        return False
    
    try:
        import rl_train
        print("✅ RL train module OK")
    except Exception as e:
        print(f"❌ RL train import failed: {e}")
        return False
    
    try:
        import rl_loop
        print("✅ RL loop module OK")
    except Exception as e:
        print(f"❌ RL loop import failed: {e}")
        return False
    
    return True


def test_model_creation():
    """Test creating models with and without policy head."""
    print("\n🧪 Testing model creation...")
    
    try:
        import torch
        from model.transformer_model import ActionValueTransformer, TransformerConfig
        
        # Test value-only model (default)
        config = TransformerConfig(
            vocab_size=32832,
            max_seq_len=77,
            d_model=128,
            num_layers=4,
            num_heads=8,
            dim_feedforward=256,
            dropout=0.1,
        )
        model = ActionValueTransformer(config)
        print(f"✅ Created value-only model: {sum(p.numel() for p in model.parameters()):,} params")
        
        # Test forward pass
        tokens = torch.randint(0, 32832, (2, 77))
        output = model(tokens)
        assert output.shape == (2,), f"Expected shape (2,), got {output.shape}"
        print(f"✅ Value-only forward pass OK: output shape {output.shape}")
        
        # Test policy + value model
        config_dual = TransformerConfig(
            vocab_size=32832,
            max_seq_len=77,
            d_model=128,
            num_layers=4,
            num_heads=8,
            dim_feedforward=256,
            dropout=0.1,
            use_policy_head=True,
            num_actions=1858,
        )
        model_dual = ActionValueTransformer(config_dual)
        print(f"✅ Created dual-head model: {sum(p.numel() for p in model_dual.parameters()):,} params")
        
        # Test dual forward pass
        value_output, policy_output = model_dual(tokens, return_policy=True)
        assert value_output.shape == (2,), f"Expected value shape (2,), got {value_output.shape}"
        assert policy_output.shape == (2, 1858), f"Expected policy shape (2, 1858), got {policy_output.shape}"
        print(f"✅ Dual-head forward pass OK: value {value_output.shape}, policy {policy_output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_encoding():
    """Test encoding/decoding self-play data."""
    print("\n🧪 Testing data encoding...")
    
    try:
        from selfplay import encode_action_value, _write_varint
        from model.coders import decode_action_value
        
        # Test data
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        move = "e2e4"
        win_prob = 0.55
        
        # Encode
        encoded = encode_action_value(fen, move, win_prob)
        print(f"✅ Encoded data: {len(encoded)} bytes")
        
        # Decode
        decoded_fen, decoded_move, decoded_prob = decode_action_value(encoded)
        
        # Verify
        assert decoded_fen == fen, f"FEN mismatch: {decoded_fen} != {fen}"
        assert decoded_move == move, f"Move mismatch: {decoded_move} != {move}"
        assert abs(decoded_prob - win_prob) < 1e-6, f"Prob mismatch: {decoded_prob} != {win_prob}"
        
        print(f"✅ Decoded correctly: fen={decoded_fen[:20]}..., move={decoded_move}, prob={decoded_prob:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data encoding failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tokenization():
    """Test chess position tokenization."""
    print("\n🧪 Testing tokenization...")
    
    try:
        from model.dataset import fen_to_tokens, action_to_token
        import numpy as np
        
        # Test FEN tokenization
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        tokens = fen_to_tokens(fen)
        
        print(f"✅ FEN tokenization: {tokens.shape} tokens, dtype={tokens.dtype}")
        print(f"   Sample tokens: {tokens[:10]}")
        
        # Test move tokenization
        move = "e2e4"
        move_token = action_to_token(move)
        assert move_token is not None, "Failed to tokenize move"
        print(f"✅ Move tokenization: {move} -> {move_token}")
        
        # Test invalid move
        invalid_token = action_to_token("invalid")
        assert invalid_token is None, "Invalid move should return None"
        print(f"✅ Invalid move handling OK")
        
        return True
        
    except Exception as e:
        print(f"❌ Tokenization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_selfplay_engine():
    """Test self-play engine with a small model."""
    print("\n🧪 Testing self-play engine...")
    
    try:
        import torch
        import chess
        from model.transformer_model import ActionValueTransformer, TransformerConfig
        from selfplay import SelfPlayEngine, SelfPlayConfig, play_game
        
        # Create small model for testing
        config = TransformerConfig(
            vocab_size=32832,
            max_seq_len=77,
            d_model=64,  # Smaller for testing
            num_layers=2,
            num_heads=4,
            dim_feedforward=128,
            dropout=0.0,  # No dropout for testing
        )
        model = ActionValueTransformer(config)
        model.eval()
        
        print(f"✅ Created test model: {sum(p.numel() for p in model.parameters()):,} params")
        
        # Create self-play engine
        engine = SelfPlayEngine(model, device="cpu")
        print(f"✅ Created self-play engine")
        
        # Test move selection
        board = chess.Board()
        move, probs = engine.choose_move(board, temperature=0.5)
        
        assert move is not None, "Failed to choose move"
        assert move in board.legal_moves, "Chose illegal move"
        assert len(probs) > 0, "No move probabilities returned"
        
        print(f"✅ Chose move: {move.uci()} (from {len(probs)} legal moves)")
        
        # Test playing a game (just a few moves)
        sp_config = SelfPlayConfig(
            temperature=0.3,
            max_moves=10,  # Just 10 moves for testing
            num_games=1,
        )
        
        moves, result = play_game(engine, sp_config)
        print(f"✅ Played test game: {len(moves)} moves, result={result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Self-play engine failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests."""
    print("="*70)
    print("🚀 Running RL Pipeline Tests")
    print("="*70)
    
    tests = [
        ("Imports", test_imports),
        ("Model Creation", test_model_creation),
        ("Data Encoding", test_data_encoding),
        ("Tokenization", test_tokenization),
        ("Self-Play Engine", test_selfplay_engine),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ Unexpected error in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("📊 Test Summary")
    print("="*70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The RL pipeline is ready to use.")
        print("\nNext steps:")
        print("  1. Generate self-play data:")
        print("     modal run src.selfplay::main --num-games 100 --checkpoint-step 22000")
        print()
        print("  2. Train on self-play data:")
        print("     modal run src.rl_train::main --checkpoint-step 22000 --max-steps 5000")
        print()
        print("  3. Or run the full loop:")
        print("     modal run src.rl_loop::main --starting-checkpoint 22000 --num-iterations 3")
        print()
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix the issues before running RL training.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

