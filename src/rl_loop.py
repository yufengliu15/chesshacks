"""Reinforcement learning loop for chess.

Orchestrates the full RL training pipeline:
1. Generate games where model plays against Stockfish
2. Train model on game outcomes  
3. Repeat

This is a local orchestration script that calls the Modal functions.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_modal_command(command: list[str]) -> dict:
    """Run a modal command and return the result."""
    print(f"🚀 Running: modal {' '.join(command)}")
    print()
    
    result = subprocess.run(
        ["modal"] + command,
        capture_output=False,
        text=True,
    )
    
    if result.returncode != 0:
        print(f"❌ Command failed with exit code {result.returncode}")
        sys.exit(1)
    
    return {"status": "success"}


def run_rl_iteration(
    checkpoint_step: int,
    num_selfplay_games: int = 100,
    num_training_steps: int = 5000,
    temperature: float = 0.3,
    learning_rate: float = 5e-5,
    num_selfplay_shards: int = 4,
    stockfish_depth: int = 4,
    stockfish_time_limit: float = 0.1,
    use_mcts: bool = True,  # ✅ Enable MCTS by default
    mcts_simulations: int = 100,
    mcts_c_puct: float = 1.5,
    mcts_max_depth: int = 5,
) -> dict:
    """Run one iteration of RL training against Stockfish.
    
    Args:
        checkpoint_step: Starting checkpoint step
        num_selfplay_games: Number of games to generate (model vs Stockfish)
        num_training_steps: Number of training steps to run
        temperature: Temperature for model move selection
        learning_rate: Learning rate for training
        num_selfplay_shards: Number of parallel game generation workers
        stockfish_depth: Stockfish search depth (higher = stronger opponent)
        stockfish_time_limit: Time limit per Stockfish move in seconds
        use_mcts: Whether to use MCTS for model's moves
        mcts_simulations: Number of MCTS simulations (if enabled)
        mcts_c_puct: Exploration constant for MCTS
        mcts_max_depth: Max depth per simulation (0 = unlimited)
    
    Returns:
        Statistics about the iteration
    """
    print(f"\n{'='*70}")
    print(f"🔄 Starting RL Iteration from checkpoint step {checkpoint_step}")
    print(f"{'='*70}\n")
    
    # Step 1: Generate games vs Stockfish
    print("📊 Step 1: Generating games (Model vs Stockfish)...")
    print(f"   Games: {num_selfplay_games}")
    print(f"   Temperature: {temperature}")
    print(f"   Stockfish depth: {stockfish_depth}")
    print(f"   Parallel shards: {num_selfplay_shards}")
    if use_mcts:
        print(f"   🌳 MCTS: Enabled ({mcts_simulations} sims, c_puct={mcts_c_puct}, max_depth={mcts_max_depth})")
    else:
        print(f"   ⚡ MCTS: Disabled")
    print()
    
    cmd = [
        "run",
        "src.selfplay::main",
        "--num-games", str(num_selfplay_games),
        "--checkpoint-step", str(checkpoint_step),
        "--temperature", str(temperature),
        "--num-shards", str(num_selfplay_shards),
        "--stockfish-depth", str(stockfish_depth),
        "--stockfish-time-limit", str(stockfish_time_limit),
    ]
    
    if use_mcts:
        cmd.extend([
            "--use-mcts",
            "--mcts-simulations", str(mcts_simulations),
            "--mcts-c-puct", str(mcts_c_puct),
            "--mcts-max-depth", str(mcts_max_depth),
        ])
    
    run_modal_command(cmd)
    
    print(f"\n✅ Self-play generation complete")
    print()
    
    # Step 2: Train on self-play data
    print("📊 Step 2: Training on self-play data...")
    print(f"   Training steps: {num_training_steps}")
    print(f"   Learning rate: {learning_rate}")
    print()
    
    run_modal_command([
        "run",
        "src.rl_train::main",
        "--checkpoint-step", str(checkpoint_step),
        "--max-steps", str(num_training_steps),
        "--learning-rate", str(learning_rate),
    ])
    
    # Calculate new checkpoint step
    final_step = checkpoint_step + num_training_steps
    
    print(f"\n✅ Training complete")
    print(f"   New checkpoint: checkpoint_step_{final_step}.pt")
    print()
    
    return {
        "starting_step": checkpoint_step,
        "final_step": final_step,
    }


def main(
    starting_checkpoint: int = 22000,
    num_iterations: int = 3,
    games_per_iteration: int = 200,
    steps_per_iteration: int = 5000,
    temperature: float = 0.3,
    learning_rate: float = 5e-5,
    num_selfplay_shards: int = 4,
    stockfish_depth: int = 4,
    stockfish_time_limit: float = 0.1,
    use_mcts: bool = False,
    mcts_simulations: int = 100,
    mcts_c_puct: float = 1.5,
    mcts_max_depth: int = 5,
):
    """Run multiple iterations of RL training against Stockfish.
    
    Args:
        starting_checkpoint: Initial checkpoint to start from
        num_iterations: Number of RL iterations to run
        games_per_iteration: Games to generate per iteration (vs Stockfish)
        steps_per_iteration: Training steps per iteration
        temperature: Temperature for model move selection
        learning_rate: Learning rate for training
        num_selfplay_shards: Parallel game generation workers per iteration
        stockfish_depth: Stockfish search depth (higher = stronger)
        stockfish_time_limit: Time limit per Stockfish move in seconds
        use_mcts: Whether to use MCTS for model's moves
        mcts_simulations: Number of MCTS simulations (if enabled)
        mcts_c_puct: Exploration constant for MCTS
        mcts_max_depth: Max depth per simulation (0 = unlimited)
    """
    print("🚀 RL Training Loop: Model vs Stockfish")
    print("=" * 70)
    print(f"Starting checkpoint: {starting_checkpoint}")
    print(f"Number of iterations: {num_iterations}")
    print(f"Games per iteration: {games_per_iteration}")
    print(f"Training steps per iteration: {steps_per_iteration}")
    print(f"Temperature: {temperature}")
    print(f"Learning rate: {learning_rate}")
    print(f"Stockfish depth: {stockfish_depth}")
    print(f"Stockfish time limit: {stockfish_time_limit}s")
    if use_mcts:
        print(f"🌳 MCTS: Enabled ({mcts_simulations} sims, c_puct={mcts_c_puct}, max_depth={mcts_max_depth})")
    else:
        print(f"⚡ MCTS: Disabled")
    print()
    
    current_checkpoint = starting_checkpoint
    iteration_results = []
    
    for iteration in range(1, num_iterations + 1):
        print(f"\n{'#'*70}")
        print(f"# ITERATION {iteration}/{num_iterations}")
        print(f"{'#'*70}")
        
        result = run_rl_iteration(
            checkpoint_step=current_checkpoint,
            num_selfplay_games=games_per_iteration,
            num_training_steps=steps_per_iteration,
            temperature=temperature,
            learning_rate=learning_rate,
            num_selfplay_shards=num_selfplay_shards,
            stockfish_depth=stockfish_depth,
            stockfish_time_limit=stockfish_time_limit,
            use_mcts=use_mcts,
            mcts_simulations=mcts_simulations,
            mcts_c_puct=mcts_c_puct,
            mcts_max_depth=mcts_max_depth,
        )
        
        iteration_results.append(result)
        
        # Update checkpoint for next iteration
        current_checkpoint = result["final_step"]
        
        print(f"\n{'='*70}")
        print(f"✅ Iteration {iteration} complete!")
        print(f"   Checkpoint progression: {result['starting_step']} → {result['final_step']}")
        print(f"{'='*70}")
    
    # Final summary
    print(f"\n\n{'='*70}")
    print("🎉 RL TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"\nIteration Summary:")
    print(f"{'Iter':<6} {'Start':>8} {'End':>8}")
    print("-" * 70)
    
    for idx, result in enumerate(iteration_results, 1):
        print(f"{idx:<6} "
              f"{result['starting_step']:>8} "
              f"{result['final_step']:>8}")
    
    print("-" * 70)
    print(f"{'TOTAL':<6} "
          f"{starting_checkpoint:>8} "
          f"{current_checkpoint:>8}")
    
    print()
    print(f"Final model checkpoint: checkpoint_step_{current_checkpoint}.pt")
    print()
    
    # List data files
    print("📁 Listing self-play data files...")
    run_modal_command(["run", "src.selfplay::main", "list"])


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run RL training loop (Model vs Stockfish)")
    parser.add_argument("--starting-checkpoint", type=int, default=22000,
                       help="Initial checkpoint step")
    parser.add_argument("--num-iterations", type=int, default=3,
                       help="Number of RL iterations")
    parser.add_argument("--games-per-iteration", type=int, default=200,
                       help="Games to generate per iteration")
    parser.add_argument("--steps-per-iteration", type=int, default=5000,
                       help="Training steps per iteration")
    parser.add_argument("--temperature", type=float, default=0.3,
                       help="Temperature for model move selection")
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                       help="Learning rate for training")
    parser.add_argument("--num-selfplay-shards", type=int, default=4,
                       help="Parallel game generation workers")
    parser.add_argument("--stockfish-depth", type=int, default=4,
                       help="Stockfish search depth (higher = stronger)")
    parser.add_argument("--stockfish-time-limit", type=float, default=0.1,
                       help="Time limit per Stockfish move in seconds")
    parser.add_argument("--use-mcts", action="store_true", default=True,
                       help="Enable MCTS for model's moves (default: enabled)")
    parser.add_argument("--no-mcts", action="store_false", dest="use_mcts",
                       help="Disable MCTS and use greedy selection")
    parser.add_argument("--mcts-simulations", type=int, default=100,
                       help="Number of MCTS simulations (if enabled)")
    parser.add_argument("--mcts-c-puct", type=float, default=1.5,
                       help="Exploration constant for MCTS")
    parser.add_argument("--mcts-max-depth", type=int, default=5,
                       help="Max depth per simulation (0 = unlimited)")
    
    args = parser.parse_args()
    
    main(
        starting_checkpoint=args.starting_checkpoint,
        num_iterations=args.num_iterations,
        games_per_iteration=args.games_per_iteration,
        steps_per_iteration=args.steps_per_iteration,
        temperature=args.temperature,
        learning_rate=args.learning_rate,
        num_selfplay_shards=args.num_selfplay_shards,
        stockfish_depth=args.stockfish_depth,
        stockfish_time_limit=args.stockfish_time_limit,
        use_mcts=args.use_mcts,
        mcts_simulations=args.mcts_simulations,
        mcts_c_puct=args.mcts_c_puct,
        mcts_max_depth=args.mcts_max_depth,
    )
