"""Monte Carlo Tree Search for chess using transformer model.

This implementation follows the AlphaZero approach with adaptations for a value-based model:
- UCB-based tree traversal (exploration + exploitation)
- Neural network provides:
  1. Prior probabilities (from value estimates or policy head)
  2. Value estimates for leaf evaluation
- Batch inference for efficiency
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import chess
import numpy as np
import torch


@dataclass
class MCTSConfig:
    """Configuration for MCTS."""
    num_simulations: int = 800  # Number of MCTS simulations per move
    c_puct: float = 1.5  # Exploration constant for UCB formula
    temperature: float = 1.0  # Temperature for move selection (1.0 = stochastic, 0.0 = greedy)
    dirichlet_alpha: float = 0.3  # Dirichlet noise for exploration at root
    dirichlet_epsilon: float = 0.25  # Weight of Dirichlet noise at root
    use_policy_head: bool = False  # Use policy head for priors if available
    virtual_loss: float = 1.0  # Virtual loss for parallel MCTS
    max_depth: Optional[int] = None  # Maximum depth per simulation (None = unlimited)
    
    # Value transformation settings
    value_temp: float = 2.0  # Temperature for converting values to priors
    min_prior: float = 1e-8  # Minimum prior probability


class MCTSNode:
    """A node in the MCTS tree."""
    
    def __init__(self, board: chess.Board, parent: Optional['MCTSNode'] = None, 
                 prior: float = 1.0, move: Optional[chess.Move] = None):
        self.board = board
        self.parent = parent
        self.move = move  # Move that led to this node
        self.prior = prior  # Prior probability P(s,a)
        
        # Statistics
        self.visit_count = 0
        self.value_sum = 0.0  # Sum of values from perspective of parent
        self.children: Dict[chess.Move, MCTSNode] = {}
        
        # Network evaluation
        self.value: Optional[float] = None  # Network value estimate
        self.is_expanded = False
    
    def mean_value(self) -> float:
        """Average value from perspective of parent."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, parent_visit_count: int, c_puct: float) -> float:
        """Upper confidence bound score for this node.
        
        UCB = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        
        where:
        - Q(s,a) = mean value (exploitation)
        - P(s,a) = prior probability
        - N(s) = parent visit count
        - N(s,a) = this node's visit count
        """
        exploration = c_puct * self.prior * math.sqrt(parent_visit_count) / (1 + self.visit_count)
        return self.mean_value() + exploration
    
    def select_child(self, c_puct: float) -> 'MCTSNode':
        """Select child with highest UCB score."""
        return max(self.children.values(), key=lambda node: node.ucb_score(self.visit_count, c_puct))
    
    def expand(self, priors: Dict[chess.Move, float]) -> None:
        """Expand this node by creating children for all legal moves."""
        legal_moves = list(self.board.legal_moves)
        
        # Create child nodes
        for move in legal_moves:
            if move not in self.children:
                # Make the move on a copy of the board
                new_board = self.board.copy()
                new_board.push(move)
                
                # Get prior probability for this move
                prior = priors.get(move, 1.0 / len(legal_moves))
                
                # Create child node
                self.children[move] = MCTSNode(new_board, parent=self, prior=prior, move=move)
        
        self.is_expanded = True
    
    def update(self, value: float) -> None:
        """Update node statistics after a simulation.
        
        Args:
            value: Value from perspective of the player to move at this node
        """
        self.visit_count += 1
        # Flip value because value_sum is from parent's perspective
        # If parent played a move leading here, they get the negation of this value
        if self.parent is not None:
            self.value_sum += -value
        else:
            # Root node
            self.value_sum += value
    
    def add_dirichlet_noise(self, alpha: float, epsilon: float) -> None:
        """Add Dirichlet noise to priors at root for exploration."""
        if not self.children:
            return
        
        moves = list(self.children.keys())
        noise = np.random.dirichlet([alpha] * len(moves))
        
        for move, noise_value in zip(moves, noise):
            child = self.children[move]
            child.prior = (1 - epsilon) * child.prior + epsilon * noise_value


class MCTS:
    """Monte Carlo Tree Search for chess."""
    
    def __init__(self, model, device: str, config: MCTSConfig):
        """Initialize MCTS.
        
        Args:
            model: The transformer model (should have choose_move_mcts or similar interface)
            device: PyTorch device
            config: MCTS configuration
        """
        self.model = model
        self.device = torch.device(device)
        self.config = config
        self.model.eval()
    
    def _fen_to_tokens(self, fen: str) -> torch.Tensor:
        """Convert FEN to token tensor."""
        from .dataset import fen_to_tokens
        tokens = fen_to_tokens(fen)
        return torch.from_numpy(tokens.astype("int64"))
    
    def _action_to_token(self, move: str) -> Optional[int]:
        """Convert move to action token."""
        from .dataset import action_to_token
        return action_to_token(move)
    
    def _evaluate_position(self, board: chess.Board) -> Tuple[float, Dict[chess.Move, float]]:
        """Evaluate a position using the neural network.
        
        Returns:
            value: Value estimate for the position (from perspective of side to move)
            priors: Prior probabilities for each legal move
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            # Terminal position
            if board.is_checkmate():
                return -1.0, {}  # Loss for side to move
            else:
                return 0.0, {}  # Draw
        
        # Get FEN tokens
        fen_tokens = self._fen_to_tokens(board.fen())
        
        # Try policy head first if available and configured
        if self.config.use_policy_head and hasattr(self.model, 'policy_head') and self.model.policy_head is not None:
            # Use policy head for priors
            with torch.no_grad():
                batch = fen_tokens.unsqueeze(0).to(self.device)
                value_logit, policy_logits = self.model(batch, return_policy=True)
                
                # Get value
                value = torch.sigmoid(value_logit).item()
                
                # Get priors from policy head
                policy = torch.softmax(policy_logits[0], dim=0).cpu().numpy()
                
                # Map policy logits to legal moves
                priors = {}
                total_prior = 0.0
                for move in legal_moves:
                    move_token = self._action_to_token(move.uci())
                    if move_token is not None and move_token < len(policy):
                        priors[move] = max(policy[move_token], self.config.min_prior)
                        total_prior += priors[move]
                    else:
                        priors[move] = self.config.min_prior
                        total_prior += self.config.min_prior
                
                # Normalize
                if total_prior > 0:
                    priors = {move: p / total_prior for move, p in priors.items()}
        else:
            # Use value network to derive priors (current approach)
            # Score all legal moves
            sequences = []
            kept_moves = []
            for move in legal_moves:
                move_token = self._action_to_token(move.uci())
                if move_token is None:
                    continue
                
                seq = torch.cat([
                    fen_tokens,
                    torch.tensor([move_token], dtype=torch.int64)
                ])
                sequences.append(seq)
                kept_moves.append(move)
            
            if not sequences:
                # Fallback to uniform priors
                uniform_prior = 1.0 / len(legal_moves)
                return 0.5, {move: uniform_prior for move in legal_moves}
            
            # Batch inference
            batch = torch.stack(sequences).to(self.device)
            with torch.no_grad():
                logits = self.model(batch)
                win_probs = torch.sigmoid(logits).cpu().numpy()
            
            # Use current position value as the mean of all moves' values
            value = float(np.mean(win_probs))
            
            # Convert win probabilities to priors using softmax with temperature
            # Higher win prob -> higher prior
            scaled_values = win_probs / self.config.value_temp
            exp_values = np.exp(scaled_values - np.max(scaled_values))
            priors_array = exp_values / exp_values.sum()
            
            # Create priors dict
            priors = {move: max(float(prior), self.config.min_prior) 
                     for move, prior in zip(kept_moves, priors_array)}
            
            # Add tiny priors for moves we couldn't encode
            for move in legal_moves:
                if move not in priors:
                    priors[move] = self.config.min_prior
            
            # Normalize
            total = sum(priors.values())
            priors = {move: p / total for move, p in priors.items()}
        
        return value, priors
    
    def _simulate(self, root: MCTSNode) -> None:
        """Run one simulation from root to leaf, evaluate, and backpropagate."""
        node = root
        search_path = [node]
        
        # 1. SELECT: Traverse tree using UCB until we reach a leaf
        # Stop if: (1) leaf node, (2) game over, or (3) max depth reached
        while node.is_expanded and not node.board.is_game_over():
            # Check max depth limit
            if self.config.max_depth is not None and len(search_path) >= self.config.max_depth:
                break
            
            node = node.select_child(self.config.c_puct)
            search_path.append(node)
        
        # 2. EXPAND & EVALUATE: Expand the leaf and get value estimate
        if node.board.is_game_over():
            # Terminal node
            if node.board.is_checkmate():
                # Loss for side to move
                value = -1.0
            else:
                # Draw
                value = 0.0
        else:
            # Evaluate with neural network
            value, priors = self._evaluate_position(node.board)
            node.value = value
            
            # Expand node
            if not node.is_expanded:
                node.expand(priors)
        
        # 3. BACKUP: Propagate value back up the tree
        for node in reversed(search_path):
            node.update(value)
            # Flip value for next level up (zero-sum game)
            value = -value
    
    def search(self, board: chess.Board, add_noise: bool = True) -> Tuple[chess.Move, Dict[str, float]]:
        """Run MCTS from the given position.
        
        Args:
            board: Current chess position
            add_noise: Whether to add Dirichlet noise at root (for exploration)
        
        Returns:
            best_move: The selected move
            visit_distribution: Dictionary mapping move UCI to visit counts
        """
        # Create root node
        root = MCTSNode(board.copy())
        
        # Initial evaluation and expansion of root
        value, priors = self._evaluate_position(board)
        root.expand(priors)
        
        # Add Dirichlet noise for exploration (typically only during training/self-play)
        if add_noise:
            root.add_dirichlet_noise(self.config.dirichlet_alpha, self.config.dirichlet_epsilon)
        
        # Run simulations
        for _ in range(self.config.num_simulations):
            self._simulate(root)
        
        # Select move based on visit counts
        if not root.children:
            # No legal moves (shouldn't happen)
            return None, {}
        
        # Get visit counts
        visit_counts = {move: child.visit_count for move, child in root.children.items()}
        
        # Select move based on temperature
        if self.config.temperature < 0.01:
            # Greedy: pick most visited
            best_move = max(visit_counts.items(), key=lambda x: x[1])[0]
        else:
            # Stochastic: sample proportional to visit count^(1/temperature)
            moves = list(visit_counts.keys())
            counts = np.array([visit_counts[m] for m in moves])
            
            # Apply temperature
            probs = counts ** (1.0 / self.config.temperature)
            probs = probs / probs.sum()
            
            # Sample move
            best_move = np.random.choice(moves, p=probs)
        
        # Create visit distribution for logging
        total_visits = sum(visit_counts.values())
        visit_dist = {move.uci(): count / total_visits for move, count in visit_counts.items()}
        
        # Also include mean values for analysis
        value_dist = {move.uci(): child.mean_value() for move, child in root.children.items()}
        
        # Combine into single dict with both visits and values
        result = {}
        for move_uci in visit_dist:
            result[f"{move_uci}_visits"] = visit_dist[move_uci]
        for move_uci in value_dist:
            result[f"{move_uci}_value"] = value_dist[move_uci]
        
        return best_move, result


def choose_move_mcts(
    model,
    board: chess.Board,
    device: str = "cuda",
    config: Optional[MCTSConfig] = None,
    add_noise: bool = False,
) -> Tuple[chess.Move, Dict[str, float]]:
    """Choose a move using MCTS.
    
    Args:
        model: The transformer model
        board: Current chess position
        device: PyTorch device
        config: MCTS configuration (uses defaults if None)
        add_noise: Whether to add exploration noise at root
    
    Returns:
        move: Selected move
        info: Dictionary with visit counts and values
    """
    if config is None:
        config = MCTSConfig()
    
    mcts = MCTS(model, device, config)
    return mcts.search(board, add_noise=add_noise)

