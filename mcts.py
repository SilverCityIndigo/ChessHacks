"""
Monte Carlo Tree Search (MCTS) for Chess.
Combines neural network predictions with lookahead search for stronger moves.
"""
import chess
import torch
import numpy as np
from typing import Dict, Optional, Tuple
from collections import defaultdict
import math


class MCTSNode:
    """Node in MCTS tree."""
    
    def __init__(self, board: chess.Board, parent: Optional['MCTSNode'] = None):
        self.board = board.copy()
        self.parent = parent
        self.children: Dict[chess.Move, 'MCTSNode'] = {}
        
        # Statistics
        self.visits = 0
        self.value_sum = 0.0
        self.children_visited = 0
        # Prior from policy network (set during expansion)
        self.policy_prob: float = 0.0
    
    def uct_value(self, c_puct: float = 1.5) -> float:
        """PUCT score combining value (Q) and prior (P)."""
        q = (self.value_sum / self.visits) if self.visits > 0 else 0.0
        p = self.policy_prob
        n_parent = self.parent.visits if self.parent else 1
        u = c_puct * p * math.sqrt(n_parent) / (1 + self.visits)
        return q + u
    
    def best_child(self, c_param: float = 1.5) -> Optional['MCTSNode']:
        """Select child with highest PUCT value."""
        if not self.children:
            return None
        
        return max(self.children.values(), key=lambda node: node.uct_value(c_param))
    
    def expand(self, policy_probs: Dict[chess.Move, float]):
        """Expand node by adding all children based on policy."""
        legal_moves = list(self.board.legal_moves)
        
        for move in legal_moves:
            child_board = self.board.copy()
            child_board.push(move)
            self.children[move] = MCTSNode(child_board, parent=self)
            
            # Store policy probability
            self.children[move].policy_prob = policy_probs.get(move, 1.0 / max(1, len(legal_moves)))
    
    def backup(self, value: float):
        """Backup value up the tree."""
        node = self
        while node is not None:
            node.visits += 1
            node.value_sum += value
            node = node.parent


class MCTS:
    """
    Monte Carlo Tree Search engine combining neural network with tree search.
    Uses network for move prioritization and position evaluation.
    """
    
    def __init__(self, model, device: str = 'cpu', num_simulations: int = 100):
        """
        Initialize MCTS.
        
        Args:
            model: Trained neural network model
            device: Device to run model on ('cpu' or 'cuda')
            num_simulations: Number of simulations per move
        """
        self.model = model
        self.device = device
        self.num_simulations = num_simulations
    
    def get_move(self, board: chess.Board) -> chess.Move:
        """
        Select best move using MCTS.
        
        Args:
            board: Current board position
            
        Returns:
            Best move according to MCTS
        """
        root = MCTSNode(board)

        # Run simulations
        for _ in range(self.num_simulations):
            node = root

            # Selection: descend until a leaf (no children)
            while node.children:
                node = node.best_child()

            # Expansion: expand once using policy if not terminal
            if not node.board.is_game_over():
                policy_probs = self._get_policy(node.board)
                node.expand(policy_probs)

                # After expansion, choose one child to evaluate
                if node.children:
                    # Pick child with highest prior (policy_prob) to evaluate first
                    node = max(
                        node.children.items(),
                        key=lambda kv: getattr(kv[1], 'policy_prob', 0.0)
                    )[1]

            # Evaluation
            value = self._evaluate(node.board)

            # Backup
            node.backup(value)
        
        # Select best move (most visited)
        if not root.children:
            legal_moves = list(board.legal_moves)
            return legal_moves[0] if legal_moves else None
        
        best_move = max(
            root.children.items(),
            key=lambda x: x[1].visits
        )[0]
        
        return best_move
    
    def get_move_probabilities(self, board: chess.Board) -> Dict[str, float]:
        """
        Get move probability distribution from MCTS.
        
        Args:
            board: Current board position
            
        Returns:
            Dict mapping move UCI to probability
        """
        root = MCTSNode(board)

        for _ in range(self.num_simulations):
            node = root

            while node.children:
                node = node.best_child()

            if not node.board.is_game_over():
                policy_probs = self._get_policy(node.board)
                node.expand(policy_probs)
                if node.children:
                    node = max(
                        node.children.items(),
                        key=lambda kv: getattr(kv[1], 'policy_prob', 0.0)
                    )[1]

            value = self._evaluate(node.board)
            node.backup(value)

        # Compute probabilities from visit counts
        total_visits = sum(child.visits for child in root.children.values())

        move_probs = {}
        for move, child in root.children.items():
            move_probs[move.uci()] = child.visits / total_visits if total_visits > 0 else 0

        return move_probs
    
    def _get_policy(self, board: chess.Board) -> Dict[chess.Move, float]:
        """Get move policy from neural network."""
        from src.training.trainer import board_to_encoding
        
        self.model.eval()
        with torch.no_grad():
            encoding = board_to_encoding(board)
            tensor = torch.FloatTensor(encoding).unsqueeze(0).to(self.device)
            
            features = self.model.feature_layers(tensor)
            logits = self.model.policy_head(features)
            probs = torch.softmax(logits, dim=1)[0]
            
            # Map to legal moves
            legal_moves = list(board.legal_moves)
            policy_probs = {}
            
            for move in legal_moves:
                move_idx = self._uci_to_index(move.uci())
                prob = probs[move_idx].item()
                policy_probs[move] = prob
            
            # Normalize
            total_prob = sum(policy_probs.values())
            if total_prob > 0:
                policy_probs = {m: p / total_prob for m, p in policy_probs.items()}
            
            return policy_probs
    
    def _evaluate(self, board: chess.Board) -> float:
        """Evaluate position using neural network value head."""
        from src.training.trainer import board_to_encoding
        
        self.model.eval()
        with torch.no_grad():
            encoding = board_to_encoding(board)
            tensor = torch.FloatTensor(encoding).unsqueeze(0).to(self.device)
            
            features = self.model.feature_layers(tensor)
            value = self.model.value_head(features)
            evaluation = value[0, 0].item()
            
            # Flip perspective if black to move
            if not board.turn:
                evaluation = 1.0 - evaluation
            
            return evaluation
    
    def _uci_to_index(self, uci: str) -> int:
        """Convert UCI move to policy head index."""
        from_square = (ord(uci[0]) - ord('a')) + (int(uci[1]) - 1) * 8
        to_square = (ord(uci[2]) - ord('a')) + (int(uci[3]) - 1) * 8
        
        move_idx = from_square * 64 + to_square
        return min(move_idx, 4671)
