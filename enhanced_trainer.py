"""
Enhanced trainer using Stockfish position evaluations.
Trains both policy and value heads with guidance from engine evaluations.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import namedtuple
from src.training.trainer import ChessNeuralNetwork, board_to_encoding
import chess


EnhancedTrainingExample = namedtuple(
    'EnhancedTrainingExample',
    ['fen', 'best_move_uci', 'move_probabilities', 'source', 'position_evaluation']
)


class EnhancedChessTrainer:
    """
    Trainer using neural network guided by Stockfish evaluations.
    Trains both policy (move selection) and value (position evaluation) heads.
    """
    
    def __init__(
        self,
        model: ChessNeuralNetwork,
        learning_rate: float = 0.001,
        device: str = 'cpu',
        policy_weight: float = 0.6,
        value_weight: float = 0.4,
        gradient_clip: float = 1.0
    ):
        """
        Initialize enhanced trainer.
        
        Args:
            model: ChessNeuralNetwork to train
            learning_rate: Learning rate for optimizer
            device: Device to train on ('cpu' or 'cuda')
            policy_weight: Weight for policy loss (move prediction)
            value_weight: Weight for value loss (position evaluation)
            gradient_clip: Max gradient norm for clipping
        """
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.gradient_clip = gradient_clip
        
        # Loss functions
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training stats
        self.stats = {
            'policy_losses': [],
            'value_losses': [],
            'total_losses': [],
            'evaluated_positions': 0,
            'evaluated_pct': 0.0
        }
    
    def train_epoch(self, examples: List[EnhancedTrainingExample], batch_size: int = 32) -> Dict:
        """
        Train for one epoch on enhanced examples.
        
        Args:
            examples: List of training examples with evaluations
            batch_size: Batch size for training
            
        Returns:
            Dictionary with epoch statistics
        """
        self.model.train()
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_loss = 0.0
        num_batches = 0
        evaluated_count = 0
        
        # Shuffle examples
        np.random.shuffle(examples)
        
        # Process in batches
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i+batch_size]
            
            # Prepare batch tensors
            boards_tensor = []
            best_moves_tensor = []
            position_evals_tensor = []
            
            for example in batch:
                # Board encoding
                encoding = board_to_encoding_from_fen(example.fen)
                boards_tensor.append(encoding)
                
                # Best move as target
                move_idx = self._uci_to_index(example.best_move_uci)
                best_moves_tensor.append(move_idx)
                
                # Position evaluation (if available)
                if example.position_evaluation is not None:
                    position_evals_tensor.append(example.position_evaluation)
                    evaluated_count += 1
            
            # Convert to tensors
            boards_tensor = torch.FloatTensor(np.array(boards_tensor)).to(self.device)
            best_moves_tensor = torch.LongTensor(best_moves_tensor).to(self.device)
            
            # Forward pass
            policy_logits, value_output = self.model(boards_tensor)
            
            # Policy loss (supervised move prediction)
            policy_loss = self.policy_loss_fn(policy_logits, best_moves_tensor)
            
            # Value loss (position evaluation)
            value_loss = 0.0
            if position_evals_tensor:
                position_evals_tensor = torch.FloatTensor(position_evals_tensor).to(self.device).unsqueeze(1)
                value_predictions = value_output[:len(position_evals_tensor)]
                value_loss = self.value_loss_fn(value_predictions, position_evals_tensor)
            
            # Weighted total loss
            total_batch_loss = (
                self.policy_weight * policy_loss +
                self.value_weight * value_loss
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()
            
            # Accumulate stats
            total_policy_loss += policy_loss.item() * len(batch)
            total_value_loss += value_loss.item() * len(batch) if value_loss > 0 else 0
            total_loss += total_batch_loss.item() * len(batch)
            num_batches += 1
        
        # Compute averages
        num_examples = len(examples)
        avg_policy_loss = total_policy_loss / num_examples
        avg_value_loss = total_value_loss / num_examples if total_value_loss > 0 else 0
        avg_loss = total_loss / num_examples
        eval_pct = (evaluated_count / num_examples * 100) if num_examples > 0 else 0
        
        # Update stats
        self.stats['policy_losses'].append(avg_policy_loss)
        self.stats['value_losses'].append(avg_value_loss)
        self.stats['total_losses'].append(avg_loss)
        self.stats['evaluated_positions'] = evaluated_count
        self.stats['evaluated_pct'] = eval_pct
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'total_loss': avg_loss,
            'evaluated_examples': evaluated_count,
            'evaluated_pct': eval_pct
        }
    
    def _uci_to_index(self, uci: str) -> int:
        """Convert UCI move string to policy head index."""
        from_square = (ord(uci[0]) - ord('a')) + (int(uci[1]) - 1) * 8
        to_square = (ord(uci[2]) - ord('a')) + (int(uci[3]) - 1) * 8
        
        move_idx = from_square * 64 + to_square
        return min(move_idx, 4671)
    
    def get_stats(self) -> Dict:
        """Get training statistics."""
        return {
            'avg_policy_loss': np.mean(self.stats['policy_losses'][-10:]) if self.stats['policy_losses'] else 0,
            'avg_value_loss': np.mean(self.stats['value_losses'][-10:]) if self.stats['value_losses'] else 0,
            'avg_total_loss': np.mean(self.stats['total_losses'][-10:]) if self.stats['total_losses'] else 0,
            'evaluated_positions': self.stats['evaluated_positions'],
            'evaluated_pct': self.stats['evaluated_pct'],
            'total_epochs_trained': len(self.stats['total_losses'])
        }


def board_to_encoding_from_fen(fen: str) -> np.ndarray:
    """Convert FEN string to board encoding."""
    board = chess.Board(fen)
    return board_to_encoding(board)
