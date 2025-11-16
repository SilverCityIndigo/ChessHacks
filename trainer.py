"""
Neural Network Trainer for Chess Bot
Trains supervised learning model on game/puzzle data.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Tuple, Optional, List
import chess
import pickle
import os
from pathlib import Path


def get_input_size() -> int:
    """Compute the input size used by board_to_encoding."""
    import chess
    return len(board_to_encoding(chess.Board()))


class ChessNeuralNetwork(nn.Module):
    """Neural network for chess move prediction."""
    
    def __init__(self, board_encoding_size: int = None, num_moves: int = 4672):
        """
        Initialize chess neural network.
        
        Args:
            board_encoding_size: Size of input board encoding (8x8x12 = 768 for piece planes)
            num_moves: Number of possible moves (UCI move space)
        """
        super(ChessNeuralNetwork, self).__init__()
        
        self.board_encoding_size = board_encoding_size or get_input_size()
        self.num_moves = num_moves
        
        # Policy head (move prediction)
        self.feature_layers = nn.Sequential(
            nn.Linear(self.board_encoding_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_moves)
        )
        
        # Value head (position evaluation, 0-1 scale)
        self.value_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input board encodings (batch_size, board_encoding_size)
        
        Returns:
            Tuple of (policy logits, value estimates)
        """
        features = self.feature_layers(x)
        policy = self.policy_head(features)
        value = self.value_head(features)
        return policy, value


def board_to_encoding(board: chess.Board) -> np.ndarray:
    """
    Encode chess board to numerical features.
    Uses piece planes: 12 planes (6 piece types × 2 colors).
    
    Args:
        board: python-chess Board object
    
    Returns:
        Numpy array of shape (768,) - 8×8×12 flattened
    """
    encoding = np.zeros((8, 8, 12), dtype=np.float32)
    
    # Piece type mapping: 0=pawn, 1=knight, 2=bishop, 3=rook, 4=queen, 5=king
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        
        # Get square coordinates
        rank = chess.square_rank(square)  # 0-7 (from bottom to top in standard notation)
        file = chess.square_file(square)  # 0-7 (from left to right)
        
        # Determine piece type and color
        piece_type_idx = piece_types.index(piece.piece_type)
        color_offset = 0 if piece.color == chess.WHITE else 6
        
        encoding[7 - rank, file, piece_type_idx + color_offset] = 1.0
    
    planes = encoding.flatten()

    # Extra global features to improve context
    extras: List[float] = []
    # Side to move: 1 if white, 0 if black
    extras.append(1.0 if board.turn == chess.WHITE else 0.0)
    # Castling rights (WK, WQ, BK, BQ)
    extras.append(1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0)
    extras.append(1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0)
    extras.append(1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0)
    extras.append(1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0)
    # Halfmove clock normalized (cap at 100)
    extras.append(min(board.halfmove_clock, 100) / 100.0)
    # En passant file scalar (0 if none)
    extras.append((chess.square_file(board.ep_square) / 7.0) if board.ep_square is not None else 0.0)

    # Mobility features
    legal_moves = list(board.legal_moves)
    max_moves_norm = 60.0  # heuristic upper bound
    extras.append(min(len(legal_moves), max_moves_norm) / max_moves_norm)  # overall mobility
    # Piece-type mobility counts for side to move
    mobility_counts = {chess.PAWN:0, chess.KNIGHT:0, chess.BISHOP:0, chess.ROOK:0, chess.QUEEN:0}
    for mv in legal_moves:
        piece = board.piece_at(mv.from_square)
        if piece and piece.color == board.turn and piece.piece_type in mobility_counts:
            mobility_counts[piece.piece_type] += 1
    # Normalization denominators (rough maxima)
    denom = {chess.PAWN:16, chess.KNIGHT:10, chess.BISHOP:13, chess.ROOK:14, chess.QUEEN:28}
    for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        extras.append(mobility_counts[pt] / denom[pt])

    # Attacked squares counts (distinct)
    attacked_white = set()
    attacked_black = set()
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            for target in board.attacks(sq):
                if piece.color == chess.WHITE:
                    attacked_white.add(target)
                else:
                    attacked_black.add(target)
    extras.append(len(attacked_white)/64.0)
    extras.append(len(attacked_black)/64.0)

    # King safety (own king attackers & pawn shield)
    own_king_sq = board.king(board.turn)
    opp_king_sq = board.king(not board.turn)
    attackers_on_own_king = 0
    attackers_on_opp_king = 0
    if own_king_sq is not None:
        for sq in board.attackers(not board.turn, own_king_sq):
            attackers_on_own_king += 1
    if opp_king_sq is not None:
        for sq in board.attackers(board.turn, opp_king_sq):
            attackers_on_opp_king += 1
    # Normalize (rough max attackers 16)
    extras.append(attackers_on_own_king / 16.0)
    extras.append(attackers_on_opp_king / 16.0)
    # Pawn shield (count pawns in front of own king within one file left/right and one/two ranks forward)
    pawn_shield = 0
    if own_king_sq is not None:
        k_file = chess.square_file(own_king_sq)
        k_rank = chess.square_rank(own_king_sq)
        rank_dir = 1 if board.turn == chess.WHITE else -1
        for df in (-1,0,1):
            f = k_file + df
            if 0 <= f < 8:
                for dr in (rank_dir, 2*rank_dir):
                    r = k_rank + dr
                    if 0 <= r < 8:
                        sq = chess.square(f, r)
                        p = board.piece_at(sq)
                        if p and p.color == board.turn and p.piece_type == chess.PAWN:
                            pawn_shield += 1
    extras.append(pawn_shield / 6.0)  # max shield considered

    return np.concatenate([planes, np.array(extras, dtype=np.float32)])


def move_to_index(move: chess.Move, legal_moves: list) -> int:
    """
    Convert move to index in the move space.
    For simplicity, return index in the legal moves list.
    
    Args:
        move: chess.Move object
        legal_moves: List of legal moves
    
    Returns:
        Index of move in legal moves list
    """
    try:
        return legal_moves.index(move)
    except ValueError:
        # Move not in legal moves - shouldn't happen with valid training data
        return 0


class ChessTrainer:
    """Trainer for chess neural network."""
    
    def __init__(self, model: Optional[ChessNeuralNetwork] = None, device: str = "cpu"):
        """
        Initialize trainer.
        
        Args:
            model: ChessNeuralNetwork instance (creates new if None)
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.model = model or ChessNeuralNetwork().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.policy_loss = nn.CrossEntropyLoss()
        self.value_loss = nn.MSELoss()
        self.training_history = {
            'policy_loss': [],
            'value_loss': [],
            'total_loss': [],
            'epoch': []
        }
    
    def prepare_training_data(self, examples: List, batch_size: int = 32) -> DataLoader:
        """
        Prepare training data from TrainingExample objects.
        
        Args:
            examples: List of TrainingExample objects from dataset
            batch_size: Batch size for training
        
        Returns:
            PyTorch DataLoader
        """
        board_encodings = []
        move_indices = []
        move_probabilities = []
        
        for example in examples:
            try:
                # Decode FEN to board
                board = chess.Board(example.fen)
                
                # Get board encoding
                encoding = board_to_encoding(board)
                board_encodings.append(encoding)
                
                # Get legal moves
                legal_moves_list = list(board.generate_legal_moves())
                if not legal_moves_list:
                    continue
                
                # Convert move to index
                move = chess.Move.from_uci(example.best_move_uci)
                if move not in legal_moves_list:
                    continue
                
                move_idx = legal_moves_list.index(move)
                move_indices.append(move_idx)
                
                # Use pre-computed move probabilities if available
                prob_dist = example.move_probabilities
                move_probabilities.append(prob_dist)
                
            except Exception as e:
                # Skip invalid examples
                print(f"Skipped example: {e}")
                continue
        
        if not board_encodings:
            raise ValueError("No valid training examples found")
        
        # Convert to tensors
        board_encodings = torch.FloatTensor(board_encodings).to(self.device)
        move_indices = torch.LongTensor(move_indices).to(self.device)
        
        # Create a dummy value tensor (0.5 for neutral evaluation)
        values = torch.FloatTensor([[0.5] for _ in range(len(board_encodings))]).to(self.device)
        
        # Create dataset and loader
        dataset = TensorDataset(board_encodings, move_indices, values)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        return loader
    
    def train(self, train_loader: DataLoader, epochs: int = 10, 
              validation_loader: Optional[DataLoader] = None) -> Dict:
        """
        Train the model.
        
        Args:
            train_loader: Training DataLoader
            epochs: Number of training epochs
            validation_loader: Optional validation DataLoader
        
        Returns:
            Dictionary with training history
        """
        self.model.train()
        
        for epoch in range(epochs):
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            epoch_total_loss = 0.0
            num_batches = 0
            
            for batch_encodings, batch_moves, batch_values in train_loader:
                self.optimizer.zero_grad()
                
                # Forward pass
                policy_logits, value_pred = self.model(batch_encodings)
                
                # Calculate losses
                policy_loss = self.policy_loss(policy_logits, batch_moves)
                value_loss = self.value_loss(value_pred, batch_values)
                total_loss = 0.7 * policy_loss + 0.3 * value_loss  # Weighted combination
                
                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Gradient clipping
                self.optimizer.step()
                
                # Track losses
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_total_loss += total_loss.item()
                num_batches += 1
            
            # Average losses over batches
            avg_policy_loss = epoch_policy_loss / num_batches
            avg_value_loss = epoch_value_loss / num_batches
            avg_total_loss = epoch_total_loss / num_batches
            
            # Store history
            self.training_history['epoch'].append(epoch + 1)
            self.training_history['policy_loss'].append(avg_policy_loss)
            self.training_history['value_loss'].append(avg_value_loss)
            self.training_history['total_loss'].append(avg_total_loss)
            
            print(f"Epoch {epoch + 1}/{epochs} - "
                  f"Policy Loss: {avg_policy_loss:.4f}, "
                  f"Value Loss: {avg_value_loss:.4f}, "
                  f"Total Loss: {avg_total_loss:.4f}")
            
            # Optional validation
            if validation_loader:
                val_loss = self.validate(validation_loader)
                print(f"  Validation Loss: {val_loss:.4f}")
        
        return self.training_history
    
    def validate(self, validation_loader: DataLoader) -> float:
        """
        Validate model on validation set.
        
        Args:
            validation_loader: Validation DataLoader
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_encodings, batch_moves, batch_values in validation_loader:
                policy_logits, value_pred = self.model(batch_encodings)
                
                policy_loss = self.policy_loss(policy_logits, batch_moves)
                value_loss = self.value_loss(value_pred, batch_values)
                total_loss_batch = 0.7 * policy_loss + 0.3 * value_loss
                
                total_loss += total_loss_batch.item()
                num_batches += 1
        
        self.model.train()
        return total_loss / num_batches
    
    def save_model(self, path: str) -> None:
        """
        Save trained model to disk.
        
        Args:
            path: File path to save model
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """
        Load trained model from disk.
        
        Args:
            path: File path to load model
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")


if __name__ == "__main__":
    # Test the trainer
    print("Chess Neural Network Trainer")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    model = ChessNeuralNetwork()
    trainer = ChessTrainer(model, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

