"""
Dataset Builder: Merge PGN and puzzle data into unified training examples.
Creates balanced training batches with move probability distributions.
"""
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import chess
from .pgn_loader import GameData, PGNToFENConverter, PGNLoader
from .puzzle_loader import Puzzle, PuzzleToTrainingData, PuzzleLoader


class TrainingExample:
    """Represents a single training example: FEN -> move probabilities."""
    
    def __init__(self, fen: str, best_move_uci: str, source: str = "game"):
        self.fen = fen
        self.best_move_uci = best_move_uci
        self.source = source  # "game" or "puzzle"
        self.move_probabilities = {}  # {move_uci: probability}
        
    def __repr__(self):
        return f"TrainingExample(source={self.source}, best={self.best_move_uci})"


class DatasetBuilder:
    """Build a unified dataset from multiple sources."""
    
    def __init__(self):
        self.training_examples: List[TrainingExample] = []
        self.fen_to_examples: Dict[str, List[TrainingExample]] = defaultdict(list)
        
    def add_pgn_games(self, pgn_files: List[str], weight: float = 1.0) -> int:
        """
        Load games from PGN files and add to dataset.
        
        Args:
            pgn_files: List of PGN file paths
            weight: Weight for balancing sources (e.g., games weighted less than puzzles)
            
        Returns:
            Number of training examples added
        """
        count = 0
        for filepath in pgn_files:
            print(f"Loading PGN: {filepath}")
            games = PGNLoader.load_from_file(filepath)
            print(f"  Loaded {len(games)} games")
            
            for game in games:
                # Enrich with FEN sequence
                PGNToFENConverter.enrich_game_data(game)
                
                # Create training examples from each FEN-move pair
                for fen, move_uci in game.fen_sequence:
                    example = TrainingExample(fen, move_uci, source="game")
                    self.training_examples.append(example)
                    self.fen_to_examples[fen].append(example)
                    count += 1
        
        print(f"Added {count} training examples from PGN games")
        return count
    
    def add_puzzles(self, puzzle_files: List[str], weight: float = 2.0) -> int:
        """
        Load puzzles and add to dataset.
        Puzzles typically receive higher weight since they are curated critical positions.
        
        Args:
            puzzle_files: List of puzzle file paths
            weight: Weight multiplier (puzzles usually more important than games)
            
        Returns:
            Number of training examples added
        """
        count = 0
        for filepath in puzzle_files:
            print(f"Loading puzzles: {filepath}")
            puzzles = PuzzleLoader.load_from_file(filepath)
            print(f"  Loaded {len(puzzles)} puzzles")
            
            for puzzle in puzzles:
                # Convert coordinate moves to UCI if needed
                PuzzleLoader.convert_coordinate_moves_to_uci(puzzle)
                
                # Generate training pairs
                pairs = PuzzleToTrainingData.generate_training_pairs(puzzle)
                
                for fen, move_uci in pairs:
                    # Add puzzle examples multiple times based on weight
                    for _ in range(int(weight)):
                        example = TrainingExample(fen, move_uci, source="puzzle")
                        self.training_examples.append(example)
                        self.fen_to_examples[fen].append(example)
                        count += 1
        
        print(f"Added {count} training examples from puzzles")
        return count
    
    def compute_move_probabilities(self) -> None:
        """
        For each unique FEN, compute normalized move probability distribution.
        Useful for supervised learning with soft targets.
        """
        for fen, examples in self.fen_to_examples.items():
            move_counts = defaultdict(int)
            
            # Count occurrences of each move for this position
            for example in examples:
                move_counts[example.best_move_uci] += 1
            
            total = sum(move_counts.values())
            
            # Normalize to probability distribution
            for example in examples:
                probabilities = {}
                for move_uci, count in move_counts.items():
                    probabilities[move_uci] = count / total
                example.move_probabilities = probabilities
    
    def get_training_examples(self) -> List[TrainingExample]:
        """Get all training examples."""
        return self.training_examples
    
    def get_unique_positions(self) -> int:
        """Get number of unique FEN positions in dataset."""
        return len(self.fen_to_examples)
    
    def get_statistics(self) -> Dict[str, any]:
        """Get dataset statistics."""
        total_examples = len(self.training_examples)
        puzzle_examples = sum(1 for ex in self.training_examples if ex.source == "puzzle")
        game_examples = total_examples - puzzle_examples
        
        return {
            "total_examples": total_examples,
            "puzzle_examples": puzzle_examples,
            "game_examples": game_examples,
            "unique_positions": self.get_unique_positions(),
            "puzzle_percentage": (puzzle_examples / total_examples * 100) if total_examples > 0 else 0,
        }
    
    def save_checkpoint(self, filepath: str) -> None:
        """
        Save dataset to a checkpoint file (for debugging/analysis).
        
        Args:
            filepath: Path to save checkpoint
        """
        import json
        
        checkpoint = {
            "examples": [
                {
                    "fen": ex.fen,
                    "best_move": ex.best_move_uci,
                    "source": ex.source,
                    "move_probs": ex.move_probabilities
                }
                for ex in self.training_examples
            ],
            "stats": self.get_statistics()
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            print(f"Dataset checkpoint saved: {filepath}")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
