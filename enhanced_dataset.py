"""
Enhanced Dataset Builder with Position Evaluations.
Augments training data with Stockfish evaluations for stronger value head training.
"""
import chess
from pathlib import Path
from typing import List, NamedTuple, Optional
from src.engines.stockfish_evaluator import StockfishEvaluator
import json


class EnhancedTrainingExample(NamedTuple):
    """Training example with position evaluation."""
    fen: str
    best_move_uci: str
    move_probabilities: dict  # UCI move -> probability
    source: str  # "game" or "puzzle"
    position_evaluation: Optional[float] = None  # 0-1, None if not evaluated


class EnhancedDatasetBuilder:
    """
    Builds training dataset with Stockfish evaluations.
    Creates stronger training targets for value head learning.
    """
    
    def __init__(self, evaluator: Optional[StockfishEvaluator] = None):
        self.examples: List[EnhancedTrainingExample] = []
        self.evaluator = evaluator or StockfishEvaluator()
        self.stats = {
            "total_examples": 0,
            "evaluated_positions": 0,
            "games_processed": 0,
            "puzzles_processed": 0,
        }
    
    def add_pgn_games(self, pgn_files: List[str], weight: float = 1.0) -> int:
        """
        Load PGN games and evaluate positions.
        
        Args:
            pgn_files: List of PGN file paths
            weight: Weight multiplier for examples from these games
            
        Returns:
            Number of examples added
        """
        from src.data.pgn_loader import PGNLoader
        
        count = 0
        loader = PGNLoader()
        
        for pgn_file in pgn_files:
            if not Path(pgn_file).exists():
                print(f"  ⚠ PGN file not found: {pgn_file}")
                continue
            
            games = loader.load(pgn_file)
            
            for game in games:
                # Get moves from game
                for i, move in enumerate(game.moves):
                    board = chess.Board()
                    
                    # Apply moves up to this position
                    for m in game.moves[:i]:
                        board.push(m)
                    
                    fen = board.fen()
                    move_uci = move.uci()
                    
                    # Evaluate position
                    evaluation, _ = self.evaluator.evaluate(board)
                    
                    # Get move probabilities (basic: best move gets 1.0)
                    move_probs = {move_uci: 1.0}
                    
                    example = EnhancedTrainingExample(
                        fen=fen,
                        best_move_uci=move_uci,
                        move_probabilities=move_probs,
                        source="game",
                        position_evaluation=evaluation
                    )
                    
                    self.examples.append(example)
                    count += 1
                    
                    if count % 100 == 0:
                        print(f"    Processed {count} moves...")
                
                self.stats["games_processed"] += 1
        
        self.stats["total_examples"] += count
        print(f"Added {count} training examples from PGN games")
        return count
    
    def add_puzzles(self, puzzle_files: List[str], weight: float = 2.0) -> int:
        """
        Load puzzle positions with evaluations.
        
        Args:
            puzzle_files: List of puzzle file paths
            weight: Weight multiplier for puzzle examples
            
        Returns:
            Number of examples added
        """
        from src.data.puzzle_loader import PuzzleLoader
        
        count = 0
        loader = PuzzleLoader()
        
        for puzzle_file in puzzle_files:
            if not Path(puzzle_file).exists():
                print(f"  ⚠ Puzzle file not found: {puzzle_file}")
                continue
            
            puzzles = loader.load(puzzle_file)
            
            for puzzle in puzzles:
                board = chess.Board(puzzle.fen)
                
                # Evaluate puzzle position
                evaluation, _ = self.evaluator.evaluate(board)
                
                # Best move should get high probability
                move_probs = {puzzle.best_move.uci(): 1.0}
                
                example = EnhancedTrainingExample(
                    fen=puzzle.fen,
                    best_move_uci=puzzle.best_move.uci(),
                    move_probabilities=move_probs,
                    source="puzzle",
                    position_evaluation=evaluation
                )
                
                self.examples.append(example)
                count += 1
            
            self.stats["puzzles_processed"] += len(puzzles)
        
        # Apply weight multiplier (duplicate puzzle examples for emphasis)
        puzzle_examples = [ex for ex in self.examples if ex.source == "puzzle"]
        for _ in range(int(weight - 1)):
            self.examples.extend(puzzle_examples[-len(puzzles):])
        
        self.stats["total_examples"] = len(self.examples)
        print(f"Added {count} training examples from puzzles")
        return count
    
    def get_training_examples(self) -> List[EnhancedTrainingExample]:
        """Get all training examples with evaluations."""
        return self.examples
    
    def get_statistics(self) -> dict:
        """Get dataset statistics."""
        evaluated = sum(1 for ex in self.examples if ex.position_evaluation is not None)
        
        return {
            "total_examples": len(self.examples),
            "evaluated_positions": evaluated,
            "games_processed": self.stats["games_processed"],
            "puzzles_processed": self.stats["puzzles_processed"],
            "avg_evaluation": (
                sum(ex.position_evaluation for ex in self.examples if ex.position_evaluation)
                / evaluated if evaluated > 0 else 0.5
            )
        }
    
    def save_dataset(self, output_path: str):
        """Save dataset to JSON for analysis."""
        data = []
        for ex in self.examples:
            data.append({
                "fen": ex.fen,
                "best_move": ex.best_move_uci,
                "evaluation": ex.position_evaluation,
                "source": ex.source
            })
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Dataset saved to {output_path}")
