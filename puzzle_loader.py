"""
Puzzle Loader: Parse FEN + solution moves from puzzle text files.
Compatible with the original puzzle.txt format from Java project.
"""
from typing import List, Tuple, Optional, Dict
import chess


class Puzzle:
    """Represents a chess puzzle with FEN and solution moves."""
    
    def __init__(self, puzzle_id: str, fen: str, solution_moves: List[str]):
        self.puzzle_id = puzzle_id  # e.g., "Puzzle 1"
        self.fen = fen
        self.solution_moves = solution_moves  # Moves in algebraic notation or UCI
        self.fen_move_pairs = []  # Will store (FEN, move_uci) for training
        
    def __repr__(self):
        return f"{self.puzzle_id}: {self.fen[:30]}... ({len(self.solution_moves)} moves)"


class PuzzleLoader:
    """Load puzzles from text files in the original format."""
    
    @staticmethod
    def load_from_file(filepath: str) -> List[Puzzle]:
        """
        Load puzzles from a text file.
        Expected format:
            # Puzzle 1
            FEN: r4r1k/6pp/8/3QN3/8/8/5PPP/6K1 w - - 0 1
            3,4-1,5
            0,7-0,6
            ...
        
        Args:
            filepath: Path to the puzzle file
            
        Returns:
            List of Puzzle objects
        """
        puzzles = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            current_puzzle_id = None
            current_fen = None
            current_moves = []
            
            for line in lines:
                line = line.strip()
                
                if not line:
                    continue
                
                # Puzzle header: # Puzzle N
                if line.startswith('#') and 'Puzzle' in line:
                    # Save previous puzzle if exists
                    if current_puzzle_id and current_fen:
                        puzzle = Puzzle(current_puzzle_id, current_fen, current_moves)
                        puzzles.append(puzzle)
                    
                    current_puzzle_id = line.replace('#', '').strip()
                    current_fen = None
                    current_moves = []
                
                # FEN line
                elif line.startswith('FEN:'):
                    current_fen = line.replace('FEN:', '').strip()
                
                # Move sequence (row,col-row,col or UCI format)
                elif current_fen and line and not line.startswith('#'):
                    current_moves.append(line)
            
            # Don't forget the last puzzle
            if current_puzzle_id and current_fen:
                puzzle = Puzzle(current_puzzle_id, current_fen, current_moves)
                puzzles.append(puzzle)
        
        except Exception as e:
            print(f"Error loading puzzle file {filepath}: {e}")
        
        return puzzles
    
    @staticmethod
    def convert_coordinate_moves_to_uci(puzzle: Puzzle) -> None:
        """
        Convert coordinate-based moves (e.g., "3,4-1,5") to UCI format.
        Updates puzzle in place.
        
        Args:
            puzzle: Puzzle object with coordinate-based moves
        """
        board = chess.Board(puzzle.fen)
        uci_moves = []
        
        for move_str in puzzle.solution_moves:
            try:
                # Check if it's coordinate format: "row,col-row,col"
                if ',' in move_str and '-' in move_str:
                    parts = move_str.split('-')
                    from_coords = parts[0].split(',')
                    to_coords = parts[1].split(',')
                    
                    from_row, from_col = int(from_coords[0]), int(from_coords[1])
                    to_row, to_col = int(to_coords[0]), int(to_coords[1])
                    
                    # Convert to chess notation (row 0 = rank 8, col 0 = file a)
                    from_square = chess.square(from_col, 7 - from_row)
                    to_square = chess.square(to_col, 7 - to_row)
                    
                    move = chess.Move(from_square, to_square)
                    
                    # Check for promotion
                    if board.piece_at(from_square) and board.piece_at(from_square).symbol().lower() == 'p':
                        if (from_row == 1 and to_row == 0) or (from_row == 6 and to_row == 7):
                            # Promotion move - default to queen
                            move.promotion = chess.QUEEN
                    
                    uci_moves.append(move.uci())
                    board.push(move)
                else:
                    # Already in UCI or SAN format - try to parse
                    move = board.parse_san(move_str) if len(move_str) > 4 else chess.Move.from_uci(move_str)
                    uci_moves.append(move.uci())
                    board.push(move)
                    
            except Exception as e:
                print(f"Error converting move '{move_str}': {e}")
                # Keep original if conversion fails
                uci_moves.append(move_str)
        
        puzzle.solution_moves = uci_moves


class PuzzleToTrainingData:
    """Convert puzzles to training examples."""
    
    @staticmethod
    def generate_training_pairs(puzzle: Puzzle) -> List[Tuple[str, str]]:
        """
        Generate (FEN, best_move_uci) pairs from a puzzle.
        
        Args:
            puzzle: Puzzle object
            
        Returns:
            List of (FEN, best_move_uci) tuples
        """
        board = chess.Board(puzzle.fen)
        pairs = []
        
        for move_uci in puzzle.solution_moves:
            try:
                pairs.append((board.fen(), move_uci))
                move = chess.Move.from_uci(move_uci)
                board.push(move)
            except Exception as e:
                print(f"Error in puzzle {puzzle.puzzle_id}: {e}")
        
        return pairs
