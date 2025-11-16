"""
PGN Loader: Parse PGN files and extract game data.
Supports both classic game format and modern multi-game files.
"""
import re
from typing import List, Tuple, Optional, Dict
import chess
import chess.pgn
from io import StringIO


class GameData:
    """Represents a single chess game with metadata and moves."""
    
    def __init__(self, headers: Dict[str, str], move_stack: List[str]):
        self.headers = headers  # Game metadata (Event, Date, Result, etc.)
        self.move_stack = move_stack  # List of moves in algebraic notation
        self.fen_sequence = []  # Will store FEN after each move
        
    def __repr__(self):
        return f"GameData({self.headers.get('Event', 'Unknown')} - {self.headers.get('Result', '?')})"


class PGNLoader:
    """Load and parse chess games from PGN format."""
    
    @staticmethod
    def load_from_file(filepath: str) -> List[GameData]:
        """
        Load games from a PGN file using python-chess library.
        Properly handles annotations, variations, and comments.
        
        Args:
            filepath: Path to the PGN file
            
        Returns:
            List of GameData objects
        """
        games = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                while True:
                    # Use python-chess built-in PGN parser (handles all annotations)
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    
                    # Extract headers
                    headers = dict(game.headers)
                    
                    # Extract clean moves (python-chess strips annotations automatically)
                    moves = []
                    board = game.board()
                    for move in game.mainline_moves():
                        moves.append(board.san(move))
                        board.push(move)
                    
                    if moves:
                        game_data = GameData(headers, moves)
                        games.append(game_data)
                    
        except Exception as e:
            print(f"Error loading PGN file {filepath}: {e}")
            
        return games
    



class PGNToFENConverter:
    """Convert PGN move sequences to FEN sequences."""
    
    @staticmethod
    def get_fen_sequence(moves: List[str], start_fen: str = None) -> List[Tuple[str, str]]:
        """
        Convert a list of moves to (FEN, move) pairs.
        
        Args:
            moves: List of moves in algebraic notation
            start_fen: Starting FEN (default: standard starting position)
            
        Returns:
            List of (FEN_before_move, move_uci) tuples for training
        """
        if start_fen is None:
            start_fen = chess.STARTING_FEN
            
        board = chess.Board(start_fen)
        fen_move_pairs = []
        
        for move_san in moves:
            try:
                # Convert SAN to move object
                move = board.parse_san(move_san)
                
                # Record FEN before move and the move itself
                fen_move_pairs.append((board.fen(), move.uci()))
                
                # Apply move
                board.push(move)
                
            except Exception as e:
                print(f"Error parsing move '{move_san}': {e}")
                # Continue with next move
                continue
        
        return fen_move_pairs
    
    @staticmethod
    def enrich_game_data(game: GameData) -> None:
        """
        Enriches GameData object with FEN sequence.
        Modifies game in place.
        
        Args:
            game: GameData object to enrich
        """
        # Try to extract starting FEN from headers
        start_fen = game.headers.get('FEN', chess.STARTING_FEN)
        game.fen_sequence = PGNToFENConverter.get_fen_sequence(game.move_stack, start_fen)
