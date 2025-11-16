"""
Chess Manager: Central coordinator for bot move generation.
Manages agent selection, game state, and move logging.
"""
import chess
from typing import Dict, Tuple, Optional
import time


class GameContext:
    """Context information for a game in progress."""
    
    def __init__(self, pgn_str: str, timeleft_ms: int):
        self.pgn = pgn_str
        self.timeleft = timeleft_ms  # Milliseconds remaining
        self.board = chess.Board()  # Will be populated from PGN
        self.move_probabilities = {}
        self.logs = []
        
        # Parse PGN to get board state
        self._parse_pgn()
    
    def _parse_pgn(self):
        """Parse PGN string and set up board."""
        self.board = chess.Board()
        
        if not self.pgn or self.pgn == '':
            return
        
        # Simple PGN parsing: split by space and apply moves
        moves_text = self.pgn.split()
        
        for move_san in moves_text:
            # Skip move numbers (1., 2., etc.)
            if move_san[-1] == '.':
                continue
            
            try:
                move = self.board.parse_san(move_san)
                self.board.push(move)
            except:
                # Skip invalid moves
                pass
    
    def log(self, message: str) -> None:
        """Add a log message."""
        self.logs.append(message)
    
    def logProbabilities(self, move_probs: Dict[chess.Move, float]) -> None:
        """Log move probabilities."""
        self.move_probabilities = move_probs


class ChessManager:
    """Manages chess bot agent and game state."""
    
    def __init__(self, agent=None):
        self.agent = agent
        self.entrypoint_func = None
        self.reset_func = None
        self.current_context = None
    
    def entrypoint(self, func):
        """Decorator to register the main move-generation function."""
        self.entrypoint_func = func
        return func
    
    def reset(self, func):
        """Decorator to register the reset function (called at game start)."""
        self.reset_func = func
        return func
    
    def set_context(self, pgn: str, timeleft_ms: int) -> None:
        """
        Set up the game context for the current move request.
        
        Args:
            pgn: PGN string representing moves so far
            timeleft_ms: Time left in milliseconds
        """
        self.current_context = GameContext(pgn, timeleft_ms)
    
    def get_model_move(self) -> Tuple[chess.Move, Dict, list]:
        """
        Get the model's move for the current position.
        
        Returns:
            Tuple of (move, move_probabilities, logs)
        """
        if self.entrypoint_func is None:
            raise RuntimeError("No entrypoint function registered")
        
        if self.current_context is None:
            raise RuntimeError("No game context set")
        
        # Call the registered entrypoint function
        move = self.entrypoint_func(self.current_context)
        
        return (
            move,
            self.current_context.move_probabilities,
            self.current_context.logs
        )
    
    def new_game(self) -> None:
        """Start a new game (call reset function)."""
        if self.reset_func:
            self.reset_func(self.current_context)
    
    def set_agent(self, agent) -> None:
        """Set the agent used for move selection."""
        self.agent = agent


# Global chess manager instance
chess_manager = ChessManager()
