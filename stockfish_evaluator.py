"""
Stockfish Engine Integration for Position Evaluation.
Evaluates board positions and provides training targets for value head.
"""
import chess
import subprocess
import platform
import os
from pathlib import Path
from typing import Optional, Tuple, List


class StockfishEvaluator:
    """
    Wraps Stockfish engine for position evaluation.
    Evaluates positions and returns centipawn scores and best moves.
    """
    
    def __init__(self, depth: int = 12, time_limit_ms: int = 100):
        """
        Initialize Stockfish evaluator.
        
        Args:
            depth: Search depth for evaluation
            time_limit_ms: Maximum time per position in milliseconds
        """
        self.depth = depth
        self.time_limit_ms = time_limit_ms
        self.stockfish_path = self._find_stockfish()
        self.engine = None
        
        if self.stockfish_path:
            try:
                self._start_engine()
                print(f"✓ Stockfish loaded from: {self.stockfish_path}")
            except Exception as e:
                print(f"⚠ Failed to start Stockfish: {e}")
                print(f"  Falling back to basic evaluation")
                self.stockfish_path = None
    
    def _find_stockfish(self) -> Optional[str]:
        """Find Stockfish binary in common locations."""
        # Try common paths
        common_paths = []
        
        if platform.system() == "Windows":
            common_paths = [
                r"C:\Users\aquat\Downloads\stockfish-windows-x86-64-avx2 (3)\stockfish\stockfish-windows-x86-64-avx2.exe",
                "C:\\Program Files\\Stockfish\\stockfish.exe",
                "C:\\Program Files (x86)\\Stockfish\\stockfish.exe",
                str(Path.home() / "stockfish" / "stockfish.exe"),
            ]
        elif platform.system() == "Darwin":  # macOS
            common_paths = [
                "/usr/local/bin/stockfish",
                "/opt/homebrew/bin/stockfish",
                str(Path.home() / "stockfish"),
            ]
        else:  # Linux
            common_paths = [
                "/usr/bin/stockfish",
                "/usr/local/bin/stockfish",
                str(Path.home() / "stockfish"),
            ]
        
        for path in common_paths:
            if os.path.exists(path):
                return path
        
        # Try to find in PATH
        try:
            result = subprocess.run(
                ["where", "stockfish"] if platform.system() == "Windows" else ["which", "stockfish"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                return result.stdout.strip().split('\n')[0]
        except:
            pass
        
        return None
    
    def _start_engine(self):
        """Start Stockfish engine process and complete UCI handshake."""
        if not self.stockfish_path:
            raise RuntimeError("Stockfish not found")

        self.engine = subprocess.Popen(
            [self.stockfish_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        # Complete UCI handshake: send 'uci' and wait for 'uciok'
        self._write("uci")
        self._read_until_token("uciok")

        # Ensure engine is ready
        self._write("isready")
        self._read_until_token("readyok")
    
    def _write(self, command: str) -> None:
        """Write a command to the engine without waiting for a specific response."""
        if not self.engine or not self.engine.stdin:
            return
        self.engine.stdin.write(command + "\n")
        self.engine.stdin.flush()

    def _read_until_token(self, token: str) -> str:
        """Read lines until a line contains the given token. Returns the captured text."""
        if not self.engine or not self.engine.stdout:
            return ""
        buf = ""
        while True:
            line = self.engine.stdout.readline()
            if not line:
                break
            buf += line
            if token in line:
                break
        return buf

    def _go_and_wait_bestmove(self, go_command: str) -> str:
        """Send a go command and wait until 'bestmove' is returned. Returns full response."""
        self._write(go_command)
        return self._read_until_token("bestmove")
    
    def evaluate(self, board: chess.Board) -> Tuple[Optional[float], Optional[chess.Move]]:
        """
        Evaluate a position using Stockfish.
        
        Args:
            board: Chess position to evaluate
            
        Returns:
            Tuple of (evaluation_score [0-1], best_move)
            - evaluation_score: 0.5 = equal, 1.0 = white winning, 0.0 = black winning
            - best_move: Best move from this position
        """
        if not self.engine or not self.stockfish_path:
            return self._basic_evaluation(board)
        
        try:
            fen = board.fen()

            # Send position and search command
            self._write(f"position fen {fen}")
            # Prefer depth if provided, otherwise use movetime
            response = self._go_and_wait_bestmove(f"go depth {self.depth}")
            
            # Parse response for evaluation and best move
            score = None
            best_move = None
            
            for line in response.split('\n'):
                if 'score cp' in line:
                    parts = line.split()
                    cp_idx = parts.index('cp') + 1
                    score = int(parts[cp_idx])
                
                if 'bestmove' in line:
                    parts = line.split()
                    move_uci = parts[1]
                    try:
                        best_move = chess.Move.from_uci(move_uci)
                    except:
                        pass
            
            if score is not None:
                # Convert centipawns to 0-1 scale
                # Positive score favors white, negative favors black
                evaluation = 0.5 + (score / 4000.0)  # Normalize to roughly [0, 1]
                evaluation = max(0.0, min(1.0, evaluation))
                
                # Flip if black to move
                if not board.turn:
                    evaluation = 1.0 - evaluation
                
                return evaluation, best_move
            
            return self._basic_evaluation(board)
        
        except Exception as e:
            print(f"Stockfish evaluation failed: {e}")
            return self._basic_evaluation(board)

    def evaluate_multipv(self, board: chess.Board, top_k: int = 5) -> List[Tuple[str, float]]:
        """Return top-K moves with centipawn scores using Stockfish MultiPV.
        Each tuple: (uci_move, cp_score from side to move perspective).
        If engine unavailable, returns empty list.
        """
        if not self.engine or not self.stockfish_path:
            return []
        try:
            fen = board.fen()
            # Set MultiPV option
            self._write(f"setoption name MultiPV value {top_k}")
            self._write(f"position fen {fen}")
            response = self._go_and_wait_bestmove(f"go depth {self.depth}")
            lines = response.split('\n')
            results = []
            for line in lines:
                if 'multipv' in line and 'score cp' in line and 'pv' in line:
                    parts = line.strip().split()
                    # Extract multipv index
                    try:
                        mp_idx = parts.index('multipv')
                        pv_number = int(parts[mp_idx + 1])
                        cp_idx = parts.index('cp')
                        cp_score = int(parts[cp_idx + 1])
                        pv_idx = parts.index('pv')
                        first_move = parts[pv_idx + 1]
                        # Perspective: cp_score already relative to side to move
                        results.append((first_move, cp_score))
                    except Exception:
                        continue
            # Reset MultiPV to 1 to not slow future single evaluations
            self._write("setoption name MultiPV value 1")
            # Deduplicate in case of repeats
            dedup = {}
            for m, cp in results:
                if m not in dedup:
                    dedup[m] = cp
            # Keep only top_k unique
            ordered = list(dedup.items())[:top_k]
            return ordered
        except Exception as e:
            # On failure, attempt to reset and return empty list
            try:
                self._write("setoption name MultiPV value 1")
            except:
                pass
            return []
    
    def _basic_evaluation(self, board: chess.Board) -> Tuple[float, Optional[chess.Move]]:
        """
        Basic evaluation without Stockfish.
        Estimates position strength from material balance.
        """
        # Material values
        material_scores = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        
        white_score = sum(
            len(board.pieces(piece_type, chess.WHITE)) * value
            for piece_type, value in material_scores.items()
        )
        
        black_score = sum(
            len(board.pieces(piece_type, chess.BLACK)) * value
            for piece_type, value in material_scores.items()
        )
        
        # Normalize to [0, 1]
        total = white_score + black_score
        if total == 0:
            evaluation = 0.5
        else:
            evaluation = white_score / (white_score + black_score)
        
        # Flip if black to move (perspective)
        if not board.turn:
            evaluation = 1.0 - evaluation
        
        # Get best legal move (first move, not optimal)
        best_move = None
        legal_moves = list(board.legal_moves)
        if legal_moves:
            best_move = legal_moves[0]
        
        return evaluation, best_move
    
    def close(self):
        """Close Stockfish engine."""
        if self.engine:
            try:
                self._write("quit")
                self.engine.terminate()
            except:
                pass
    
    def __del__(self):
        """Ensure engine is closed on deletion."""
        self.close()
