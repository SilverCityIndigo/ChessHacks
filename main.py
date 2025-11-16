from .utils import chess_manager, GameContext
from .agents.supervised_agent import SupervisedAgent
from .agents.mcts_agent import MCTSAgent
from .training.trainer import ChessNeuralNetwork
try:
    from .training.resnet_model import ResNetChessNetwork
except ImportError:
    ResNetChessNetwork = None
from .engines.stockfish_evaluator import StockfishEvaluator
from chess import Move
import random
import chess
import time
import torch
from pathlib import Path
import os

# Load pre-trained model at startup
_model = None
_agent = None
_evaluator = None

# Configuration
USE_MCTS = os.getenv('USE_MCTS', 'false').lower() == 'true'
MCTS_SIMULATIONS = int(os.getenv('MCTS_SIMULATIONS', '100'))
USE_STOCKFISH_GUARD = os.getenv('USE_STOCKFISH_GUARD', 'true').lower() == 'true'
STK_GUARD_DEPTH = int(os.getenv('STK_GUARD_DEPTH', '10'))  # Deeper search
STK_GUARD_TOPK = int(os.getenv('STK_GUARD_TOPK', '5'))  # Check more moves

def _load_model():
    """Load the trained chess model."""
    global _model, _agent
    
    # Try improved model first, then fallback to original
    use_resnet = os.getenv('USE_RESNET', 'false').lower() == 'true'
    model_path_resnet = Path("src/models/chess_model_resnet.pth")
    model_path_improved = Path("src/models/chess_model_improved.pth")  # legacy naming if exists
    model_path_original = Path("src/models/chess_model.pth")
    
    if use_resnet and model_path_resnet.exists():
        model_path = model_path_resnet
    else:
        # Prefer improved then original
        model_path = model_path_improved if model_path_improved.exists() else model_path_original
    
    if not model_path.exists():
        print("⚠ Warning: Trained model not found at", model_path_original)
        print("  Falling back to random agent")
        return False
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Auto-detect input size from encoding helper
        if use_resnet and model_path == model_path_resnet and ResNetChessNetwork is not None:
            print("[OK] Loading ResNetChessNetwork model")
            _model = ResNetChessNetwork(num_moves=4672)
        else:
            if use_resnet and ResNetChessNetwork is None:
                print("[WARN] USE_RESNET requested but ResNetChessNetwork unavailable; falling back to ChessNeuralNetwork")
            _model = ChessNeuralNetwork(num_moves=4672)
        _model.load_state_dict(torch.load(model_path, map_location=device))
        _model.to(device)
        _model.eval()
        
        # Create appropriate agent
        if USE_MCTS:
            print("[OK] Initializing MCTS agent ({} simulations)".format(MCTS_SIMULATIONS))
            _agent = MCTSAgent(model=_model, device=device, num_simulations=MCTS_SIMULATIONS)
        else:
            print("[OK] Loaded trained model from {}".format(model_path))
            _agent = SupervisedAgent(name="ChessBot (Supervised)", model=_model)
        
        return True
    except Exception as e:
        print("[ERROR] Failed to load model: {}".format(e))
        return False


# Try to load trained model on startup
_model_loaded = _load_model()


@chess_manager.entrypoint
def supervised_move(ctx: GameContext):
    """
    EMERGENCY FIX: Use Stockfish directly until we can train a proper model.
    The puzzle-only training created a bot that only knows tactics, not chess.
    """
    global _evaluator
    
    # Initialize Stockfish if needed (graceful fallback to random if unavailable)
    if _evaluator is None:
        try:
            _evaluator = StockfishEvaluator(depth=12)
        except Exception as e:
            print(f"[WARN] Stockfish unavailable: {e}")
            _evaluator = False  # Mark as unavailable
    
    legal_moves = list(ctx.board.legal_moves)
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available")
    
    # Fallback to random if Stockfish failed to initialize
    if _evaluator is False:
        move = random.choice(legal_moves)
        ctx.logProbabilities({m: 1.0 / len(legal_moves) for m in legal_moves})
        return move
    
    # Evaluate ALL legal moves with Stockfish
    best_move = None
    best_score = float('-inf')
    move_scores = {}
    
    for move in legal_moves:
        try:
            ctx.board.push(move)
            score, _ = _evaluator.evaluate(ctx.board)
            ctx.board.pop()
            
            # Flip score for opponent's perspective
            signed_score = -score  # After our move, it's opponent's turn
            move_scores[move] = signed_score
            
            if signed_score > best_score:
                best_score = signed_score
                best_move = move
        except Exception as e:
            ctx.board.pop() if len(ctx.board.move_stack) > 0 else None
            continue
    
    # Convert scores to probabilities for logging
    if move_scores:
        # Use softmax-like conversion
        max_score = max(move_scores.values())
        exp_scores = {m: pow(2, (s - max_score) / 100) for m, s in move_scores.items()}
        total = sum(exp_scores.values())
        move_probs = {m: exp_s / total for m, exp_s in exp_scores.items()}
        ctx.logProbabilities(move_probs)
    
    return best_move if best_move else legal_moves[0]


@chess_manager.reset
def reset_func(ctx: GameContext):
    """Reset state for new game."""
    pass
