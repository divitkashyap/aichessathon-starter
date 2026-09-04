"""Competition entrypoint exposing the required ``get_move`` function."""

import chess

from engine import ChessEngine

# One process serves one game, so search state and transpositions survive between moves.
ENGINE = ChessEngine()


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation.

    fen           the position to move in; your colour is the side to move
    time_left_ms  your clock before this move, in milliseconds
    returns       "e2e4", or "e7e8q" for a promotion

    The process stays alive between your moves, so state you keep on a module or in a
    closure survives to the next call. It does not survive to the next game.

    print() is safe. Your stdout is redirected away from the protocol stream, discarded
    during rated games and shown back to you in the validation log.
    """
    board = chess.Board(fen)

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        # The referee never requests a move from a terminal position, but returning a
        # protocol-shaped value is safer than crashing if a malformed match does.
        return "0000"

    fallback = min(legal_moves, key=lambda move: move.uci())
    try:
        result = ENGINE.choose_move(board, time_left_ms)
    except Exception as error:  # A legal move is always better than a crash loss.
        print(f"search fallback: {type(error).__name__}: {error}")
        return fallback.uci()

    if result.move not in board.legal_moves:
        print("search fallback: engine returned an illegal move")
        return fallback.uci()
    pv = " ".join(move.uci() for move in result.principal_variation)
    print(
        f"depth={result.depth} score={result.score} nodes={result.nodes} "
        f"time={result.elapsed_ms:.1f}ms pv={pv}"
    )
    return result.move.uci()
