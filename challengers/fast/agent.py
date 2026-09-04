"""Local harness entrypoint for the compiled-search challenger."""

import chess

try:
    from fastcore import HASH_KEY, position_from_fen
    from fastsearch import search_timed, warm_up
except ModuleNotFoundError:
    # Local harness challengers live below the repository root. Submission files
    # are flat in /agent, so their imports succeed without entering this branch.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from fastcore import HASH_KEY, position_from_fen
    from fastsearch import search_timed, warm_up

warm_up()
GAME_HISTORY: list[int] = []


def _remember_after(board: chess.Board, move: str) -> None:
    after = board.copy(stack=False)
    after.push_uci(move)
    _, after_state = position_from_fen(after.fen(en_passant="fen"))
    GAME_HISTORY.append(int(after_state[HASH_KEY]))


def get_move(fen: str, time_left_ms: int) -> str:
    board = chess.Board(fen)
    legal = list(board.legal_moves)
    fallback = min(legal, key=lambda move: move.uci()).uci()
    _, root_state = position_from_fen(fen)
    root_key = int(root_state[HASH_KEY])
    if not GAME_HISTORY or GAME_HISTORY[-1] != root_key:
        GAME_HISTORY.append(root_key)
    try:
        result = search_timed(fen, time_left_ms, GAME_HISTORY)
    except Exception as error:
        print(f"fast search fallback: {type(error).__name__}: {error}")
        _remember_after(board, fallback)
        return fallback
    if chess.Move.from_uci(result.move) not in board.legal_moves:
        print("fast search fallback: illegal move")
        _remember_after(board, fallback)
        return fallback
    _remember_after(board, result.move)
    print(
        f"depth={result.depth} score={result.score} nodes={result.nodes} "
        f"time={result.elapsed_ms:.1f}ms"
    )
    return result.move
