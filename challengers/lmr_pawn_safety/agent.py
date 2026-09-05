"""Local harness entrypoint for the isolated LMR challenger."""

import chess
from lmr_core import HASH_KEY, position_from_fen
from lmr_search import search_timed, warm_up

warm_up()
BUILD_ID = "einsteinanium-lmr-pawn-safety-challenger"
print(f"build={BUILD_ID}")
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
        print(f"lmr search fallback: {type(error).__name__}: {error}")
        _remember_after(board, fallback)
        return fallback
    if chess.Move.from_uci(result.move) not in board.legal_moves:
        print("lmr search fallback: illegal move")
        _remember_after(board, fallback)
        return fallback
    _remember_after(board, result.move)
    print(
        f"depth={result.depth} score={result.score} nodes={result.nodes} "
        f"time={result.elapsed_ms:.1f}ms"
    )
    return result.move
