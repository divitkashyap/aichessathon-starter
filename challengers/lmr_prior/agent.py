"""Local harness entrypoint for the isolated LMR challenger."""

import math
import time

import chess
from lmr_core import HASH_KEY, position_from_fen
from lmr_search import TT_MASK, _tt_key, new_search_memory, search_timed, warm_up
from move_prior import prior_move

warm_up()
prior_move(*position_from_fen(chess.STARTING_FEN))
BUILD_ID = "einsteinanium-lmr-neural-prior-challenger"
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
        memory = None
        remaining = time_left_ms
        if len(board.piece_map()) >= 17 and time_left_ms >= 1000:
            started = time.perf_counter()
            pieces, state = position_from_fen(fen)
            preferred = prior_move(pieces, state)
            memory = new_search_memory()
            key = _tt_key(state)
            index = int(key) & TT_MASK
            memory.tt_keys[index] = key
            memory.tt_moves[index] = preferred
            # Depth zero supplies root ordering only. Positive-depth search
            # cannot reuse its dummy score, and leaf search bypasses the TT.
            memory.tt_depths[index] = 0
            remaining = max(1, time_left_ms - math.ceil((time.perf_counter() - started) * 1000))
        result = search_timed(fen, remaining, GAME_HISTORY, memory=memory)
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
