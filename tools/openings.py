"""Small, varied opening suite for deterministic paired engine matches."""

import chess

DEFAULT_SUITE = "legacy"

OPENING_LINES = (
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6",
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6",
    "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4",
    "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6",
    "c2c4 e7e5 b1c3 g8f6 g1f3 b8c6 g2g3 d7d5",
    "g1f3 d7d5 g2g3 g8f6 f1g2 g7g6 e1g1 f8g7",
    "e2e4 c7c6 d2d4 d7d5 b1c3 d5e4 c3e4 c8f5",
    "e2e4 e7e6 d2d4 d7d5 b1c3 g8f6 e4e5 f6d7",
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6",
    "e2e4 c7c5 g1f3 b8c6 d2d4 c5d4 f3d4 g7g6",
    "d2d4 g8f6 c2c4 c7c5 d4d5 e7e6 b1c3 e6d5 c4d5",
    "e2e4 c7c5 c2c3 d7d5 e4d5 d8d5 d2d4 g8f6",
)

VALIDATION_OPENING_LINES = (
    "e2e4 e7e5 b1c3 g8f6 f2f4 d7d5 e4d5",
    "e2e4 e7e5 d2d4 e5d4 c2c3 d4c3 b1c3",
    "e2e4 c7c5 b1c3 b8c6 g2g3 g7g6 f1g2",
    "e2e4 c7c6 d2d4 d7d5 e4e5 c8f5 f2f4",
    "e2e4 d7d6 d2d4 g8f6 b1c3 g7g6",
    "e2e4 g7g6 d2d4 f8g7 b1c3 c7c6",
    "d2d4 g8f6 g1f3 e7e6 c2c4 f8b4",
    "d2d4 d7d5 g1f3 g8f6 c2c4 c7c6",
    "c2c4 c7c5 b1c3 b8c6 g2g3 g7g6",
    "c2c4 g8f6 g2g3 d7d5 c4d5 f6d5",
    "g1f3 d7d5 g2g3 c7c5 f1g2 b8c6",
    "b2b3 d7d5 c1b2 g8f6 e2e3 e7e6",
)


def opening_lines(suite: str = DEFAULT_SUITE) -> tuple[str, ...]:
    if suite == DEFAULT_SUITE:
        return OPENING_LINES
    if suite == "validation":
        return VALIDATION_OPENING_LINES
    raise ValueError(f"unknown opening suite: {suite}")


def opening_fens(suite: str = DEFAULT_SUITE) -> tuple[str, ...]:
    fens: list[str] = []
    for line in opening_lines(suite):
        board = chess.Board()
        for uci in line.split():
            board.push_uci(uci)
        fens.append(board.fen())
    return tuple(fens)
