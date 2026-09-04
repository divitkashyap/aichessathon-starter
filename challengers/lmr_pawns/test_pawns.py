"""Tests for the isolated pawn-structure evaluator challenger."""

import unittest

import chess

from challengers.lmr_pawns.pawn_core import position_from_fen
from challengers.lmr_pawns.pawn_search import evaluate, search_fixed_depth, search_timed


def _evaluation(fen: str) -> int:
    board, state = position_from_fen(fen)
    return int(evaluate(board, state))


class PawnEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        search_fixed_depth(chess.STARTING_FEN, 1)

    def test_mirrored_position_preserves_color_oriented_score(self) -> None:
        board = chess.Board("4k3/8/8/3p4/2P5/8/8/4K3 w - - 0 1")
        mirrored = board.mirror()
        self.assertEqual(_evaluation(board.fen()), _evaluation(mirrored.fen()))

    def test_isolated_and_doubled_pawns_are_penalized(self) -> None:
        isolated = "4k3/8/8/8/P1P5/8/8/4K3 w - - 0 1"
        connected = "4k3/8/8/8/1PP5/8/8/4K3 w - - 0 1"
        doubled = "4k3/8/8/8/8/P7/P7/4K3 w - - 0 1"
        spread = "4k3/8/8/8/8/P7/1P6/4K3 w - - 0 1"
        self.assertLess(_evaluation(isolated), _evaluation(connected))
        self.assertLess(_evaluation(doubled), _evaluation(spread))

    def test_passed_pawn_bonus_increases_with_relative_advancement(self) -> None:
        early = "4k3/8/8/8/8/8/P7/4K3 w - - 0 1"
        advanced = "4k3/8/8/8/8/P7/8/4K3 w - - 0 1"
        self.assertGreater(_evaluation(advanced), _evaluation(early))

    def test_enemy_pawn_uses_opposite_color_orientation(self) -> None:
        blocked = "4k3/3p4/8/8/4P3/8/8/4K3 w - - 0 1"
        clear = "4k3/7p/8/8/4P3/8/8/4K3 w - - 0 1"
        self.assertLess(_evaluation(blocked), _evaluation(clear))
        blocked_board = chess.Board(blocked)
        mirrored = blocked_board.mirror()
        self.assertEqual(_evaluation(blocked), _evaluation(mirrored.fen()))

    def test_search_returns_a_legal_move_on_pawn_fixture(self) -> None:
        fen = "4k3/8/8/8/8/P7/8/4K3 w - - 0 1"
        result = search_timed(fen, 250)
        self.assertIn(chess.Move.from_uci(result.move), chess.Board(fen).legal_moves)


if __name__ == "__main__":
    unittest.main()
