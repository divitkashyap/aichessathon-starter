"""Tests for the check-extension search with the draw-aware evaluator."""

from __future__ import annotations

import random
import unittest

import chess

from challengers.lmr_checks_draws import lmr_core as core
from challengers.lmr_checks_draws import lmr_search as search
from challengers.lmr_draws import lmr_search as reference


BEFORE_BD7 = "r1b3k1/ppp2r2/1b1p3B/3N1p2/q1P5/5P2/PP1Q2PP/4RR1K b - - 4 22"


class ChecksDrawsTests(unittest.TestCase):
    def test_round12_nf6_mate(self) -> None:
        board = chess.Board(BEFORE_BD7)
        board.push_uci("c8d7")
        result = search.search_fixed_depth(board.fen(), 4)
        self.assertEqual(result.move, "d5f6")
        self.assertEqual(result.score, search.MATE - 9)

    def test_bare_material_is_zero(self) -> None:
        fen = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
        fixture = chess.Board(fen)
        self.assertTrue(fixture.is_valid())
        board, state = core.position_from_fen(fen)
        self.assertEqual(search.evaluate(board, state), 0)

    def test_evaluator_matches_draws_on_128_legal_positions(self) -> None:
        rng = random.Random(0xD4A75)
        for _ in range(128):
            board = chess.Board()
            for _ in range(rng.randrange(90)):
                if board.is_game_over():
                    break
                board.push(rng.choice(list(board.legal_moves)))
            fen = board.fen()
            actual_board, actual_state = core.position_from_fen(fen)
            reference_board, reference_state = reference.position_from_fen(fen)
            self.assertEqual(
                search.evaluate(actual_board, actual_state),
                reference.evaluate(reference_board, reference_state),
                msg=fen,
            )


if __name__ == "__main__":
    unittest.main()
