"""Preserve the won endgame and a diagnostic false-positive regression."""

import json
from pathlib import Path
import unittest

import chess

from challengers.lmr_lazy_order import lmr_search as v8
from tools.review_pgn import _forced_move_score


class Round18Tests(unittest.TestCase):
    def test_recorded_conversion_is_legal_through_mate(self):
        fixture = json.loads(Path(__file__).with_name('round18_conversion.json').read_text())
        board = chess.Board(fixture['fen'])
        self.assertTrue(board.is_valid())
        promoted = en_passant = False
        for uci in fixture['line']:
            move = chess.Move.from_uci(uci)
            self.assertIn(move, board.legal_moves)
            promoted |= bool(move.promotion)
            en_passant |= board.is_en_passant(move)
            board.push(move)
        self.assertTrue(promoted)
        self.assertTrue(en_passant)
        self.assertTrue(board.is_checkmate())
        self.assertEqual(board.result(), '0-1')

    def test_f2_check_has_no_spurious_depth_seven_loss(self):
        fen = '8/8/8/1p6/p2p4/P2P1pk1/1P6/4K3 b - - 1 53'
        root = v8.search_fixed_depth(fen, 7)
        forced = _forced_move_score(v8, fen, 'f3f2', 7)
        self.assertEqual(root.move, 'f3f2')
        self.assertEqual(root.score, forced)
        self.assertGreater(forced, 800)


if __name__ == '__main__':
    unittest.main()
