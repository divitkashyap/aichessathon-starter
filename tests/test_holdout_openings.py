"""Legality and separation checks for the non-engine-selected holdout suite."""

import unittest

import chess

from tools.openings import (
    HOLDOUT_OPENING_LINES,
    opening_fens,
    opening_lines,
)


class HoldoutOpeningTests(unittest.TestCase):
    def test_holdout_has_twelve_unique_legal_lines_and_fens(self) -> None:
        lines = opening_lines("holdout")
        self.assertEqual(lines, HOLDOUT_OPENING_LINES)
        self.assertEqual(len(lines), 12)
        self.assertEqual(len(set(lines)), 12)
        for line in lines:
            board = chess.Board()
            for move in line.split():
                board.push_uci(move)
            self.assertTrue(board.is_valid(), msg=line)

        fens = opening_fens("holdout")
        self.assertEqual(len(fens), 12)
        self.assertEqual(len(set(fens)), 12)

    def test_holdout_fens_are_disjoint_from_existing_suites(self) -> None:
        # Different move counters do not make the same position independent.
        identity = lambda fen: " ".join(fen.split()[:4])
        holdout = {identity(fen) for fen in opening_fens("holdout")}
        self.assertEqual(len(holdout), 12)
        self.assertTrue(holdout.isdisjoint(map(identity, opening_fens("legacy"))))
        self.assertTrue(holdout.isdisjoint(map(identity, opening_fens("validation"))))


if __name__ == "__main__":
    unittest.main()
