"""Tests for the paired-arena opening suites."""

import unittest

import chess

from tools.openings import (
    DEFAULT_SUITE,
    OPENING_LINES,
    VALIDATION_OPENING_LINES,
    opening_fens,
    opening_lines,
)


class ArenaOpeningTests(unittest.TestCase):
    def test_every_line_is_legal_and_unique(self) -> None:
        for suite in ("legacy", "validation"):
            lines = opening_lines(suite)
            self.assertEqual(len(lines), len(set(lines)))
            fens = opening_fens(suite)
            self.assertEqual(len(fens), len(set(fens)))
            for line in lines:
                board = chess.Board()
                for move in line.split():
                    board.push_uci(move)

    def test_suites_do_not_overlap(self) -> None:
        self.assertTrue(set(OPENING_LINES).isdisjoint(VALIDATION_OPENING_LINES))
        self.assertTrue(set(opening_fens("legacy")).isdisjoint(opening_fens("validation")))

    def test_default_suite_is_stable_legacy(self) -> None:
        self.assertEqual(DEFAULT_SUITE, "legacy")
        self.assertEqual(opening_lines(), OPENING_LINES)
        self.assertEqual(opening_fens(), opening_fens("legacy"))
        self.assertEqual(len(opening_lines()), 12)
        self.assertEqual(len(opening_lines("validation")), 12)


if __name__ == "__main__":
    unittest.main()
