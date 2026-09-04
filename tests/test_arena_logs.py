"""Tests for paired-arena search-log summaries."""

import unittest

from tools.paired_arena import summarize_search_log


class ArenaLogTests(unittest.TestCase):
    def test_summarizes_search_lines_and_ignores_noise(self) -> None:
        log = (
            "build=example\n"
            "depth=4 score=23 nodes=100 time=1.5ms\n"
            "depth=5 score=-10 nodes=300 time=2.5ms\n"
            "diagnostic without search fields\n"
        )
        self.assertEqual(
            summarize_search_log(log),
            {
                "coverage": "partial_observed",
                "moves_logged": 2,
                "mean_completed_depth": 4.5,
                "total_reported_nodes": 400,
                "mean_move_ms": 2.0,
            },
        )

    def test_empty_log_has_zero_summary(self) -> None:
        self.assertEqual(
            summarize_search_log("build=example\n"),
            {
                "coverage": "partial_observed",
                "moves_logged": 0,
                "mean_completed_depth": None,
                "total_reported_nodes": 0,
                "mean_move_ms": None,
            },
        )


if __name__ == "__main__":
    unittest.main()
