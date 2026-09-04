"""Tests for the offline PGN review tool."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from tools import review_pgn


class Result:
    def __init__(self, move: str, score: int) -> None:
        self.move = move
        self.score = score


class ReviewPgnTests(unittest.TestCase):
    def write_pgn(self, directory: Path, text: str) -> Path:
        path = directory / "game.pgn"
        path.write_text(text, encoding="utf-8")
        return path

    def test_proxy_loss_uses_side_to_move_perspective(self) -> None:
        pgn = """[Event \"synthetic\"]

1. e4 {[%clk 0:10:00]} e5 {[%clk 0:10:00]} 2. Nf3 {[%clk 0:09:58]} *
"""
        temporary = tempfile.mkdtemp(prefix="review-pgn-test-")
        path = self.write_pgn(Path(temporary), pgn)
        calls: list[tuple[str, int]] = []

        def fake_search(fen: str, depth: int) -> Result:
            calls.append((fen, depth))
            if len(calls) == 1:
                return Result("e2e4", 80)
            if len(calls) == 2:
                return Result("a7a6", -30)
            return Result("g1f3", 20)

        report = review_pgn.review_pgn(path, "white", search=fake_search)
        record = report["games"][0]["moves"][0]  # type: ignore[index]
        self.assertEqual(record["recommended_move"], "e2e4")
        self.assertEqual(record["played_move"], "e2e4")
        self.assertTrue(record["same_as_recommended"])
        self.assertEqual(record["parent_view_child_score"], 30)
        self.assertEqual(record["proxy_loss"], 50)
        self.assertEqual(calls[1][1], 4)

    def test_illegal_pgn_move_is_rejected(self) -> None:
        pgn = "1. e5 {[%clk 0:10:00]} *\n"
        temporary = tempfile.mkdtemp(prefix="review-pgn-test-")
        path = self.write_pgn(Path(temporary), pgn)
        with self.assertRaises(ValueError):
            review_pgn.review_pgn(path, "white", search=lambda _fen, _depth: Result("e2e4", 0))

    def test_nonterminal_child_mate_distance_is_root_relative(self) -> None:
        for sign in (-1, 1):
            directory = Path(tempfile.mkdtemp(prefix="review-pgn-mate-test-"))
            path = self.write_pgn(directory, "1. e4 {[%clk 0:10:00]} *\n")
            search = Mock(side_effect=[Result("e2e4", 0), Result("e7e5", sign * (review_pgn.MATE - 3))])
            report = review_pgn.review_pgn(path, "white", search=search)
            record = report["games"][0]["moves"][0]
            self.assertEqual(record["parent_view_child_score"], -sign * (review_pgn.MATE - 4))

    def test_terminal_child_uses_mate_score_without_search(self) -> None:
        pgn = """[SetUp \"1\"]
[FEN \"7k/5Q2/6K1/8/8/8/8/8 w - - 0 1\"]

1. Qh7# {[%clk 0:10:00]} *
"""
        temporary = tempfile.mkdtemp(prefix="review-pgn-test-")
        path = self.write_pgn(Path(temporary), pgn)
        search = Mock(return_value=Result("f7h7", review_pgn.MATE - 1))
        report = review_pgn.review_pgn(path, "white", search=search)
        record = report["games"][0]["moves"][0]  # type: ignore[index]
        self.assertTrue(record["child_terminal"])
        self.assertTrue(record["same_as_recommended"])
        self.assertEqual(record["child_score"], -review_pgn.MATE + 1)
        self.assertEqual(search.call_count, 1)

    def test_output_refuses_overwrite_and_build_is_explicit(self) -> None:
        pgn = "1. e4 {[%clk 0:10:00]} *\n"
        temporary = tempfile.mkdtemp(prefix="review-pgn-test-")
        directory = Path(temporary)
        path = self.write_pgn(directory, pgn)
        output = directory / "report.json"
        output.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            review_pgn.write_new_json({}, output)
        report = review_pgn.review_pgn(
            path,
            "white",
            build="test-build",
            search=lambda _fen, _depth: Result("e2e4", 0),
        )
        self.assertEqual(report["build"], "test-build")
        self.assertEqual(output.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
