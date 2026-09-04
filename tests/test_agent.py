"""Correctness and clock-safety tests for the submission entrypoint."""

import time
import unittest

import chess

import agent
from engine import ChessEngine, evaluate


class AgentTests(unittest.TestCase):
    def setUp(self) -> None:
        agent.ENGINE = ChessEngine()

    def assert_legal_reply(self, fen: str, time_left_ms: int = 1_000) -> chess.Move:
        board = chess.Board(fen)
        uci = agent.get_move(fen, time_left_ms)
        move = chess.Move.from_uci(uci)
        self.assertIn(move, board.legal_moves)
        return move

    def test_starting_position_returns_legal_move(self) -> None:
        self.assert_legal_reply(chess.STARTING_FEN)

    def test_mate_in_one_is_taken(self) -> None:
        fen = "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"
        board = chess.Board(fen)
        board.push(self.assert_legal_reply(fen))
        self.assertTrue(board.is_checkmate())

    def test_promotion_position_returns_legal_move(self) -> None:
        self.assert_legal_reply("8/P6k/8/8/8/8/6K1/8 w - - 0 1")

    def test_low_clock_keeps_a_safety_margin(self) -> None:
        started = time.perf_counter()
        self.assert_legal_reply(chess.STARTING_FEN, time_left_ms=100)
        elapsed_ms = (time.perf_counter() - started) * 1_000
        self.assertLess(elapsed_ms, 100)

    def test_evaluation_is_symmetric(self) -> None:
        board = chess.Board()
        self.assertEqual(evaluate(board), 0)
        board.turn = chess.BLACK
        self.assertEqual(evaluate(board), 0)


if __name__ == "__main__":
    unittest.main()
