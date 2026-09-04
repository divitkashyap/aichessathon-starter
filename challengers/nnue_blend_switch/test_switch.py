"""Import and dispatch-wiring checks for the sparse-piece search switch."""

from __future__ import annotations

import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import chess


class SwitchImportTests(unittest.TestCase):
    def test_agent_imports_both_searches(self) -> None:
        agent = importlib.import_module("agent")
        self.assertEqual(agent.MIN_BLEND_PIECES, 17)
        self.assertTrue(callable(agent.classical_search_timed))
        self.assertTrue(callable(agent.blended_search_timed))
        self.assertTrue(callable(agent.classical_warm_up))
        self.assertTrue(callable(agent.blended_warm_up))

    def test_classical_search_uses_private_core(self) -> None:
        classical = importlib.import_module("nnue_blend_classical_search")
        self.assertEqual(classical.__name__, "nnue_blend_classical_search")

    def test_sparse_and_dense_dispatch(self):
        agent = importlib.import_module("agent")
        for fen, sparse in (("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", True),
                            (chess.STARTING_FEN, False)):
            move = next(iter(chess.Board(fen).legal_moves)).uci()
            result = SimpleNamespace(move=move, depth=1, score=0, nodes=1, elapsed_ms=0.)
            agent.GAME_HISTORY.clear()
            with patch.object(agent, "classical_search_timed", return_value=result) as classical:
                with patch.object(agent, "blended_search_timed", return_value=result) as neural:
                    self.assertEqual(agent.get_move(fen, 1000), move)
                    self.assertEqual(classical.call_count, int(sparse))
                    self.assertEqual(neural.call_count, int(not sparse))


if __name__ == "__main__":
    unittest.main()
