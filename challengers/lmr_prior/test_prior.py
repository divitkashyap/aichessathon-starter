"""Run separately: unittest discover -s challengers/lmr_prior -p test_prior.py."""

import math
import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import chess
import numpy as np

from lmr_core import HASH_KEY, move_to_uci, position_from_fen
from lmr_search import TT_MASK, _history_array, _search_root, _tt_key, new_search_memory
from move_prior import prior_move


class PriorTests(unittest.TestCase):
    def test_agent_prior_budget_and_fresh_memory(self):
        import agent

        result = SimpleNamespace(move="e2e4", depth=1, score=0, nodes=1, elapsed_ms=1)
        agent.GAME_HISTORY.clear()
        move = prior_move(*position_from_fen(chess.STARTING_FEN))
        memories = []
        for _ in range(2):
            with patch.object(agent, "prior_move", return_value=move), \
                    patch.object(agent.time, "perf_counter", side_effect=[0, 0.005]), \
                    patch.object(agent, "search_timed", return_value=result) as search:
                self.assertEqual(agent.get_move(chess.STARTING_FEN, 2000), "e2e4")
                self.assertEqual(search.call_args.args[1], 1995)
                memory = search.call_args.kwargs["memory"]
                _, state = position_from_fen(chess.STARTING_FEN)
                index = int(_tt_key(state)) & TT_MASK
                self.assertEqual(memory.tt_depths[index], 0)
                self.assertEqual(memory.tt_moves[index], move)
                memories.append(memory)
        self.assertIsNot(memories[0], memories[1])

    def test_agent_skips_prior_on_low_clock(self):
        import agent

        result = SimpleNamespace(move="e2e4", depth=1, score=0, nodes=1, elapsed_ms=1)
        agent.GAME_HISTORY.clear()
        with patch.object(agent, "prior_move") as prior, \
                patch.object(agent, "search_timed", return_value=result) as search:
            agent.get_move(chess.STARTING_FEN, 999)
            prior.assert_not_called()
            self.assertEqual(search.call_args.args[1], 999)
            self.assertIsNone(search.call_args.kwargs["memory"])

    def test_legal_prior_and_exact_restoration(self):
        board = chess.Board()
        rng = random.Random(604)
        for _ in range(96):
            if board.is_game_over():
                board.reset()
            pieces, state = position_from_fen(board.fen())
            old_pieces, old_state = pieces.copy(), state.copy()
            move = prior_move(pieces, state)
            self.assertIn(chess.Move.from_uci(move_to_uci(move)), board.legal_moves)
            np.testing.assert_array_equal(pieces, old_pieces)
            np.testing.assert_array_equal(state, old_state)
            board.push(rng.choice(list(board.legal_moves)))

    def test_seed_changes_order_not_full_width_score(self):
        for fen in (chess.STARTING_FEN,
                    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"):
            pieces, state = position_from_fen(fen)
            history, count = _history_array((), int(state[HASH_KEY]))
            scores = []
            for seeded in (False, True):
                memory = new_search_memory()
                if seeded:
                    key = _tt_key(state)
                    index = int(key) & TT_MASK
                    memory.tt_keys[index] = key
                    memory.tt_moves[index] = prior_move(pieces, state)
                    memory.tt_depths[index] = 0
                    # Deliberately poisonous score must not be used as a bound.
                    memory.tt_scores[index] = 987654
                result = _search_root(
                    pieces, state, 2, math.inf, history, count,
                    memory.tt_keys, memory.tt_scores, memory.tt_moves,
                    memory.tt_depths, memory.tt_bounds,
                    memory.killer_moves, memory.history_scores,
                )
                self.assertTrue(result[3])
                scores.append(result[0])
            self.assertEqual(*scores)

    def test_terminal_prior_is_empty_and_restores(self):
        pieces, state = position_from_fen("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
        old_pieces, old_state = pieces.copy(), state.copy()
        self.assertEqual(prior_move(pieces, state), 0)
        np.testing.assert_array_equal(pieces, old_pieces)
        np.testing.assert_array_equal(state, old_state)
