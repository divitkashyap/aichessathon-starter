"""Real-v3 integer parity and isolated search restoration gates."""
import math
from pathlib import Path
import random
import sys
import unittest

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "challengers/nnue_v3"))
import nnue_search as search
import nnue_core as core
import nnue_incremental as incremental
from nnue.features import encode_board, PADDING_INDEX


class NeuralSearchTests(unittest.TestCase):
    def assert_position(self, source, board, state, white, black):
        encoded = encode_board(source)
        for indices, actual in ((encoded.white, white), (encoded.black, black)):
            expected = search.NNUE_FEATURE_BIAS.astype(np.int64) + search.NNUE_FEATURE[
                indices[indices != PADDING_INDEX]
            ].sum(axis=0, dtype=np.int64)
            np.testing.assert_array_equal(actual, expected)
        expected_board, expected_state = core.position_from_fen(source.fen(en_passant="fen"))
        np.testing.assert_array_equal(board, expected_board)
        np.testing.assert_array_equal(state, expected_state)
        us, them = (white, black) if source.turn else (black, white)
        delta = np.clip(us, 0, search.NNUE_FEATURE_SCALE) - np.clip(them, 0, search.NNUE_FEATURE_SCALE)
        total = (search.NNUE_TEMPO + int(search.NNUE_OUTPUT_WEIGHT.astype(np.int64) @ delta)) * search.NNUE_OUTPUT_SCALE_CP
        denominator = search.NNUE_FEATURE_SCALE * search.NNUE_WEIGHT_SCALE
        expected_score = (1 if total >= 0 else -1) * ((abs(total) + denominator // 2) // denominator)
        self.assertEqual(search.evaluate_nnue(state, white, black), expected_score)

    def exercise(self, fen, first_move=None):
        source = chess.Board(fen)
        board, state = core.position_from_fen(fen)
        white, black, kings = incremental.initialize_accumulators(board, search.NNUE_FEATURE, search.NNUE_FEATURE_BIAS)
        rng = random.Random(42)
        stack = []
        self.assert_position(source, board, state, white, black)
        for ply in range(128 if first_move is None else 1):
            legal = {core.move_to_uci(int(move)): int(move) for move in core.generate_legal_moves(board, state)}
            self.assertEqual(set(legal), {move.uci() for move in source.legal_moves})
            if not legal:
                break
            uci = first_move if first_move else rng.choice(sorted(legal))
            move = legal[uci]
            undo = np.empty(core.UNDO_SIZE, dtype=np.int64)
            incremental.make_move_with_accumulators(board, state, move, undo, search.NNUE_FEATURE, search.NNUE_FEATURE_BIAS, white, black, kings)
            source.push_uci(uci)
            stack.append((move, undo))
            self.assert_position(source, board, state, white, black)
        for move, undo in reversed(stack):
            incremental.unmake_move_with_accumulators(board, state, move, undo, search.NNUE_FEATURE, search.NNUE_FEATURE_BIAS, white, black, kings)
            source.pop()
            self.assert_position(source, board, state, white, black)

    def test_real_weights_random_playout_and_unmake(self):
        self.exercise(chess.STARTING_FEN)

    def test_real_weights_special_moves(self):
        for fen, move in [
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
            ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8c8"),
            ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"),
            ("4k2r/6P1/8/8/8/8/8/4K3 w - - 0 1", "g7h8q"),
        ]:
            with self.subTest(move=move):
                self.exercise(fen, move)

    def test_search_restores_board_completed_and_timeout(self):
        search.warm_up()
        for depth, deadline in ((3, math.inf), (30, 0.0)):
            board, state = core.position_from_fen(chess.STARTING_FEN)
            before_board, before_state = board.copy(), state.copy()
            _, move, _, complete = search.search_root(board, state, depth, deadline)
            self.assertEqual(complete, deadline == math.inf)
            self.assertIn(chess.Move.from_uci(core.move_to_uci(move)), chess.Board().legal_moves)
            np.testing.assert_array_equal(board, before_board)
            np.testing.assert_array_equal(state, before_state)
