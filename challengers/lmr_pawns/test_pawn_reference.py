"""Slow board-based oracle independent of the compiled pawn rank arrays."""
import random
import unittest

import chess

from challengers.lmr_pawns import pawn_search as candidate


def reference(board):
    mg = eg = phase = 0
    for square, piece in board.piece_map().items():
        sign = 1 if piece.color else -1
        relative = square if piece.color else square ^ 56
        mg += sign * (int(candidate.PIECE_VALUES[piece.piece_type]) + int(candidate.MG_TABLE[piece.piece_type, relative]))
        eg += sign * (int(candidate.PIECE_VALUES[piece.piece_type]) + int(candidate.EG_TABLE[piece.piece_type, relative]))
        phase += int(candidate.PHASE_WEIGHTS[piece.piece_type])
    for color in (chess.WHITE, chess.BLACK):
        sign = 1 if color else -1
        if len(board.pieces(chess.BISHOP, color)) >= 2:
            mg += sign * 28
            eg += sign * 38
        pawns = list(board.pieces(chess.PAWN, color))
        enemies = list(board.pieces(chess.PAWN, not color))
        for file in range(8):
            own = [sq for sq in pawns if chess.square_file(sq) == file]
            if not own:
                continue
            if not any(abs(chess.square_file(sq) - file) == 1 for sq in pawns):
                mg -= sign * 10
                eg -= sign * 12
            mg -= sign * (len(own) - 1) * 10
            eg -= sign * (len(own) - 1) * 15
            front = max(own) if color else min(own)
            rank = chess.square_rank(front)
            blocked = any(abs(chess.square_file(sq) - file) <= 1 and
                          (chess.square_rank(sq) > rank if color else chess.square_rank(sq) < rank)
                          for sq in enemies)
            if not blocked:
                bonus = (0, 0, 8, 16, 30, 55, 90, 0)[rank if color else 7 - rank]
                mg += sign * (bonus // 2)
                eg += sign * bonus
    phase = min(24, phase)
    white_score = (mg * phase + eg * (24 - phase)) // 24
    return white_score if board.turn else -white_score


class PawnReferenceTests(unittest.TestCase):
    def test_random_positions_match_slow_oracle(self):
        rng = random.Random(20260904)
        for _game in range(4):
            board = chess.Board()
            for _ply in range(128):
                pieces, state = candidate.position_from_fen(board.fen())
                self.assertEqual(int(candidate.evaluate(pieces, state)), reference(board), board.fen())
                moves = list(board.legal_moves)
                if not moves:
                    break
                board.push(rng.choice(moves))
