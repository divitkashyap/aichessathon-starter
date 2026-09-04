import json
import unittest
from pathlib import Path

import chess


class CompetitionFixtureTests(unittest.TestCase):
    def test_recorded_positions_and_moves_are_legal(self):
        payload = json.loads(Path(__file__).with_name("competition_regressions.json").read_text())
        self.assertEqual(len({p["id"] for p in payload["positions"]}), len(payload["positions"]))
        for fixture in payload["positions"]:
            board = chess.Board(fixture["fen"])
            self.assertTrue(board.is_valid(), fixture["id"])
            for move in fixture.get("best", []) + fixture.get("avoid", []):
                self.assertIn(chess.Move.from_uci(move), board.legal_moves)
            if "recorded_mating_line" in fixture:
                for move in fixture["recorded_mating_line"]:
                    board.push_uci(move)
                self.assertTrue(board.is_checkmate(), fixture["id"])
