"""Exploratory bounded mobility fit on existing diagnostic positions, offline only."""

import argparse
import json
from pathlib import Path

import chess
import numpy as np

from challengers.lmr_lazy_order.lmr_core import position_from_fen
from challengers.lmr_lazy_order.lmr_search import evaluate


def mobility(board: chess.Board) -> int:
    """Side-to-move pseudo-mobility; excludes own pieces and enemy pawn attacks.

    This does not claim squares are safe from every piece, or account for pins.
    """
    score = 0
    for color in (chess.WHITE, chess.BLACK):
        pawn_attacks = 0
        for square in board.pieces(chess.PAWN, not color):
            pawn_attacks |= int(board.attacks(square))
        forbidden = board.occupied_co[color] | pawn_attacks
        count = 0
        for kind in (chess.KNIGHT, chess.BISHOP, chess.ROOK):
            for square in board.pieces(kind, color):
                count += (int(board.attacks(square)) & ~forbidden).bit_count()
        score += count if color == board.turn else -count
    return score


def load_rows(path: Path) -> list[dict]:
    rows = []
    for item in json.loads(path.read_text())["positions"]:
        fen = item["fen"]
        if len(fen.split()) == 4:
            fen += " 0 1"
        board = chess.Board(fen)
        raw, state = position_from_fen(fen)
        rows.append(dict(item, baseline=int(evaluate(raw, state)),
                         mobility=mobility(board), pieces=len(board.piece_map())))
    return rows


def report(rows: list[dict], coefficient: int) -> dict:
    result = {}
    for name, limit in (("ordinary", 300), ("nonmate_le_1000", 1000)):
        subset = [r for r in rows if r["mate"] is None and abs(r["target"]) <= limit]
        result[name] = {"count": len(subset)}
        for label, weight in (("baseline", 0), ("mobility", coefficient)):
            errors = [abs(r["baseline"] + weight * r["mobility"] - r["target"])
                      for r in subset]
            result[name][label + "_mae_cp"] = float(np.mean(errors))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit", type=Path)
    parser.add_argument("replication", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite report")
    fit = load_rows(args.fit)
    replication = load_rows(args.replication)
    overlap = {r["fen"] for r in fit} & {r["fen"] for r in replication}
    if overlap:
        raise SystemExit("Diagnostic sets overlap")
    # Choose only on fit set; preserve material/PST and cap at eight cp/square.
    grid = {c: report(fit, c)["nonmate_le_1000"]["mobility_mae_cp"]
            for c in range(9)}
    coefficient = min(grid, key=grid.get)
    result = dict(coefficient_cp=coefficient, fit_grid=grid,
                  fit=report(fit, coefficient),
                  replication=report(replication, coefficient),
                  caveat="Previously explored diagnostic sets, not pristine holdouts. "
                         "Static errors do not establish playing strength.")
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
