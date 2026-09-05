"""Reproduce depth sensitivity, without claiming our evaluator is an oracle."""

import argparse
import hashlib
import json
from pathlib import Path

from challengers.lmr_lazy_order import lmr_search as engine
from tools.review_pgn import _forced_move_score

CASES = (
    ("round12_mate", "r5k1/pppb1r2/1b1p3B/3N1p2/q1P5/5P2/PP1Q2PP/4RR1K w - - 5 23", "d5f6"),
    ("round16_h4", "1rb2rk1/6pp/p2p1p2/2q1p1P1/PpP1P1n1/3B1Q2/1P1RN1PP/2K4R w - - 0 20", "h2h4"),
    ("round17_Rf7", "r2q1rk1/6b1/1pnnbp2/pNp3p1/2P1P2p/P4N1P/1PQ1BPP1/3R1RK1 b - - 1 23", "f8f7"),
    ("round18_f2", "8/8/8/1p6/p2p4/P2P1pk1/1P6/4K3 b - - 1 53", "f3f2"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depths", type=int, nargs="+", default=[4, 5, 6, 7])
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite")
    if any(d < 1 for d in args.depths):
        raise SystemExit("Depth must be positive")
    result = {"notice": "Depth sensitivity is not proof of evaluation error. "
              "No prior repetition history; playing build unknown. Forced move "
              "uses interior search semantics. All scores from v8, not an oracle.",
              "source_sha256": hashlib.sha256(Path(engine.__file__).read_bytes()).hexdigest(),
              "cases": []}
    for name, fen, move in CASES:
        rows = []
        for depth in args.depths:
            best = engine.search_fixed_depth(fen, depth)
            forced = _forced_move_score(engine, fen, move, depth)
            row = dict(depth=depth, best_move=best.move, best_score=best.score,
                       forced_score=forced, gap=best.score-forced, nodes=best.nodes)
            rows.append(row)
            print(name, row, flush=True)
        result["cases"].append(dict(name=name, fen=fen, move=move, depths=rows))
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)


if __name__ == "__main__":
    main()
