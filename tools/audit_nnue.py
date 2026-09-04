"""Bounded public-data diagnostic, never imported by a competition agent.

Sample complete FEN groups at distant row offsets. This is a diagnostic sample,
not a certified held-out set: the original training FEN manifest is unavailable.
"""

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import chess
import numpy as np

from fastsearch import evaluate
from tools.stream_lichess_nnue import _score


def sample_rows(pages, offset_start=50_000_000):
    selected = {}
    for index in range(pages):
        offset = offset_start + index * 23_000_000
        query = urllib.parse.urlencode(
            dict(
                dataset="Lichess/chess-position-evaluations",
                config="default",
                split="train",
                offset=offset,
                length=100,
            )
        )
        with urllib.request.urlopen(
            "https://datasets-server.huggingface.co/rows?" + query, timeout=30
        ) as response:
            rows = [item["row"] for item in json.load(response)["rows"]]
        # Boundaries can cut through a multi-PV group. Discard both entire groups.
        boundaries = {rows[0]["fen"], rows[-1]["fen"]}
        for row in rows:
            fen = row["fen"]
            if fen in boundaries or fen in selected or int(row["depth"]) < 18:
                continue
            score = _score(row)
            if score is None:
                continue
            board = chess.Board(fen + " 0 1")
            if board.is_valid() and not board.is_game_over():
                selected[fen] = dict(row, offset=offset)
        print(f"pages={index + 1} unique_positions={len(selected)}", flush=True)
    return list(selected.values())


def metrics(rows):
    result = {}
    for name, predicate in (
        ("all", lambda r: True),
        ("ordinary_abs_target_le_300", lambda r: r["mate"] is None and abs(r["target"]) <= 300),
        (
            "quiet_ordinary",
            lambda r: (
                r["mate"] is None
                and abs(r["target"]) <= 300
                and not r["in_check"]
                and not r["has_capture"]
            ),
        ),
        ("extreme_or_mate", lambda r: r["mate"] is not None or abs(r["target"]) > 1000),
        (
            "ordinary_17plus_pieces",
            lambda r: r["mate"] is None and abs(r["target"]) <= 300 and r["pieces"] >= 17,
        ),
        (
            "ordinary_8orless_pieces",
            lambda r: r["mate"] is None and abs(r["target"]) <= 300 and r["pieces"] <= 8,
        ),
    ):
        subset = [r for r in rows if predicate(r)]
        if not subset:
            continue
        stats = {"count": len(subset)}
        for evaluator in ("classical", "neural", "blend", "gated_blend"):
            errors = np.array([abs(r[evaluator] - r["target"]) for r in subset])
            stats[evaluator] = dict(
                mae_cp=round(float(errors.mean()), 2),
                median_error_cp=round(float(np.median(errors)), 2),
                p90_error_cp=round(float(np.quantile(errors, 0.9)), 2),
            )
        result[name] = stats
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=20)
    parser.add_argument("--offset-start", type=int, default=50_000_000)
    parser.add_argument("--exclude", type=Path, help="Exclude positions from a prior diagnostic")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--input", type=Path, help="Reuse raw_rows from a prior audit without network"
    )
    args = parser.parse_args()
    if not 1 <= args.pages <= 35:
        parser.error("pages must be between 1 and 35")
    if args.offset_start < 0 or args.offset_start + (args.pages - 1) * 23_000_000 > 950_000_000:
        parser.error("requested offset range exceeds this diagnostic's bounded dataset window")
    if args.output.exists():
        parser.error("refusing to replace existing report")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "challengers/nnue_v3"))
    import nnue_search as neural

    raw = (json.loads(args.input.read_text())["raw_rows"] if args.input
           else sample_rows(args.pages, args.offset_start))
    if args.exclude:
        excluded = {row["fen"] for row in json.loads(args.exclude.read_text())["raw_rows"]}
        raw = [row for row in raw if row["fen"] not in excluded]
    rows = []
    for item in raw:
        board = chess.Board(item["fen"] + " 0 1")
        compact, state = neural.position_from_fen(board.fen(en_passant="fen"))
        white, black, _ = neural.initialize_accumulators(
            compact, neural.NNUE_FEATURE, neural.NNUE_FEATURE_BIAS
        )
        classical = int(evaluate(compact, state))
        score = int(neural.evaluate_nnue(state, white, black))
        target = float(_score(item)) * (1 if board.turn else -1)
        pieces = len(board.piece_map())
        total = 8 * classical + max(-200, min(200, score - classical))
        blend = (1 if total >= 0 else -1) * ((abs(total) + 4) // 8)
        rows.append(
            dict(
                fen=item["fen"],
                depth=item["depth"],
                mate=item["mate"],
                target=target,
                classical=classical,
                neural=score,
                blend=blend,
                gated_blend=blend if pieces >= 17 else classical,
                pieces=pieces,
                in_check=board.is_check(),
                has_capture=any(board.is_capture(move) for move in board.legal_moves),
            )
        )
    summary = metrics(rows)
    report = dict(
        source="https://huggingface.co/datasets/Lichess/chess-position-evaluations",
        training_overlap="unknown",
        excluded_report=str(args.exclude) if args.exclude else None,
        blend_rounding="nearest_symmetric",
        raw_rows=raw,
        positions=rows,
        metrics=summary,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(summary, indent=2))
    for row in sorted(rows, key=lambda r: abs(r["neural"] - r["target"]), reverse=True)[:5]:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
