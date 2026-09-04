"""Offline PGN review against the terminal LMR search."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import chess
import chess.pgn

from challengers.lmr_terminal.lmr_search import MATE, search_fixed_depth

SearchFunction = Callable[[str, int], Any]


def _clock_seconds(node: chess.pgn.ChildNode, game_number: int, ply: int) -> float:
    clock = node.clock()
    if clock is None:
        raise ValueError(f"game {game_number} ply {ply}: missing [%clk] annotation")
    return float(clock)


def _terminal_score(board: chess.Board) -> int:
    if board.is_checkmate():
        # The child is one ply below the reviewed root.  This matches the
        # search's mate-distance convention from its root perspective.
        return -MATE + 1
    return 0


def _review_game(
    game: chess.pgn.Game,
    game_number: int,
    color: chess.Color,
    depth: int,
    search: SearchFunction,
) -> list[dict[str, object]]:
    board = game.board()
    records: list[dict[str, object]] = []
    for ply, node in enumerate(game.mainline(), start=1):
        move = node.move
        if move is None or move not in board.legal_moves:
            raise ValueError(f"game {game_number} ply {ply}: illegal PGN move")
        if board.turn == color:
            fen = board.fen(en_passant="fen")
            clock_secs = _clock_seconds(node, game_number, ply)
            root_result = search(fen, depth)
            recommended = str(root_result.move)
            if chess.Move.from_uci(recommended) not in board.legal_moves:
                raise ValueError(
                    f"game {game_number} ply {ply}: engine returned illegal move {recommended}"
                )
            root_score = int(root_result.score)
            played = move.uci()
            board.push(move)
            outcome = board.outcome(claim_draw=False)
            if outcome is not None:
                child_score = _terminal_score(board)
            else:
                child_result = search(board.fen(en_passant="fen"), depth - 1)
                child_score = int(child_result.score)
            parent_view_child_score = -child_score
            records.append(
                {
                    "fen": fen,
                    "ply": ply,
                    "clock_secs": clock_secs,
                    "recommended_move": recommended,
                    "played_move": played,
                    "same_as_recommended": played == recommended,
                    "root_score": root_score,
                    "child_score": child_score,
                    "parent_view_child_score": parent_view_child_score,
                    "proxy_loss": max(0, root_score - parent_view_child_score),
                    "child_terminal": outcome is not None,
                }
            )
            continue
        board.push(move)
    return records


def review_pgn(
    pgn_path: Path,
    color_name: str,
    depth: int = 5,
    build: str | None = None,
    search: SearchFunction = search_fixed_depth,
) -> dict[str, object]:
    if color_name not in {"white", "black"}:
        raise ValueError("color must be white or black")
    if depth < 2:
        raise ValueError("depth must be at least 2 because child search uses depth - 1")
    if not pgn_path.is_file():
        raise ValueError(f"PGN path is not a file: {pgn_path}")

    color = chess.WHITE if color_name == "white" else chess.BLACK
    games: list[dict[str, object]] = []
    with pgn_path.open(encoding="utf-8") as stream:
        game_number = 0
        while True:
            game = chess.pgn.read_game(stream)
            if game is None:
                break
            game_number += 1
            if game.errors:
                raise ValueError(f"game {game_number}: PGN parse error: {game.errors[0]}")
            games.append(
                {
                    "game_number": game_number,
                    "moves": _review_game(game, game_number, color, depth, search),
                }
            )
    if not games:
        raise ValueError(f"PGN contains no games: {pgn_path}")
    return {
        "diagnostic_notice": "Scores are our-engine diagnostics, not ground truth.",
        "build": build if build is not None else "unknown",
        "color": color_name,
        "depth": depth,
        "games": games,
    }


def write_new_json(report: dict[str, object], output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pgn", type=Path, help="explicit local PGN path")
    parser.add_argument("--color", required=True, choices=("black", "white"))
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--build")
    arguments = parser.parse_args()
    if arguments.output.exists():
        parser.error(f"refusing to overwrite existing output: {arguments.output}")
    try:
        report = review_pgn(arguments.pgn, arguments.color, arguments.depth, arguments.build)
        write_new_json(report, arguments.output)
    except (FileExistsError, OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
