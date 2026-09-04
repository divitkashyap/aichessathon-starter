"""Stream a bounded NNUE training set from Lichess evaluations on Colab.

The optional ``datasets`` dependency is intentionally not part of the agent's
runtime environment.  Install it only in the training notebook.
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Iterable
from typing import Any

import chess

from nnue.features import canonical_position_key, encode_fen
from nnue.shards import ShardWriter

MATE_CP = 10_000


def _stable_hash(value: str) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode(), digest_size=8).digest())


def _score(row: dict[str, Any]) -> float | None:
    cp = row.get("cp")
    if cp is not None:
        return float(cp)
    mate = row.get("mate")
    if mate is None or int(mate) == 0:
        return None
    distance = min(500, abs(int(mate)) * 10)
    return float((MATE_CP - distance) if int(mate) > 0 else (-MATE_CP + distance))


def build(
    rows: Iterable[dict[str, Any]],
    writer: ShardWriter,
    *,
    positions: int,
    minimum_depth: int,
    validation_permyriad: int,
) -> tuple[int, int]:
    """Keep the first (best/highest-depth) row for each de-normalized FEN."""
    seen: set[int] = set()
    accepted = 0
    rejected = 0
    for row in rows:
        if accepted >= positions:
            break
        if int(row.get("depth") or 0) < minimum_depth:
            rejected += 1
            continue
        target = _score(row)
        if target is None:
            rejected += 1
            continue
        try:
            key = canonical_position_key(str(row["fen"]))
        except (KeyError, ValueError):
            rejected += 1
            continue
        fingerprint = _stable_hash(key)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        try:
            board = chess.Board(f"{key} 0 1")
            if not board.is_valid():
                rejected += 1
                continue
            encoded = encode_fen(key)
        except ValueError:
            rejected += 1
            continue
        split = "validation" if fingerprint % 10_000 < validation_permyriad else "train"
        writer.add(split, encoded, max(-MATE_CP, min(MATE_CP, target)))
        accepted += 1
        if accepted % 100_000 == 0:
            print(f"accepted={accepted:,} rejected={rejected:,}")
    return accepted, rejected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--positions", type=int, default=2_000_000)
    parser.add_argument("--minimum-depth", type=int, default=18)
    parser.add_argument("--validation-permyriad", type=int, default=200)
    parser.add_argument("--shard-size", type=int, default=100_000)
    arguments = parser.parse_args()
    if not 0 < arguments.validation_permyriad < 10_000:
        raise SystemExit("validation-permyriad must be between 1 and 9999")

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise SystemExit("install the training-only dependency: pip install datasets") from error

    rows = load_dataset(
        "Lichess/chess-position-evaluations",
        split="train",
        streaming=True,
    )
    writer = ShardWriter(arguments.output, arguments.shard_size)
    accepted, rejected = build(
        rows,
        writer,
        positions=arguments.positions,
        minimum_depth=arguments.minimum_depth,
        validation_permyriad=arguments.validation_permyriad,
    )
    manifest = writer.close(
        {
            "source": "Lichess/chess-position-evaluations",
            "label_pov": "white",
            "minimum_depth": arguments.minimum_depth,
            "accepted": accepted,
            "rejected": rejected,
        }
    )
    print(manifest)


if __name__ == "__main__":
    main()
