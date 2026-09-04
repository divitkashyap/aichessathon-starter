"""Portable sharded dataset writer used by local and Colab data pipelines."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nnue.features import EncodedPosition


@dataclass(frozen=True, slots=True)
class ShardBatch:
    white: np.ndarray
    black: np.ndarray
    turn: np.ndarray
    target_cp: np.ndarray


class ShardWriter:
    """Write bounded compressed shards without ever replacing an existing dataset."""

    def __init__(self, output: str | Path, shard_size: int = 100_000) -> None:
        if shard_size <= 0:
            raise ValueError("shard_size must be positive")
        self.output = Path(output)
        if self.output.exists() and any(self.output.iterdir()):
            raise FileExistsError(f"refusing to overwrite non-empty dataset: {self.output}")
        self.output.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.buffers: dict[str, list[tuple[EncodedPosition, float]]] = {
            "train": [],
            "validation": [],
        }
        self.shard_counts = {"train": 0, "validation": 0}
        self.position_counts = {"train": 0, "validation": 0}

    def add(self, split: str, encoded: EncodedPosition, target_cp: float) -> None:
        if split not in self.buffers:
            raise ValueError(f"unknown split: {split}")
        buffer = self.buffers[split]
        buffer.append((encoded, target_cp))
        if len(buffer) >= self.shard_size:
            self._flush(split)

    def _flush(self, split: str) -> None:
        buffer = self.buffers[split]
        if not buffer:
            return
        index = self.shard_counts[split]
        destination = self.output / f"{split}-{index:05d}.npz"
        if destination.exists():
            raise FileExistsError(f"refusing to replace shard: {destination}")
        np.savez_compressed(
            destination,
            white=np.stack([item.white for item, _ in buffer]),
            black=np.stack([item.black for item, _ in buffer]),
            turn=np.asarray([item.turn for item, _ in buffer], dtype=np.int8),
            target_cp=np.asarray([target for _, target in buffer], dtype=np.float32),
        )
        self.position_counts[split] += len(buffer)
        self.shard_counts[split] += 1
        buffer.clear()

    def close(self, metadata: dict[str, object] | None = None) -> Path:
        for split in self.buffers:
            self._flush(split)
        manifest = {
            "format_version": 1,
            "positions": self.position_counts,
            "shards": self.shard_counts,
            **(metadata or {}),
        }
        destination = self.output / "manifest.json"
        if destination.exists():
            raise FileExistsError(f"refusing to replace manifest: {destination}")
        destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return destination


def shard_paths(dataset: str | Path, split: str) -> list[Path]:
    paths = sorted(Path(dataset).glob(f"{split}-*.npz"))
    if not paths:
        raise FileNotFoundError(f"no {split} shards found in {dataset}")
    return paths


def load_shard(path: str | Path) -> ShardBatch:
    with np.load(Path(path), allow_pickle=False) as archive:
        return ShardBatch(
            white=archive["white"].copy(),
            black=archive["black"].copy(),
            turn=archive["turn"].copy(),
            target_cp=archive["target_cp"].copy(),
        )


def batches(
    shard: ShardBatch,
    batch_size: int,
    rng: np.random.Generator | None = None,
) -> Iterator[ShardBatch]:
    indices = np.arange(len(shard.turn))
    if rng is not None:
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        yield ShardBatch(
            white=shard.white[selected],
            black=shard.black[selected],
            turn=shard.turn[selected],
            target_cp=shard.target_cp[selected],
        )
