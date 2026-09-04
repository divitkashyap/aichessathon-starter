"""Train Einsteinanium's sparse value network from encoded NPZ shards."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as functional

from nnue.model import NNUEConfig, SparseNNUE, save_checkpoint
from nnue.shards import ShardBatch, batches, load_shard, shard_paths


def _tensor_batch(batch: ShardBatch, device: torch.device) -> tuple[Tensor, ...]:
    # Lichess cloud-evaluation scores use the FEN's side-to-move perspective,
    # which is also the perspective returned by SparseNNUE.  Flipping Black's
    # rows here would invert half the training set.
    return (
        torch.from_numpy(batch.white).to(device),
        torch.from_numpy(batch.black).to(device),
        torch.from_numpy(batch.turn).to(device),
        torch.from_numpy(batch.target_cp).to(device),
    )


def _loss(predicted_cp: Tensor, target_cp: Tensor) -> Tensor:
    # Training in pawn units keeps Huber's transition point interpretable.
    return functional.smooth_l1_loss(predicted_cp / 100.0, target_cp / 100.0, beta=1.0)


@torch.inference_mode()
def validate(
    model: SparseNNUE,
    paths: list[Path],
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_absolute_error = 0.0
    count = 0
    for path in paths:
        for batch in batches(load_shard(path), batch_size):
            white, black, turn, target = _tensor_batch(batch, device)
            predicted = model(white, black, turn)
            size = len(target)
            total_loss += float(_loss(predicted, target)) * size
            total_absolute_error += float(torch.abs(predicted - target).sum())
            count += size
    return total_loss / count, total_absolute_error / count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16_384)
    parser.add_argument("--feature-hidden", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260904)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise SystemExit(f"refusing to overwrite checkpoint: {arguments.output}")

    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    train_paths = shard_paths(arguments.dataset, "train")
    validation_paths = shard_paths(arguments.dataset, "validation")
    model = SparseNNUE(NNUEConfig(arguments.feature_hidden)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.learning_rate)
    rng = np.random.default_rng(arguments.seed)
    best_loss = math.inf

    for epoch in range(1, arguments.epochs + 1):
        model.train()
        rng.shuffle(train_paths)
        running_loss = 0.0
        seen = 0
        for path in train_paths:
            for batch in batches(load_shard(path), arguments.batch_size, rng):
                white, black, turn, target = _tensor_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                predicted = model(white, black, turn)
                loss = _loss(predicted, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
                optimizer.step()
                size = len(target)
                running_loss += float(loss.detach()) * size
                seen += size

        validation_loss, validation_mae = validate(
            model, validation_paths, arguments.batch_size, device
        )
        print(
            f"epoch={epoch} train_loss={running_loss / seen:.5f} "
            f"validation_loss={validation_loss:.5f} validation_mae_cp={validation_mae:.1f}"
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            save_checkpoint(
                arguments.output,
                model,
                optimizer=optimizer,
                metadata={
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                    "validation_mae_cp": validation_mae,
                    "seed": arguments.seed,
                },
            )
            print(f"saved={arguments.output}")


if __name__ == "__main__":
    main()
