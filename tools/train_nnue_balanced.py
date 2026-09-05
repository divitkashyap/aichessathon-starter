"""Controlled v4 pilot: capped targets and train-only phase balancing.

Architecture/export remain unchanged. Validation selects on balanced capped
error, while raw errors remain visible. No previous checkpoint is overwritten.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from nnue.features import PADDING_INDEX
from nnue.model import NNUEConfig, SparseNNUE, save_checkpoint
from nnue.shards import batches, load_shard, shard_paths
from tools.train_nnue import _tensor_batch
from tools.validate_nnue import summarize


def groups(white):
    pieces = (white != PADDING_INDEX).sum(axis=1) + 2
    return np.where(pieces <= 8, 0, np.where(pieces <= 16, 1, 2))


def balanced_weights(counts):
    if np.any(counts == 0):
        raise ValueError("All three piece-count groups must exist in training")
    # Bounded inverse frequency prevents a tiny group dominating gradients.
    return np.clip(counts.sum() / (3.0 * counts), 0.25, 4.0)


def objective(prediction, target, weight, cap):
    errors = F.smooth_l1_loss(prediction / 100, target.clamp(-cap, cap) / 100,
                              reduction="none")
    return (errors * weight).mean()


@torch.inference_mode()
def validation(model, paths, device, cap):
    model.eval()
    targets, predictions, pieces = [], [], []
    for path in paths:
        for batch in batches(load_shard(path), 4096):
            w, b, turn, target = _tensor_batch(batch, device)
            predictions.append(model(w, b, turn).cpu().numpy())
            targets.append(target.cpu().numpy())
            pieces.append((batch.white != PADDING_INDEX).sum(axis=1) + 2)
    t, p, n = map(np.concatenate, (targets, predictions, pieces))
    report = summarize(t, p, n)
    group_errors = []
    for name, mask in (("sparse", n <= 8), ("middle", (n > 8) & (n <= 16)),
                       ("dense", n > 16)):
        if not mask.any():
            raise ValueError("Missing validation group: " + name)
        error = float(np.abs(p[mask] - np.clip(t[mask], -cap, cap)).mean())
        report[name + "_capped_mae"] = error
        group_errors.append(error)
    report["selection_score"] = float(np.mean(group_errors))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--cap", type=float, default=1000)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    if args.epochs < 1 or args.cap <= 0:
        parser.error("epochs and cap must be positive")
    if args.run.exists():
        parser.error("Run exists; choose a NEW path. No overwrite or resume.")
    train = shard_paths(args.dataset, "train")
    val = shard_paths(args.dataset, "validation")
    if not train or not val:
        parser.error("Training and validation shards required")
    counts = np.zeros(3, dtype=np.int64)
    for path in train:
        counts += np.bincount(groups(load_shard(path).white), minlength=3)
    weights = balanced_weights(counts)
    args.run.mkdir(parents=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SparseNNUE(NNUEConfig(128)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    print(f"device={device} train_counts={counts.tolist()} weights={weights.tolist()}", flush=True)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(train)
        for path in train:
            for batch in batches(load_shard(path), 16384, rng):
                w, b, turn, target = _tensor_batch(batch, device)
                weight = torch.tensor(weights[groups(batch.white)], dtype=torch.float32, device=device)
                optimizer.zero_grad(set_to_none=True)
                loss = objective(model(w, b, turn), target, weight, args.cap)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 4)
                optimizer.step()
        report = validation(model, val, device, args.cap)
        report.update(epoch=epoch, cap=args.cap, seed=args.seed,
                      train_counts=counts.tolist(), train_weights=weights.tolist())
        with (args.run / f"epoch-{epoch:02d}.json").open("x") as stream:
            json.dump(report, stream, indent=2)
        print(json.dumps(report), flush=True)
        if report["selection_score"] < best:
            best = report["selection_score"]
            # Unique filename for every saved improvement, never overwrite.
            save_checkpoint(args.run / f"epoch-{epoch:02d}.pt", model, metadata=report)
    print("Finished. Select lowest selection_score among epoch JSONs for export.", flush=True)


if __name__ == "__main__":
    main()
