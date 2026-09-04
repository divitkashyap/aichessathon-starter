"""Report NNUE error by target range and piece count on existing validation shards."""

import argparse
import json
from pathlib import Path

import numpy as np

from nnue.features import PADDING_INDEX, white_cp_to_side_to_move
from nnue.quantize import load_quantized
from nnue.runtime import evaluate_quantized_arrays
from nnue.shards import load_shard, shard_paths


def summarize(target, prediction, pieces):
    result = {}
    for name, mask in {
        "all": np.ones(len(target), dtype=bool),
        "ordinary_abs_target_le_300": np.abs(target) <= 300,
        "ordinary_17plus_pieces": (np.abs(target) <= 300) & (pieces >= 17),
        "ordinary_8orless_pieces": (np.abs(target) <= 300) & (pieces <= 8),
        "extreme_abs_target_gt_1000": np.abs(target) > 1000,
    }.items():
        if not mask.any():
            continue
        error = np.abs(target[mask] - prediction[mask])
        result[name] = dict(
            count=int(mask.sum()),
            mae_cp=float(error.mean()),
            median_error_cp=float(np.median(error)),
            p90_error_cp=float(np.quantile(error, 0.9)),
            error_gt_1000_cp=int((error > 1000).sum()),
            zero_baseline_mae_cp=float(np.abs(target[mask]).mean()),
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("refusing to replace existing report")
    weights = load_quantized(args.weights)
    targets, predictions, counts = [], [], []
    for path in shard_paths(args.dataset, "validation"):
        shard = load_shard(path)
        target = white_cp_to_side_to_move(shard.target_cp, shard.turn)
        predicted = np.empty(len(target), dtype=np.float64)
        for index in range(len(target)):
            predicted[index] = evaluate_quantized_arrays(
                shard.white[index],
                shard.black[index],
                int(shard.turn[index]),
                weights.feature,
                weights.feature_bias,
                weights.output_weight,
                int(weights.tempo),
                weights.output_scale_cp,
                weights.feature_scale,
                weights.weight_scale,
            )
        targets.append(target)
        predictions.append(predicted)
        counts.append((shard.white != PADDING_INDEX).sum(axis=1) + 2)
        print(f"validated={path.name} positions={len(target)}", flush=True)
    result = summarize(np.concatenate(targets), np.concatenate(predictions), np.concatenate(counts))
    print(json.dumps(result, indent=2))
    if args.output is not None:
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2)


if __name__ == "__main__":
    main()
