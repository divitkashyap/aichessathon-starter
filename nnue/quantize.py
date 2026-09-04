"""Deterministic integer export for the trained sparse evaluator."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from nnue.features import INPUT_FEATURES
from nnue.model import SparseNNUE

FEATURE_SCALE: Final = 256
WEIGHT_SCALE: Final = 128
FORMAT_VERSION: Final = 1


@dataclass(frozen=True, slots=True)
class QuantizedNNUE:
    feature: np.ndarray
    feature_bias: np.ndarray
    output_weight: np.ndarray
    tempo: np.int64
    output_scale_cp: int
    feature_scale: int = FEATURE_SCALE
    weight_scale: int = WEIGHT_SCALE


def _round_clip(value: np.ndarray, scale: int, dtype: np.dtype[np.signedinteger]) -> np.ndarray:
    limits = np.iinfo(dtype)
    return np.clip(np.rint(value * scale), limits.min, limits.max).astype(dtype)


def quantize(model: SparseNNUE) -> QuantizedNNUE:
    """Quantize a float model with accumulator scales preserved in biases."""
    state = model.to("cpu").eval().state_dict()
    feature = state["feature.weight"][:INPUT_FEATURES].numpy()
    feature_bias = state["feature_bias"].numpy()
    output_weight = state["output.weight"].numpy().reshape(-1)
    tempo = state["tempo"].numpy().reshape(-1)[0]
    return QuantizedNNUE(
        feature=_round_clip(feature, FEATURE_SCALE, np.int16),
        feature_bias=_round_clip(feature_bias, FEATURE_SCALE, np.int32),
        output_weight=_round_clip(output_weight, WEIGHT_SCALE, np.int16),
        tempo=np.int64(round(float(tempo) * FEATURE_SCALE * WEIGHT_SCALE)),
        output_scale_cp=model.config.output_scale_cp,
    )


def save_quantized(path: str | Path, weights: QuantizedNNUE) -> None:
    metadata = {
        "format_version": FORMAT_VERSION,
        "feature_scale": weights.feature_scale,
        "weight_scale": weights.weight_scale,
        "input_features": INPUT_FEATURES,
        "feature_hidden": int(weights.feature.shape[1]),
        "head": "antisymmetric-linear",
        "output_scale_cp": weights.output_scale_cp,
    }
    np.savez_compressed(
        Path(path),
        feature=weights.feature,
        feature_bias=weights.feature_bias,
        output_weight=weights.output_weight,
        tempo=np.asarray([weights.tempo], dtype=np.int64),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def load_quantized(path: str | Path) -> QuantizedNNUE:
    with np.load(Path(path), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        if metadata["format_version"] != FORMAT_VERSION:
            raise ValueError(f"unsupported NNUE format {metadata['format_version']}")
        if metadata["input_features"] != INPUT_FEATURES:
            raise ValueError("NNUE feature schema does not match this engine")
        return QuantizedNNUE(
            feature=archive["feature"].copy(),
            feature_bias=archive["feature_bias"].copy(),
            output_weight=archive["output_weight"].copy(),
            tempo=np.int64(archive["tempo"][0]),
            output_scale_cp=int(metadata["output_scale_cp"]),
            feature_scale=int(metadata["feature_scale"]),
            weight_scale=int(metadata["weight_scale"]),
        )
