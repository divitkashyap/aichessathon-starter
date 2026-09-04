"""Reference integer NNUE inference suitable for compilation with Numba."""

from __future__ import annotations

import numpy as np
from numba import njit

from nnue.features import PADDING_INDEX
from nnue.quantize import QuantizedNNUE


def evaluate_quantized_reference(
    white: np.ndarray,
    black: np.ndarray,
    turn: int,
    weights: QuantizedNNUE,
) -> int:
    """Straight NumPy oracle for the compiled integer implementation."""
    white_active = white[white != PADDING_INDEX]
    black_active = black[black != PADDING_INDEX]
    white_accumulator = weights.feature_bias.astype(np.int64) + weights.feature[
        white_active
    ].sum(axis=0, dtype=np.int64)
    black_accumulator = weights.feature_bias.astype(np.int64) + weights.feature[
        black_active
    ].sum(axis=0, dtype=np.int64)
    us = white_accumulator if turn > 0 else black_accumulator
    them = black_accumulator if turn > 0 else white_accumulator
    difference = np.clip(us, 0, weights.feature_scale) - np.clip(
        them, 0, weights.feature_scale
    )
    total = int(weights.tempo) + int(
        np.dot(weights.output_weight.astype(np.int64), difference)
    )
    denominator = weights.feature_scale * weights.weight_scale
    if total >= 0:
        return (total + denominator // 2) // denominator
    return -((-total + denominator // 2) // denominator)


@njit(cache=False)
def evaluate_quantized_arrays(
    white: np.ndarray,
    black: np.ndarray,
    turn: int,
    feature: np.ndarray,
    feature_bias: np.ndarray,
    output_weight: np.ndarray,
    tempo: int,
    feature_scale: int,
    weight_scale: int,
) -> int:
    feature_hidden = feature.shape[1]
    white_accumulator = feature_bias.astype(np.int64)
    black_accumulator = feature_bias.astype(np.int64)
    for index in white:
        if index != PADDING_INDEX:
            for channel in range(feature_hidden):
                white_accumulator[channel] += feature[index, channel]
    for index in black:
        if index != PADDING_INDEX:
            for channel in range(feature_hidden):
                black_accumulator[channel] += feature[index, channel]

    us = white_accumulator if turn > 0 else black_accumulator
    them = black_accumulator if turn > 0 else white_accumulator
    total = np.int64(tempo)
    for channel in range(feature_hidden):
        us_value = min(feature_scale, max(0, us[channel]))
        them_value = min(feature_scale, max(0, them[channel]))
        total += np.int64(output_weight[channel]) * (us_value - them_value)
    denominator = feature_scale * weight_scale
    if total >= 0:
        return int((total + denominator // 2) // denominator)
    return -int((-total + denominator // 2) // denominator)


def evaluate_quantized(
    white: np.ndarray,
    black: np.ndarray,
    turn: int,
    weights: QuantizedNNUE,
) -> int:
    return evaluate_quantized_arrays(
        white,
        black,
        turn,
        weights.feature,
        weights.feature_bias,
        weights.output_weight,
        int(weights.tempo),
        weights.feature_scale,
        weights.weight_scale,
    )
