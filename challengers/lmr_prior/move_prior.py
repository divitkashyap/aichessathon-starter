"""Use our trained network only to choose the first root move to investigate."""

import json
from pathlib import Path

import numpy as np
from lmr_core import SIDE, UNDO_SIZE, generate_legal_moves, make_move, unmake_move
from lmr_search import evaluate
from numba import njit
from prior_nnue import evaluate_accumulator_arrays, initialize_accumulators

with np.load(Path(__file__).with_name("nnue_v3_weights.npz"), allow_pickle=False) as archive:
    metadata = json.loads(str(archive["metadata"]))
    FEATURE = archive["feature"].copy()
    BIAS = archive["feature_bias"].copy()
    OUTPUT = archive["output_weight"].copy()
    TEMPO = int(archive["tempo"][0])
FEATURE_SCALE = int(metadata["feature_scale"])
WEIGHT_SCALE = int(metadata["weight_scale"])
OUTPUT_SCALE = int(metadata["output_scale_cp"])


@njit(cache=False)
def prior_move(board, state):
    """No neural scores enter the actual search or its cached bounds."""
    best_move = 0
    best_score = -2_000_000
    undo = np.empty(UNDO_SIZE, dtype=np.int64)
    for move in generate_legal_moves(board, state):
        make_move(board, state, int(move), undo)
        classical = evaluate(board, state)
        correction = 0
        if np.count_nonzero(board) >= 17:
            white, black, _ = initialize_accumulators(board, FEATURE, BIAS)
            neural = evaluate_accumulator_arrays(
                white,
                black,
                int(state[SIDE]),
                OUTPUT,
                TEMPO,
                OUTPUT_SCALE,
                FEATURE_SCALE,
                WEIGHT_SCALE,
            )
            correction = max(-200, min(200, neural - classical))
        # Child side-to-move is the opponent. Keep integer eighth-centipawn units
        # solely for ordering, without rounding a score into a search bound.
        score = -(8 * classical + correction)
        unmake_move(board, state, int(move), undo)
        if score > best_score:
            best_score = score
            best_move = int(move)
    return best_move
