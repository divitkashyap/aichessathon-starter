"""Export and verify an integer NNUE artifact for the competition agent."""

from __future__ import annotations

import argparse
from pathlib import Path

import chess
import torch

from nnue.features import encode_board
from nnue.model import load_checkpoint
from nnue.quantize import quantize, save_quantized
from nnue.runtime import evaluate_quantized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-error-cp", type=float, default=8.0)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise SystemExit(f"refusing to overwrite weights: {arguments.output}")

    model = load_checkpoint(arguments.checkpoint).eval()
    weights = quantize(model)
    positions = [
        chess.Board(),
        chess.Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
        chess.Board("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
    ]
    maximum_error = 0.0
    with torch.inference_mode():
        for board in positions:
            encoded = encode_board(board)
            white = torch.from_numpy(encoded.white).unsqueeze(0)
            black = torch.from_numpy(encoded.black).unsqueeze(0)
            turn = torch.tensor([encoded.turn], dtype=torch.int8)
            expected = float(model(white, black, turn)[0])
            actual = evaluate_quantized(encoded.white, encoded.black, int(encoded.turn), weights)
            maximum_error = max(maximum_error, abs(expected - actual))
    if maximum_error > arguments.max_error_cp:
        raise SystemExit(f"quantization parity failed: maximum error {maximum_error:.2f} cp")
    save_quantized(arguments.output, weights)
    print(f"saved={arguments.output} max_error_cp={maximum_error:.2f}")


if __name__ == "__main__":
    main()
