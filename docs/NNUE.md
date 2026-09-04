# Einsteinanium NNUE pipeline

This is a challenger pipeline, not yet the active competition evaluator.  V4 stays as the
rollback champion until trained weights pass parity, speed and paired-game gates.

## Architecture

- Two perspective-relative sparse encodings share one feature table.
- The friendly king selects one of 32 horizontally normalized buckets.
- Ten piece planes represent friendly/enemy pawn, knight, bishop, rook and queen locations.
- Both kings are represented by the king buckets rather than duplicated as piece features.
- A 128-wide clipped accumulator feeds a shared antisymmetric linear head:
  `tempo + head(side-to-move) - head(opponent)`.
- The dimensionless head is scaled by 400 to produce search-friendly centipawn values without
  forcing training to grow unnecessarily large output weights.
- Training uses PyTorch; deployment uses integer NumPy weights and Numba inference.

At 128 accumulator channels, the quantized feature table is about 5 MiB.  It remains comfortably
inside the 50 MB uncompressed submission cap while leaving space for code and later additions.

## Colab run

The checked-in notebook performs the complete first experiment.  Its commands are equivalent to:

```bash
python -m tools.stream_lichess_nnue \
  --output /content/nnue-data-v1 \
  --positions 2000000 \
  --minimum-depth 18

python -m tools.train_nnue \
  --dataset /content/nnue-data-v1 \
  --output /content/einsteinanium-nnue-v1.pt \
  --epochs 6 \
  --batch-size 16384

python -m tools.export_nnue \
  /content/einsteinanium-nnue-v1.pt \
  /content/einsteinanium-nnue-v1.npz
```

The Lichess source is de-normalized: the same FEN can appear for multiple principal variations and
search depths.  The builder deliberately keeps the first occurrence, which is the source's
highest-priority evaluation, and deduplicates by the four position-defining FEN fields.  Labels are
stored from White's point of view and oriented to side-to-move only when training.

## Promotion gate

1. Python and Numba encoders agree exactly on at least 100,000 positions.
2. Python and Numba integer inference agree exactly on at least 100,000 positions.
3. Incremental accumulators agree with full refresh after captures, en passant, promotions,
   castling and king moves.
4. Measure nodes/second before and after NNUE; reject a candidate that loses a search ply.
5. Play at least 100 color-swapped games against frozen V4 with zero technical failures.
6. Package the weights explicitly and validate cold initialization below 45 seconds.

Do not upload a model merely because validation loss improves.  Game score at the competition
clock is the promotion criterion.
