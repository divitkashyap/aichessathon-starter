# Neural evaluator diagnosis — 4 September 2026

## Finding

V3's overall error hid severe errors on ordinary sparse positions. A lower
overall validation loss did not establish a better chess evaluator.

Diagnostic source: [Lichess evaluation dataset](https://huggingface.co/datasets/Lichess/chess-position-evaluations).
Sampled 20 public API pages of 100 rows, at offsets 50,000,000 then +23,000,000
per page. Discarded partial FEN groups at page boundaries, duplicate FENs,
depth <18, invalid and terminal boards. Retained 690 positions. Labels are
converted from White's perspective to side-to-move perspective.

This is **not certified held-out data**: the original training-position
manifest is unavailable here. Small adjacent groups may also be correlated.
These are static evaluation errors, not search accuracy or Elo measurements.

| Subset | Positions | Classical MAE | Neural v3 MAE |
| --- | ---: | ---: | ---: |
| All labels, including mapped mates | 690 | 1343 cp | 1188 cp |
| Ordinary: non-mate label, within ±300 cp | 452 | 126 cp | 353 cp |
| Ordinary, 17+ pieces | 371 | 125 cp | 133 cp |
| Ordinary, at most 8 pieces | 27 | 184 cp | 2973 cp |

Example: `7k/5Kn1/6P1/8/2B5/8/8/8 b - -` has label 0 at depth 54.
Classical evaluates -260 cp; v3 evaluates -12,798 cp. One pawn is roughly
100 cp. This is not a small rounding discrepancy. Exact integer accumulator
tests passed independently, so the observed issue lies beyond those tested
runtime updates; training coverage/objective/model capacity remain hypotheses.

## Controlled blend experiment

New isolated `challengers/nnue_blend` uses the existing classical evaluator
plus 1/8 of the neural disagreement, with disagreement capped at ±200 cp.
Thus the model can adjust the score by at most 25 cp. Below 17 occupied
squares, use classical evaluation alone. Runtime rounds symmetrically.
The exploratory audit initially used floor rounding (within 1 cp difference).
This was chosen from diagnostics, so any later dataset validation must be
independent of this selection. Actual paired games decide promotion.

First screen: 3 openings, both colors, 1000+50 ms against the frozen v5 ZIP
(`9c61dee88ad864a7818a65fb3e7c8dd09e2ba7c6fd3294e80a63b318cc00d22c`).
Result **2 wins, 2 draws, 2 losses (50%)**, no technical terminations. Game
records and source hashes are retained in `artifacts/nnue-blend-screen-1`.
No promotion. Formatting-only cleanup followed this screen.

The bounded correction reduced ordinary-position MAE in this diagnostic
sample by about 5 cp. That small gain may not compensate for slower search.
Do not promote without games against the exact frozen active-agent archive.

## Independent sample and follow-up screens

A second sample starts at row 500,000,000 and uses 20 pages spaced by
23,000,000 rows. Explicitly excluded all FENs in the original report. It
retains 850 positions, with no position overlap with the first sample;
overlap with the original training set remains unknown. Gate, weight and cap
were held fixed. The audit now uses the runtime's symmetric rounding.

| Ordinary subset | Positions | Classical MAE | Neural MAE | Guarded blend MAE |
| --- | ---: | ---: | ---: | ---: |
| All | 592 | 129.75 cp | 248.72 cp | 123.79 cp |
| 17+ pieces | 492 | 121.99 cp | 123.15 cp | 114.82 cp |
| At most 8 pieces | 21 | 334.10 cp | 2042.19 cp | 334.10 cp |

Report: `artifacts/nnue-audit-v3-independent-20260904.json`. These static
results replicate a small bounded benefit, not an improvement in playing
strength. A phase-switch variant scored 2 wins, 3 draws, 1 loss against v5
in six short-clock games. Adding the guarded blend to LMR then scored
**1 win, 2 draws, 3 losses against frozen v6**, at 1000+50 ms. No technical
terminations. Records: `artifacts/lmr-blend-screen-1`. **Not promoted.**

The next isolated experiment uses the network only to choose the first root
move investigated, retaining classical evaluation throughout the actual
search. This is a hypothesis, not an established improvement.

## Validate the actual Colab holdout before further training

After pulling this branch, run with the existing paths (no new GPU training):

```python
!python -m tools.validate_nnue --dataset "$DATASET" --weights "$WEIGHTS"
```

This uses validation shards only, prints errors for ordinary positions and
sparse endgames separately, and leaves all data/checkpoints untouched. Save
the printed report. No v3 `.pt` is required for this integer diagnostic.

Further training should target the measured weak slices, not simply add epochs
to optimize the same aggregate number. Potential next experiments are bounded
targets, phase-balanced sampling and a learned correction to material. These
are proposals, not established improvements or implemented training changes.

## Reproduce public diagnostic

```sh
python -m tools.audit_nnue --pages 20 --output artifacts/new-audit.json
```

Reports refuse to overwrite existing files. Raw sampled rows and predictions
are retained in the report. `--input` reuses a previous report without network.
The dataset API is used only by this offline tool, never by an agent.
