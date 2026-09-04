# NNUE v3 integration screen — 4 September 2026

Decision: experimental only. Keep the submitted v5 classical agent active.

## Reproducible candidates

- Neural runtime: `challengers/nnue_v3`, isolated module names, no PyTorch import.
- User-provided v3 weights SHA256:
  `8113ebabcdabb1c1b5bfc94b00e48e872bf60ed479c1e0c310a57d3e857e162b`.
- Frozen champion: `candidate-fast-v5-ordering.zip`, SHA256:
  `9c61dee88ad864a7818a65fb3e7c8dd09e2ba7c6fd3294e80a63b318cc00d22c`.
- No persistent search memory, aspiration experiment, or root agent change included.

## Correctness

`python -m unittest discover -s tests -p test_nnue_search.py -v`
passes three tests: 128-ply deterministic legal play and full undo, explicit
castling/en-passant/capture-promotion cases, and search-board restoration after
completed and deadline-aborted search. Real-v3 accumulators and integer scores
match an independent Python-chess feature encoding plus NumPy arithmetic.
The test oracle must use `fen(en_passant="fen")` to preserve the same en-passant
state convention as the core; Python-chess's default omits unusable EP targets.

This is integer-runtime parity, NOT float checkpoint/export parity, and not
the 100,000-position promotion gate. The v3 float checkpoint was not supplied.

## Playing-strength screen

Command: `python -m tools.paired_arena --candidate challengers/nnue_v3
--champion candidate-fast-v5-ordering.zip --base-ms 1000 --increment-ms 50
--openings 2` (join onto one line).

Result: **0 wins, 0 draws, 4 losses**, all checkmate, no reported technical
terminations. Each of two openings used both colors. This small, short-clock
screen is not an Elo estimate or proof about longer competition clocks. It
does not justify a promotion.

A second screen at `--base-ms 10000 --increment-ms 100 --openings 1`
also lost both color-swapped games by checkmate, with no reported technical
terminations. Both screens together are six losses, not six independent
opening tests. These clocks are local diagnostics, not competition validation.
Full repository discovery also passed **45 tests** in 23.857 seconds.

## Speed and score sanity

One warmed starting-position depth-5 measurement on the local Mac:

| Search | Nodes | Elapsed | Nodes/second |
| --- | ---: | ---: | ---: |
| Current classical default | 25,946 | 13.09 ms | 1,981,581 |
| Isolated neural v3 | 37,719 | 38.06 ms | 991,101 |

Different evaluators search different trees. This single sample is diagnostic,
not a rigorous speed distribution or frozen-archive strength comparison.
Basic symmetric pawn fixtures with an extra queen score +709 cp for White's
extra queen and -644 cp for Black's extra queen (White to move in both).

## Follow-up priorities

1. Keep v5 active; no new upload based on this model.
2. Before requesting training, audit held-out errors separately for ordinary
   balanced positions and extreme/mate labels. Overall v3 MAE of 1045.5 cp
   does not establish useful accuracy on positions the search actually visits.
3. Measure a material-only baseline and consider a learned correction to a
   reliable material evaluator rather than replacing it outright. Test first.
4. Read-only specialist review found no concrete new accumulator/orientation
   defect. It flagged inherited quiescence-horizon mate handling and
   repetition-history-dependent transposition scores. These remain unfixed;
   build focused regression tests before changing search behavior.
5. Do not ask the user simply to rerun more epochs of the same recipe.
