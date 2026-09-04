# Einsteinanium development roadmap

The competition build advances through a champion/challenger loop. A candidate is uploaded only
after it beats the current champion in paired games and passes correctness, clock and packaging
checks. The latest known-good zip is always retained outside the submission tree.

## 4 September: safe classical search

- Iterative-deepening principal-variation search, alpha-beta and quiescence.
- Transposition table, capture ordering, killer moves and history ordering.
- Tapered positional evaluation and hard clock interruption.
- Legal fallback, edge-case tests and repeatable candidate gate.

Exit gate: no technical failures; beat every starter baseline at a short clock.

## 5 September: compiled board and search core

- Add an integer move encoding and compact bitboard position.
- Implement Numba move generation, make/unmake and Zobrist hashing.
- Compare legal move sets with python-chess and add standard perft positions.
- Port search and evaluation only after move generation is exact.

Exit gate: zero perft/move-set mismatches and a clear node-throughput gain.

## 6 September: search strength and data pipeline

- Add late-move reductions, null-move pruning, aspiration windows and static exchange evaluation
  one at a time.
- Build a diverse, deduplicated sample of engine-evaluated positions.
- Fit the handcrafted evaluation weights before training a neural evaluator.

Exit gate: each search feature wins a champion/challenger match; dataset orientation is verified.

## 7 September: compact NNUE

- Train sparse king-bucket value models on Colab.
- Quantise the best model and verify training/runtime numerical parity.
- Add incremental accumulators, with refreshes after king-bucket changes.

Exit gate: the NNUE engine beats the tuned classical champion without losing effective depth.

## 8 September: learned ordering

- Train a compact policy head from principal-variation moves.
- Apply it at the root and shallow nodes only.
- Measure whether improved ordering offsets inference cost.

Exit gate: higher game score and equal-or-better completed search depth.

## 9 September: pondering and hard-position mining

- Search predicted opponent replies between calls and reuse transpositions on a match.
- Mine blunders and engine disagreements from tournament games.
- Fine-tune only if the hard-position set exposes a consistent model weakness.

Exit gate: no race, clock or state-corruption failures over a long stress run.

## 10 September: final freeze

- Run the largest paired tournament time allows.
- Test real clocks, read-only filesystem behavior, import time, memory and package size.
- Upload the strongest stable champion and retain a rollback zip.

## 11 September: final morning

- Make only fixes for demonstrated failures.
- Submit before 11:00 Europe/London with a large safety margin.

## Daily upload policy

The live dashboard currently permits ten uploads per team per day. Upload one stable champion each day and use
additional slots only for a candidate with positive local evidence or a platform-specific fix.
Never spend a slot merely to test a hypothesis that the local harness can answer.

## Measured progress

- V1 classical champion: valid and active on 4 September.
- Fast-core correctness: exact start-position depth-5 and Kiwipete depth-4 perft;
  100,000 differential positions matched `python-chess`, including exact make/unmake state.
- Fast-search V2: 1.26--1.54 million nodes/second at start-position depths 4--6 and a
  72.9% score (`+15 =5 -4`) against frozen V1 over 24 color-swapped games at 2s+0.1s.
- Known V2 weakness: all five draws were threefold repetitions. Add history-aware repetition
  scoring and rerun the same suite before promoting another challenger.
- Platform v2 failed before startup because its local-only wrapper assumed a deeper filesystem.
  The corrected archive was submitted as v3: valid and active on 4 September, initialized in
  14.6--20.8 seconds of the 90-second allowance, and won both organiser smoke games by checkmate.
- History-aware repetition challenger: correctly detects third occurrences but scored only
  52.1% (`+9 =7 -8`) against frozen v3 over 24 paired games. Retained as correctness
  infrastructure; rejected as a standalone strength promotion.
- Fixed-size transposition-table challenger: reduced fixed-depth nodes by 7% at depth 4,
  17% at depth 5 and 36% at depth 6, then scored 62.5% (`+9 =12 -3`) against frozen v3
  over 24 paired games with zero technical failures. Cleared the local promotion gate.
