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
  over 24 paired games with zero technical failures. Submitted as v4; valid and active on
  4 September after initializing in 28.8--29.8 seconds and passing both platform smoke games.
- Principal-variation search challenger preserved every move and score over the 12-opening
  fixed-depth suite while reducing nodes by 35% at depth 4 (1,090,928 to 705,590) and 43% at
  depth 5 (8,377,767 to 4,795,164). It scored 62.5% in a 12-game screen, then 52.1%
  (`+10 =5 -9`) in the 24-game gate against frozen v4, with zero technical failures. The
  full result is too close to distinguish from match noise, so this was not uploaded alone.
- Legal generation now finds the moving side's king once per position rather than once per
  candidate move. The 12-opening depth-5 benchmark kept the same 4,795,164 nodes and improved
  median throughput from about 670,000 to 725,000 nodes/second (8%); 100,000 differential
  positions matched `python-chess`, including 273 castles, 180 en-passant captures and 899
  promotions.
- Quiet-move killer/history ordering preserved every move and score over the 12-opening suite.
  Together with tactical-only quiescence ordering it reduced depth-4 nodes from 705,590 to
  304,343 (57%) and depth-5 nodes from 4,795,164 to 1,688,240 (65%). It still requires paired
  games against frozen v4 before promotion.
- Rated round 5 was lost by checkmate after `16...Ne4 17.Bxe4 gxf3`. Replay matched all Black
  moves with the older Python V1 engine, while compiled V4 rejects `16...Ne4` from depth 4 and
  sees the forced mate after `17...gxf3` at depth 5. This is evidence that the game was probably
  queued on an older build snapshot, not proof that V4 made the blunder. The positions are now
  permanent regressions and future archives print an explicit build identifier during validation.
- NNUE v1 pipeline: 32 horizontally normalized king buckets, ten relative non-king piece planes,
  shared 128-wide accumulators, antisymmetric direct head, deterministic Lichess shards and integer
  export. It remains an untrained challenger until Colab training, parity, speed and paired-game
  gates pass.
- NNUE v2 trained for six epochs on two million positions and exported with 1.87 cp parity on the
  export fixtures. An initial sanity check compared its side-to-move output against a mistakenly
  White-relative classical score and falsely rejected it. Corrected testing over 20,000 legal
  positions found 0.560 correlation and 86.0% sign agreement outside a one-pawn margin. Keep v2 as
  the trained challenger and require broad parity, speed and paired-game gates before promotion.
- Incremental NNUE accumulators exactly matched full reconstruction through a 192-ply synthetic
  make/unmake test and a 256-ply test with the real v2 weights, including captures, en passant,
  promotions, castling and king-bucket refreshes. Reusing the accumulator made evaluation alone
  about 3.0 times faster on the local benchmark; search integration remains a separate gate.
- Rated round 6 drew as Black against Brokefish by threefold repetition with no technical failure.
  Eight of nine sampled game moves matched compiled V4 at fixed depth, strong evidence that V4 was
  the playing build. It defended a rook ending one pawn down and forced the draw while retaining
  more than 21 seconds; keep the game for future evaluator and repetition-policy tests.
