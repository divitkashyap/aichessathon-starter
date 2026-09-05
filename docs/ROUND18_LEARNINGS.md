# Rated 18 win: conversion, diagnostic correction, and broader v8 tests

Source: our signed-in dashboard's rated round 18 PGN, against team
"How does the knight move?" (the game page displays bot "Friday"). We played
Black and won by checkmate on move 60. Local source:
`artifacts/rated-day2/round18.pgn`. The initial FEN is
`r1bq1rk1/pp2ppbp/2np1np1/2p5/4P2P/2NP2P1/PPP1NPB1/R1BQ1RK1 b - - 4 8`.

V8 was active when the result was inspected, but the PGN leaves build metadata
unknown. Without the rated agent log, do not attribute this game's decisions
to a verified archive or claim that the speed change caused the win.

## What the moves actually show

- Queens came off on move 25; `27...Rxc2` then captured the c2 pawn.
- The king became active through `...Ke5`, `...Kf4`, and `...Kg3`.
- After rook and bishop exchanges, the f-pawn advanced and promoted on move 55.
- `57...axb3+` was a legal en-passant capture. The game ended with `60...Qc1#`,
  with 28.057 seconds on our clock after the move.
- We also played `11...g5` in this win. The previous losses do not justify a
  blanket rule against kingside pawn pushes. Context matters; yesterday's
  isolated pawn-cover reward experiment did not earn promotion in its games.

These describe the recorded game, not a proof that every move was optimal.
The organizer's aggregate review reports 99.3% accuracy and ACPL 2 for Black,
but no independent per-move annotations were obtained. Do not turn that single
game statistic into an Elo estimate or an opponent-implementation claim.

The recorded continuation from move 50 through checkmate is preserved in
`tests/round18_conversion.json`. A test verifies every move, promotion,
en-passant and the final checkmate. This is an own rated game, not Daily Five.

## The win caught a review-tool problem

Before `53...f2+`:
`8/8/8/1p6/p2p4/P2P1pk1/1P6/4K3 b - - 1 53`.

The old review compared a root search with a fresh search rooted at the child.
That fresh child root does not have the same check-extension handling as an
interior node reached by the parent. Selective search and cache effects also
mean that independently searched scores need not agree. The old method could
therefore report a large apparent loss even for the engine's preferred move.

Measured using frozen v8 source (all scores from Black's perspective):

| Depth | Preferred move | Root score | Old played-move score | Interior played-move score |
| --- | --- | ---: | ---: | ---: |
| 5 | Kg2 | 920 | 440 | 424 |
| 7 | f2+ | 1058 | 438 | 1058 |
| 9 | f2+ | 1117 | 1117 | 1117 |

At depth seven the corrected comparison removes a **620 cp false warning**
for the selected move. At depth five, however, the shallow search still prefers
another move: matching search semantics does not eliminate the search horizon
or make this engine an oracle. Do not classify all large proxies as blunders.

`tools.review_pgn` now allows explicit `--engine terminal|v6|v7|v8`, records
the selected search/core hashes and review-tool hash, and evaluates the played
move through that engine's interior search at ply one. Playing-build identity
remains a separate explicit field. Custom test searches retain their documented
fresh-child behavior. Prior game repetition history is still not imported.

New tests verify engine attribution, illegal forced moves, mate-distance
counting, retained check extensions, and this depth-seven false-positive case.
Existing reports are preserved, not overwritten:
`artifacts/rated-day2/round18-v8-review-d5.json` and
`artifacts/rated-day2/round18-v8-forced-review-d5.json`.

## Reserved opening tests: no new candidate selected from their results

Added twelve `holdout` opening lines before running these games. They were
chosen for legal variety, not engine scores or certified equality. Position
identities (ignoring move counters) are distinct from all legacy/validation
starting positions. Some opening families are related, so this is not a claim
of statistically independent chess distributions. Once used for tuning, this
suite will no longer be an untouched holdout.

All games use the already-active, unchanged v8 archive
`c3861b36f16e72db12425a4ef4d6c11f9f09a4b485681a1c8bddd5026265facf`.
Clock: 10000+100 ms. Both colors per opening; local log instrumentation enabled
symmetrically. The two matches ran concurrently on this Mac, not the platform.

| Frozen opponent | Opening range | Wins | Draws | Losses | Score |
| --- | --- | ---: | ---: | ---: | ---: |
| v6 (`3738ae751e65`) | holdout 1–6 | 7 | 3 | 2 | 70.8% |
| v7 (`2274f766f415`) | holdout 7–12 | 6 | 4 | 2 | 66.7% |

No technical failures. Exact FENs, archive hashes and game records remain in
`artifacts/v8-holdout-v6-1` and `artifacts/v8-holdout-v7-1`. These are different
opening ranges, so the percentages cannot rank v6 against v7 directly. Both
opponents are from our own engine family; this is broader internal evidence,
not proof of superiority over other teams or the leading engines.

Outcome: keep v8 active. No upload, runtime agent modification, or retraining
was performed in this follow-up. Next improvements should be driven by deeper
verified position errors and representative evaluation data, not this win's
result alone or the old shallow review warnings.
