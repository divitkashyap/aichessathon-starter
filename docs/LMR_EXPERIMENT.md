# Conservative late-move reductions — 4 September 2026

Isolated challenger: `challengers/lmr`. The submitted v5 archive remains
unchanged. This experiment changes search allocation, not evaluation/training.

After the first four ordered moves, a quiet nonpromotion move can receive a
one-ply shallower initial search. Only at non-root depth >=3, neither side in
check, and excluding known cutoff moves and the table's preferred move.
If the reduced probe beats the current best score, verify it at full depth
before applying the existing full-window re-search. Timeout always unwinds
the move. No persistent between-move table reuse is enabled.

## Correctness and depth screen

Legal-move, mate-in-one and timeout restoration tests passed. Both known
round-five regression positions also pass at depth five: avoid `f6e4`, find
`d2g5` with mate score 999995.

Across the existing 12 opening FENs:

| Depth | Classical nodes | LMR nodes | Reduction | Same best move |
| --- | ---: | ---: | ---: | ---: |
| 4 | 304343 | 193850 | 36.3% | 10/12 |
| 5 | 1688240 | 744954 | 55.9% | 9/12 |

Selective search does not guarantee identical scores/moves to full-depth
search. Node reduction alone does not establish stronger play.

## Games

Opponent: frozen `candidate-fast-v5-ordering.zip`, SHA256
`9c61dee88ad864a7818a65fb3e7c8dd09e2ba7c6fd3294e80a63b318cc00d22c`.

Initial screen: first 3 openings, both colors, 1000+50 ms.
**5 wins, 0 draws, 1 loss (83.3%)**, no technical terminations.
Records: `artifacts/lmr-screen-1` (PGNs, source hashes, result).

Validation: remaining 9 openings, both colors, 2000+100 ms.
**9 wins, 5 draws, 4 losses (63.9%)**, no technical terminations.
Records: `artifacts/lmr-validation-1`.
Combined with screening: **14 wins, 5 draws, 5 losses (68.75%)** over 24 games
at two clock settings. The separate validation score is more informative than
the combined score because selection followed the initial screen. Neither is
a reliable Elo estimate or proof against leaderboard opponents.

Packaged archive `candidate-lmr-v6.zip`, 49,096 bytes uncompressed, SHA256
`3738ae751e65a397c410a029f56f782f816407142564cc82527834eec05e8e98`.
Only `agent.py`, `lmr_core.py`, `lmr_search.py` are shipped; no neural weights
or offline datasets. Full-clock 120000+500 ms paired test against the exact v5
archive finished **1 win, 1 draw, 0 losses**, no technical terminations.
Records: `artifacts/lmr-full-clock-1`. Two games are a clock/reliability check,
not a precise strength estimate.

After explicit user confirmation of upload and rules acceptance, this exact
archive was uploaded as platform v6 at 22:22 London on 4 September. Dashboard
confirmed hash prefix `3738ae751e65`, sixth upload of the day. Platform
validation finished at 21:24:30 UTC: **valid and active**. Both initialization
times were 24.3 seconds of the 90-second allowance; smoke outcomes were White
by adjudication and Black by checkmate. A round already in play retained v5.

The inherited quiescence terminal-horizon and history-dependent table concerns
remain outside this controlled change. Those require separate regression fixes.
