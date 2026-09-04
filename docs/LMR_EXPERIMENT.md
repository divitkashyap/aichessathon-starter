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

Validation underway separately: remaining 9 openings, both colors, 2000+100 ms.
Records: `artifacts/lmr-validation-1`. Do not treat the initial six games as an
Elo estimate, a final validation result, or proof against leaderboard opponents.

The inherited quiescence terminal-horizon and history-dependent table concerns
remain outside this controlled change. Those require separate regression fixes.
