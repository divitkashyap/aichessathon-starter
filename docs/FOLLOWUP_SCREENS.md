# Follow-up screens after v6 — 4 September 2026

These are isolated experiments, not active submissions. All short screens use
three openings with colors swapped. Six games are a rejection screen, not a
reliable strength estimate. No technical terminations occurred in the completed
screens below. Full records and source hashes remain under `artifacts/` locally.

| Candidate | Opponent | Clock (ms) | W/D/L | Decision |
| --- | --- | --- | --- | --- |
| `nnue_blend_switch` | frozen v5 | 1000+50 | 2/3/1 | Inconclusive; test with LMR |
| `lmr_blend` | frozen v6 | 1000+50 | 1/2/3 | Do not promote |
| `lmr_pawns` | frozen v6 | 1000+50 | 2/1/3 | Do not promote |
| `lmr_null` | frozen terminal-fix challenger | 1000+50 | 1/2/3 | Do not promote |
| `lmr_prior` | frozen v6 | 2000+100 | 1/3/2 | Do not promote |

`nnue_blend_switch` avoids maintaining neural state once the root has fewer
than 17 pieces. `lmr_blend` combines that switch and bounded neural correction
with v6's reduced search of late quiet moves. `lmr_pawns` adds modest passed,
isolated and doubled pawn terms. `lmr_null` tests a guarded hypothetical pass
to reject unpromising branches; it is disabled in sparse positions. None of
these tests justifies replacing v6.

Correctness checks passed independently: switch 3 tests, LMR blend 3 tests,
pawn candidate 5 tests plus a 512-ply independent numeric reference, null
candidate 4 tests. A pawn-perspective error was found and corrected before
its game screen. Correctness tests alone do not establish playing strength.

## Neural root move-ordering experiment

`lmr_prior` asks the bounded neural/classical combination only which legal
root move to investigate first. Actual search evaluation remains classical.
The root preference is stored in a fresh depth-zero cache entry, which cannot
supply a positive-depth score cutoff. No cache persists between moves.

Below 17 pieces or below 1000 ms remaining, skip the neural work. Its elapsed
time and fresh cache allocation are deducted from the remaining game clock.
Warmed prior evaluation across 12 opening positions, repeated ten times,
measured median 0.044 ms, maximum 0.056 ms on this Mac. This is not a platform
timing guarantee. Both search and prior compilation happen at initialization.

Five tests pass: 96-ply legal-choice/restoration differential test, empty
terminal preference, unchanged depth-two full-width scores with deliberately
poisoned dummy cache scores, fresh-memory/budget accounting, low-clock bypass.
A second reviewer independently found no sign or cache-bound corruption;
the prior has no internal deadline, so the conservative low-clock gate remains
important. The six-game screen against frozen v6 at 2000+100 ms finished
**1 win, 3 draws, 2 losses (41.7%)**, no technical terminations. Records:
`artifacts/lmr-prior-screen-1`. This gives no evidence of an improvement;
the candidate is not promoted. Tests were expanded during the screen without
changing any agent runtime file after the match started.
