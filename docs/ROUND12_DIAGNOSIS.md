# Round 12 diagnosis and targeted experiments

Source: the signed-in dashboard's rated round 12 PGN, downloaded 4 September
2026. Our team played Black against OMI and lost by checkmate. The PGN does
not identify the engine build; do not attribute it to v6 without its game log.
The game is retained locally in `artifacts/rated-day1/` alongside rounds 13
and 15. It is not a Daily Five puzzle.

## Finding

Black played `22...Bd7` with **82.636 seconds** remaining, followed by:

`23.Nf6+ Kh8 24.Bg7+ Kxg7 25.Qg5+ Kh8 26.Qh6+ Rh7 27.Qxh7#`.

This is not a clock-forfeit failure. The before-move position is:

```text
r1b3k1/ppp2r2/1b1p3B/3N1p2/q1P5/5P2/PP1Q2PP/4RR1K b - - 4 22
```

An offline depth-five review using our terminal-fix engine assigns `...Bd7`
the largest loss proxy in this game (133 cp), preferring `...Qc4`. These
scores are **our own engine's estimates, not an independent oracle**. Root
and child searches differ in ordering/selective search, so even the same
recommended/played move can yield a nonzero proxy. The tool labels this.

More importantly, after `...Bd7`, v6 and the terminal-fix variant both choose
`b2b3` at depths five and six and do not recognize the mating sequence.
The shelter-only candidate also misses it at these depths. The tactical
failure is therefore reproducible in the current engine, independent of
which older build actually played this game.

## Separate hypotheses

1. `lmr_shelter`: a small middlegame-only penalty for missing pawn cover in
   front of a castled king when the opponent retains a queen. Maximum 60 cp
   before phase scaling. Isolated against the frozen terminal-fix variant;
   no changes to its search. Four tests pass. Initial 6-game screen at
   2000+100 ms: **3 wins, 2 draws, 1 loss**, no technical terminations.
2. `lmr_checks`: extend positive-depth checked positions by one ply, at most
   four times along a search line. Remaining extension budget is included in
   cache keys, and each recursive branch receives its own counter value.
   Actual depth-zero leaves still enter the existing quiescence search.
   This candidate finds `Nf6+` and score **mate in nine plies at depth four**.
   The displayed nine-ply mating line is verified legal through checkmate.
   Five focused tests pass, including timeout restoration and the exhausted
   extension-budget comparison against the reference engine.

Solving one tactical position is not proof of stronger general play. Both
candidates must pass new-opening matches and longer-clock checks before any
promotion. Coefficients and extension limits are held fixed during validation.

The shelter candidate completed the separate 24-game new-opening match versus
the terminal reference at 2000+100 ms: **9 wins, 8 draws, 7 losses (54.2%)**,
no technical terminations. This modest margin does not establish a reliable
strength improvement; it is not included in the check-extension candidate.

The check-extension candidate's initial 2000+100 ms screen finished **2 wins,
1 draw, 3 losses** against the terminal reference, no technical terminations.
At fixed depth four across 12 legacy openings it used 223,236 nodes versus
193,850 (15.2% more), retaining all 12 best moves. Warmed local elapsed times
were approximately 381 versus 343 ms while other tests were running; this is
not a clean hardware benchmark. It is undergoing a longer-clock comparison
with unchanged parameters, not being promoted on the tactical test alone.

That longer-clock comparison finished **6 wins, 3 draws, 3 losses (62.5%)**
against the terminal reference, on the first six validation openings with
colors swapped at 10000+100 ms. No technical terminations. Records:
`artifacts/checks-longer-unseen-1`. This improves the evidence at a longer
clock but does not erase the failed short-clock screen. Full competition-clock
testing versus exact v6 is still required; the candidate ZIP is
`candidate-checks-v7.zip`, SHA256
`976ab1e75c40c6a1e13702bf3b765d93fcdf35832dad560435cd713b010b9c3a`.
The filename is a candidate label, not a submitted platform version.

## Other game reviews

Round 13, a draw as White against StockZero, highlights the endgame position
`8/8/1R1p2pk/3B1p2/P4P1P/2rn4/5K2/8 w - - 5 41`: the played `K e3`
(UCI `f2e3`) had the largest non-matching depth-five loss proxy, 204 cp, with
40.616 seconds remaining. This needs deeper independent evaluation before
calling it a proven blunder. Some other large proxies occur even when the
played move equals the reviewer's best move, demonstrating the measure's noise.

Round 15 was a checkmate win as White against Cheeky. Its largest non-matching
proxy was 100 cp in an already-favorable position, not a comparable immediate
king-safety collapse. These observations are descriptive, not build attribution.

The review also exposed nonzero evaluations after entering insufficient-
material endings. `lmr_draws` separately sets only clearly recognized dead
material positions to zero, leaving all other evaluation values unchanged.
It has four tests, including a randomized sparse-position differential check
against python-chess. No neural retraining is involved in these experiments.
Its initial six-game screen against the terminal reference finished **2 wins,
3 draws, 1 loss (58.3%)**, no technical terminations. This is preliminary,
not sufficient standalone evidence for a playing-strength claim.

## Timing and submission context

At approximately 23:07 London on 4 September, the dashboard showed v6 active,
6 of 10 uploads used, rating 1498 and rank 126 of 245 teams. Next rated round:
5 September at 08:00. Multiple uploads tonight therefore would not produce
multiple new rated games before midnight; local tests are the feedback loop.
