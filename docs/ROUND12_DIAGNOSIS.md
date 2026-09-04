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

## Timing and submission context

At approximately 23:07 London on 4 September, the dashboard showed v6 active,
6 of 10 uploads used, rating 1498 and rank 126 of 245 teams. Next rated round:
5 September at 08:00. Multiple uploads tonight therefore would not produce
multiple new rated games before midnight; local tests are the feedback loop.
