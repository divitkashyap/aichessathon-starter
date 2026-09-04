# Terminal-position correctness challenger

`challengers/lmr_terminal` is a separate variant of the LMR challenger. It
does not modify the frozen v5 or the initial LMR candidate.

Fixed with regression tests:

- Checkmate at the quiescence depth cap returns a mate score, not static material.
- Stalemate returns zero even when a static score would trigger a shortcut.
- Checkmate takes precedence over the fifty-move draw in both main and
  quiescence search.

A short-circuit `has_legal_move` routine stops at the first legal move and
always undoes its temporary move. It agrees with Python-chess and restores
the board in a 128-ply deterministic differential test.

Four dedicated tests pass. On all 12 opening positions at depth five, it
matches LMR's best move and node count exactly (744954 total nodes). One
interleaved local timing measured 1203 ms versus 1117 ms for LMR, about 7.8%
overhead. Other local tests were running; this is not a clean hardware benchmark.

The repetition-history/transposition concern remains unresolved. No claim
that all draw-handling edge cases have been fixed.

Paired-game test versus the frozen LMR archive: **4 wins, 1 draw, 1 loss**
on 3 paired openings at 1000+50 ms, no technical terminations. Recorded in
`artifacts/lmr-terminal-screen-1`. The small sample does not establish a
precise improvement over LMR. An additional 18-game test versus frozen v5
finished **9 wins, 4 draws, 5 losses (61.1%)**, no technical terminations,
in `artifacts/lmr-terminal-validation-1`. LMR without these fixes scored
63.9% on that same validation suite in a separate run. These small samples
do not distinguish the variants reliably. Keep the fixes available for a
future release; the current approved v6 upload remains unchanged.

Independent new-opening comparison against frozen v6 (`--suite validation`,
12 paired openings, 2000+100 ms) finished **7 wins, 11 draws, 6 losses
(52.1%)**, no technical terminations. Records:
`artifacts/terminal-unseen-v6-1`. This is compatible with similar strength;
do not claim a demonstrated playing-strength increase from this small margin.
