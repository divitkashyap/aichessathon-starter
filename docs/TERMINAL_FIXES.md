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

Paired-game test versus the frozen LMR archive is separate and recorded in
`artifacts/lmr-terminal-screen-1`. Do not infer its result before completion.
