# Day 2: rated losses and exact-search speed experiment

On 5 September the dashboard showed losses by checkmate in rated rounds 16
(White versus bussin_csv/Jarvis) and 17 (Black versus David Naylor). PGNs are
retained locally in `artifacts/rated-day2/round16.pgn` and `round17.pgn`.
These are own rated games, not Daily Five positions.

V7 was active when inspected. The PGNs do not name the playing build, and the
browser blocked the agent-log downloads; build attribution and actual search
depths remain unverified pending those logs. Both games ended with time left:
39.248 seconds after our last move in round 16, 44.642 in round 17. They were
not clock forfeits, but this does not prove time allocation was optimal.

The dashboard still showed 7/10 uploads used when inspected on 5 September;
do not assume the daily allowance reset at local midnight.

## Observed positions, not certified best-move labels

- Round 16: before `20.h4`, White has 87.083 seconds after its preceding move.
  The subsequent `...Nf2` and bishop exchanges precede a material deficit.
  FEN: `1rb2rk1/6pp/p2p1p2/2q1p1P1/PpP1P1n1/3B1Q2/1P1RN1PP/2K4R w - - 0 20`.
- Round 17: `10...f6`, `11...g5`, `12...h5`, and `14...h4` advance the pawn
  cover around Black's castled king. This is a positional hypothesis, not
  proof that every push is bad. The existing evaluation rewards pawn rank
  advancement without checking whether that advancement exposes the king.
- Before `23...Rf7`, the knight on d6 is attacked; the played continuation
  permits `24.Rxd6`. FEN:
  `r2q1rk1/6b1/1pnnbp2/pNp3p1/2P1P2p/P4N1P/1PQ1BPP1/3R1RK1 b - - 1 23`.

The organizer's aggregate reviews report 35/33 ACPL for us in rounds 16/17,
but no per-move oracle labels were obtained. Our engine's shallow scores must
not be presented as independent proof of a blunder or an optimal alternative.

## Isolated lazy-ordering candidate

Source: `challengers/lmr_lazy_order`, copied from exact v7. Only full-search
move ordering changes: compute each ordering score once, then select the next
move only when it will be searched. Cutoffs skip the remaining sorting work.
Scores are frozen before child searches mutate history, and strict tie/swap
behavior matches v7's eager selection sort. Quiescence, evaluation, move
generation, time allocation and check-extension rules are unchanged.

Four tests passed:

1. 256 randomized positions with preferred, killer, tied and history scores;
   every ordered move matches the eager reference, including after history
   mutation following score preparation.
2. All 24 legacy/validation openings at depth four match v7's move, score and
   exact node count.
3. An expired search deadline restores board and state exactly.
4. The round 12 forced mate remains found at depth four.

A warmed benchmark alternated engine order over three passes of all twelve
validation openings at depth five. All **36** searches had exact move, score
and node-count parity. V7 total: **3835.73 ms**; candidate: **2305.24 ms**;
aggregate speedup **1.664x**, median per-position speedup **1.675x** on this Mac.
This is elapsed search time, not a platform guarantee or playing-strength gain.
No simultaneous main-agent arena was running during this benchmark; other
system/subagent activity was not controlled. Compilation is excluded.

An independent reviewer found no ordering-semantic difference but noted the
extra score-array allocation as a timing risk to verify in actual games.

Archive: `candidate-lazy-order-day2-1.zip`, 53,184 bytes uncompressed.
SHA256: `c3861b36f16e72db12425a4ef4d6c11f9f09a4b485681a1c8bddd5026265facf`.
The first 12-game paired match against exact v7 uses validation openings 1–6,
10000+100 ms, with partial diagnostic logs enabled symmetrically for both
agents. Records: `artifacts/lazy-order-day2-screen-1`. Match in progress;
not submitted. V7 remains active.
