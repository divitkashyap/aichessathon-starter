# Hallucinated Gambits — Einsteinanium

## Current competition agent: v8

Platform-validated on 5 September 2026. The active source is
`challengers/lmr_lazy_order/`, **not the older root `agent.py` or `make zip` target**.
This is our own Numba search with classical evaluation and conservative
late-move reductions, bounded check extensions, terminal-position fixes and
dead-material recognition; it ships no neural weights or third-party engine.
In plain terms, it follows forcing checks further and avoids assigning a
winning score to clearly drawn material endings. V8 preserves v7's evaluation
and search decisions at fixed depth, but avoids sorting moves that will never
be searched. Identical warmed search work ran about 1.66x faster on this Mac.

- Against frozen v7: 11 wins, 6 draws, 7 losses over 24 games at 10 seconds + 0.1 second/move.
- Separate full competition-clock pair: 1 win, 1 draw, no technical failures.
- Platform initialization: 26.3 and 26.4 seconds, within the 90-second budget.
- Active status verified at 09:57 London on 5 September; 8 of 10 uploads shown used.
- [Release manifest](releases/v8.json), [experiment evidence](docs/DAY2_FIXES.md).

These are small local samples against one opponent, not a proven Elo gain.
V7's exact archive and [release record](releases/v7.json) remain preserved;
its earlier evidence is in the [check/draw experiment](docs/ROUND12_DIAGNOSIS.md).

Rebuild the exact active archive without overwriting the original:

```sh
.venv/bin/python -m tools.package_challenger \
  --source challengers/lmr_lazy_order --out candidate-v8-rebuilt.zip \
  --files agent.py lmr_core.py lmr_search.py
```

Expected SHA256: `c3861b36f16e72db12425a4ef4d6c11f9f09a4b485681a1c8bddd5026265facf`.
Other `challengers/` directories are experiments, not automatically approved upgrades.
See [NNUE diagnosis](docs/NNUE_DIAGNOSIS.md) before spending more training time.

## Changes, lessons and evidence

Most demonstrated gains so far came from faster, better-directed search and
paired games against frozen versions of our own agent—not neural training or
simply beating the starter minimax bot. Scores below count draws as half a win.

| Change | What we learned | Evidence and limits |
| --- | --- | --- |
| Compiled move generation and search | More useful calculation fits within the clock. | 100,000 differential positions matched python-chess; compiled candidate scored 72.9% over 24 games against v1. |
| Search-position cache | Avoid repeating work when move sequences reach the same position. | v4 candidate scored 62.5% over 24 games against v3. |
| Better move ordering | Finding promising moves first makes the remaining search cheaper. | v5 candidate scored 77.1% over 24 games against **v3**, not v4; the earlier opponent attribution was corrected. |
| Conservative late-move reductions | Initially search late quiet moves less deeply, then verify promising ones. | v6 scored 9W/5D/4L in a separate 18-game validation against v5; full-clock pair was 1W/1D. |
| Follow checks further; recognize dead material | A real rated loss exposed a forced mate our search missed. The combined candidate finds it at depth four; simple drawn endings now evaluate to zero. | v7 scored 3W/2D/1L directly against v6, then 2W at the full clock. Check-only failed its separate full-clock pair (0W/1D/1L); tactical success alone was not enough. |
| Avoid unnecessary move sorting | Same fixed-depth moves, scores and node counts, with less repeated ordering work. More depth fits into the clock. | v8 matched 24 depth-four searches and 36 depth-five benchmark searches; about 1.66x faster locally. Two 12-game screens combined to 11W/6D/7L vs v7; full-clock pair 1W/1D. Rated strength gain not established yet. |
| Remove pawn-advancement reward around our king, not promoted | A targeted positional change altered the observed pawn-push decisions, but changing behavior is not proof of better play. | Eight tests passed, including 256 independent numeric evaluations; game screen 2W/1D/3L vs v7. Not included in v8. |
| King pawn-cover penalty, not promoted | A sensible positional rule can look promising against one reference and fail against the actual champion. | 9W/8D/7L against the terminal-fix reference, but 4W/3D/5L directly against v6 at a longer clock; full-clock pair split 1W/1L. |
| Neural evaluator and guarded variants | Lower aggregate prediction error does not guarantee stronger play. Sparse endgames are a major weakness. | On a second 850-position sample, ordinary-position MAE was 130 cp classical vs 249 cp neural; a bounded blend reached 124 cp but lost its v6 game screen. No neural candidate promoted. |
| Rejected search/evaluation ideas | Plausible improvements still need to earn promotion. | Persistent memory scored 39.6% over 24 games against v5. Pawn bonuses, guarded null search and neural root ordering also failed to earn promotion in small screens. |

These are local, opponent-specific results, not universal win rates or Elo
estimates. Small samples are noisy. The public diagnostic samples are disjoint
from one another, but overlap with the original training set is not certified.
See the [development record](docs/ROADMAP.md), [neural diagnosis](docs/NNUE_DIAGNOSIS.md)
and [follow-up screens](docs/FOLLOWUP_SCREENS.md) for caveats and reproducibility.

## How we improve next

1. Review competition losses, verify the playing build where possible, and
   retain critical positions as regression tests. Our own analysis scores are
   diagnostic estimates, not an independent chess oracle.
2. State one concrete hypothesis and change one feature in an isolated candidate.
3. Pass legality, board-restoration, tactical, clock and packaging checks.
4. Screen on development openings, then validate on different openings with both
   colors and multiple frozen opponents. Use longer-clock games before release.
5. Promote only with supporting evidence; preserve the exact previous archive.
   Keep failures in the record instead of silently retuning the validation set.

The arena now supports `--suite legacy` (the unchanged original 12 openings)
and `--suite validation` (12 additional openings). Each match manifest records
its selected FENs and source/archive hashes. Once used to select a change, a
validation suite is no longer an untouched final test for further tuning.

Training is a separate experiment: first measure errors on the existing Colab
validation shards by position type, then choose targeted data/objective changes.
Do not spend GPU time merely adding epochs to the same misleading average.

## Frozen qualifier rules: organizer update, 4 September

The organizer email confirms these rules apply from 5 September at 08:00:
90-second startup allowance, ten uploads per team per day, and process
suspension while the opponent thinks. The next rated round shown after v7's
validation is 5 September at 08:00 London. Spare uploads do not create extra
rated rounds; keep the best validated agent active rather than using slots
for their own sake. The daily reset timezone was not specified in the email.

Hourly rated rounds continue until uploads close at 11:00 on 11 September.
The final Swiss that afternoon decides the fifty London seats; today's ladder
rank is feedback, not confirmation of qualification. See the
[official rules](https://aichessathon.com/docs/rules.md).

## Upstream starter guide (historical)

The original guide below describes the starter workflow, not the active v8 packaging command.

Fork this to build an agent for [AI Chessathon](https://aichessathon.com). It gives you a working
submission, baselines to beat, and a local harness that speaks the same protocol and enforces the
same clock as the platform, so you can see whether a change actually helped before you upload it.

```
git clone https://github.com/advitrocks9/aichessathon-starter
cd aichessathon-starter
make setup
make play
```

That plays your agent against a baseline over a full 120 s + 0.5 s game and prints the result.
When you like it, `make zip` and drop `submission.zip` on your dashboard.

## Writing an agent

`agent.py` is the whole submission. One function:

```python
def get_move(fen: str, time_left_ms: int) -> str:
    return "e2e4"
```

The fork ships a legal random-mover, so the loop works before you write anything. Replace the body.

```
make play                                          # one game, real time control
make arena                                         # 20 fast games, prints a score
make play FEN="<fen>"                              # start from a given position
uv run python -m harness.play --black baselines/minimax --pgn game.pgn
uv run python -m harness.arena --opponent ../my-old-version --games 200
```

Anything your agent writes to stdout or stderr shows up under the result, so `print` debugging
works. The platform discards it during rated games and shows it in your validation log.

## The ladder

Measured with `harness/arena.py`. Beating greedy is a search. Beating minimax is a search plus an
evaluation worth searching with.

| Matchup | Games | Time control | Score |
|---|---|---|---|
| random vs greedy | 20 | 10 s + 0.1 s | 10.0% (+1 =2 -17) |
| greedy vs minimax | 6 | 120 s + 0.5 s | 0.0% (+0 =0 -6) |
| numba vs minimax | 6 | 10 s + 0.5 s | 66.7% (+2 =4 -0) |

- `baselines/random` plays a uniformly random legal move. It is what `agent.py` starts as.
- `baselines/greedy` searches one ply on material.
- `baselines/minimax` searches two plies on material and mobility, with no time management.
- `baselines/numba` is `minimax` with the evaluation jitted. It is barely stronger, which is
  the point: jitting a shallow search buys headroom, not depth. Read it for the warm-up call
  at the bottom, which is how you keep compilation off your clock.

## What's here

```
agent.py             your submission
baselines/           random, greedy, minimax, numba; each is a directory with an agent.py
harness/runner.py    the process the platform runs your agent in
harness/referee.py   the clock, legality, draw and adjudication rules
harness/rules.py     the event constants the harness enforces
harness/sandbox.py   the one process, spoken to as the platform speaks to a container
harness/play.py      one game between two agent directories
harness/arena.py     many games, with a score
harness/package.py   builds submission.zip with agent.py at the root
docs/IDEAS.md        where the strength actually comes from
```

Local games start from the normal position unless you pass `--fen`. Rated games start from
curated neutral positions.

The harness is here so your games are honest, not so you can pre-validate an upload. Acceptance
happens on the platform, and the validation log on your dashboard is the authority on it.

## The rules

[aichessathon.com/docs](https://aichessathon.com/docs) is canonical and changes. Read it before
you upload.
