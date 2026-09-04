# Hallucinated Gambits — Einsteinanium

## Current competition agent: v6

Platform-validated on 4 September 2026. The active source is
`challengers/lmr/`, **not the older root `agent.py` or `make zip` target**.
This is our own Numba search with classical evaluation and conservative
late-move reductions; it ships no neural weights or third-party engine.

- Against frozen v5: 14 wins, 5 draws, 5 losses over 24 short-clock games.
- Separate full competition-clock pair: 1 win, 1 draw, no technical failures.
- Platform initialization: 24.3 seconds in both validation smoke games.
- [Release manifest](releases/v6.json), [experiment evidence](docs/LMR_EXPERIMENT.md).

Rebuild the exact active archive without overwriting the original:

```sh
.venv/bin/python -m tools.package_challenger \
  --source challengers/lmr --out candidate-lmr-v6-rebuilt.zip \
  --files agent.py lmr_core.py lmr_search.py
```

Expected SHA256: `3738ae751e65a397c410a029f56f782f816407142564cc82527834eec05e8e98`.
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

## Upstream starter guide (historical)

The original guide below describes the starter workflow, not the active v6 packaging command.

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
