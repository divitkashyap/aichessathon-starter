# Depth diagnosis — 5 September

Replayed four explicit positions using frozen v8 at depths 4–7, forcing the
examined move through the same interior search code. Scores are our engine's
opinion, not independent ground truth. Prior repetition history is absent;
the original playing builds remain unverified. Depth counts individual moves,
with selective reductions and extensions, not an exhaustive uniform tree.

| Position | Depth 4 | Depth 7 | Interpretation |
| --- | --- | --- | --- |
| Round 12, Nf6+ mating threat | Finds mate in 9 plies | Same mate | Current search detects this known tactical threat |
| Round 16, h4 | Prefers gxf6; h4 trails 87 cp | Prefers h4; forced score differs 7 cp | Unstable search evidence; not a confirmed evaluation defect |
| Round 17, Rf7 | Trails preferred move by 199 cp | Preferred, gap 0 | Shallow review overstated the evidence against Rf7 |
| Round 18, f2 | Trails Kg2 by 480 cp | Preferred, gap 0 | Clear depth sensitivity in the diagnostic |

The nonzero gap for a preferred move can arise from selective search and
different search ordering/windows. It is not automatically a bug or a blunder.
These tests do not establish that all moves are objectively best, nor that
deeper search always improves play. In particular, round 16 still needs an
independent assessment before changing evaluation rules.

## What this separates

- Demonstrated: shallow review can falsely identify mistakes; depth sensitivity
  is present in these cases. Keep them as diagnostic fixtures.
- Independently observed: neural v3 has large static errors on sparse ordinary
  positions (see NNUE_DIAGNOSIS.md). It is not the active v8 evaluator, so those
  errors cannot explain v8's actual moves.
- Not established: which remaining rated losses are caused by classical
  evaluation rather than selective search, clock constraints or earlier play.

No runtime change or upload follows from this analysis alone. Next controlled
experiment should compare fixed-node/selectivity settings on unresolved
positions, with independent reference assessments before labelling training
examples. Do not use our shallow engine as its own training ground truth.

## Colab: diagnostics, not retraining yet

Use the existing notebook after dataset creation, explicitly pointing at v3
so stale notebook variables do not silently test v1. This reads existing data
and weights, starts no training, and does not need a new checkpoint or GPU:

```python
%cd /content/Einsteinanium
!python -m tools.validate_nnue --dataset /content/drive/MyDrive/einsteinanium/nnue-v1/data --weights /content/drive/MyDrive/einsteinanium/nnue-v3/einsteinanium-nnue-v3.npz
```

Share the printed summary. Aggregate training loss can conceal sparse-position
regressions. The existing validation split is from the same source, not a
certified independent benchmark. We need its subgroup results before selecting
a new training objective or sampling balance.

Reproduction (choose a new output filename):

```
.venv/bin/python -m tools.diagnose_depth --output artifacts/depth-diagnosis-v8-NEW.json
```
