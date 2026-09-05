# Mobility pilot: not promoted

The offline pilot recomputes the active v8 evaluator on the existing 690-position
and 850-position diagnostic samples. It counts knight, bishop and rook attack
squares excluding friendly pieces and enemy pawn attacks. This is not legal
mobility: pins and attacks from other pieces are not accounted for.

One coefficient (0–8 centipawns per square) was selected on non-mate positions
with target magnitude at most 1000 in the first sample, then applied unchanged
to the second sample. Material values and piece-square scores were unchanged.
The samples are disjoint but previously explored, not untouched holdouts.

Selected coefficient: 1 cp. First-sample MAE: 180.524 -> 180.465 cp.
Second-sample MAE: 161.051 -> 160.727 cp. Ordinary-position error (target magnitude
at most 300) worsened: 125.697 -> 126.179 and 129.750 -> 129.769 cp respectively.
These tiny changes are not a compelling improvement. No runtime integration,
tournament, retraining or submission was justified by this pilot.

Reproduce with a new output path:

```
.venv/bin/python -m tools.audit_mobility artifacts/nnue-audit-v3-20260904.json artifacts/nnue-audit-v3-independent-20260904.json --output artifacts/mobility-audit-NEW.json
.venv/bin/python -m unittest discover -s tests -p test_mobility_audit.py
```

Next evidence needed: evaluation errors on quiet positions reached by our actual
search, and the existing neural validation report broken down by position type.
Public position-label error alone mixes tactical search and static evaluation
errors, so it cannot establish whether a change will win more games.
