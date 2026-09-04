# Experiment review — 4 September 2026

The platform dashboard was checked after rated round 14: v5 remains active, rating 1427,
rank 140 of 232, overall record 3 wins / 5 draws / 2 losses. Latest game: checkmate win.

## Evidence corrections

The previously reported 77.1% promotion match used `artifacts/champions/day1-v3`,
not frozen v4. Do not cite that result as v5-versus-v4 evidence. Arena output now
prints resolved candidate/opponent paths and hashes when using archives.

## Rejected experiments

- Persistent tables: 12-game screen 66.7%; 24-game test against the exact v5 archive
  39.6% (+6 =7 -11), no reported technical failures. The playing wrapper uses fresh
  tables. The optional implementation is retained for investigation.
- Aspiration windows: specialist benchmark showed narrow windows increased work;
  wider windows offered negligible gains. Five focused correctness tests passed
  independently. No paired game test completed; no promotion justified.

## Neural challenger

The isolated v3 neural search implementation compiles and warms locally in 8.49 seconds.
Depth-three smoke checks returned a legal starting move and a mating move in a mate-in-one
fixture. These checks do not establish match strength or platform startup time.

The specialist left search-level parity, make/unmake stress testing, broader timing and
paired matches incomplete. The original incremental evaluator's tests do not automatically
validate its integration into this search implementation.

V3's reported validation MAE improved from approximately 1106 to 1046 centipawns, about 5.5%.
This error metric is not an Elo rating. Extreme labels may dominate it. Before another
training run, measure error on balanced positions, confirm runtime parity and play against
the immutable v5 archive. Retraining should follow an identified evaluation problem.

Next priority: complete the neural integration tests and compare against exact v5.
Keep experiment files isolated until reviewed; do not upload them based on smoke checks.
