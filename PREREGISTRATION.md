# Pre-registration of decision rules

This file records the hypotheses, decision rules, and refutation criteria that
were fixed *before* the corresponding runs, so that the reported results are not
outcome-selected. Every rule below was decided in advance; results are reported
whichever way the rule resolves. Nulls and costs are reported alongside gains.

## Ceiling-robust audit gate (Sec. VI)

- **Below-ceiling fold:** full-budget AUROC < 0.95.
- **Testable corpus:** >= 3 below-ceiling folds; otherwise the audit returns a
  null (not-testable), by rule, regardless of any downstream statistic.
- **Naive criterion (pre-registered, then refuted by our own control):** a
  Kendall-tau agreement < 0.5 between full and matched-budget rankings was our
  initial non-identifiability criterion. It was retained as the "naive" baseline
  precisely because the saturated ASVspoof2019-LA negative control showed it
  fires on tied-at-ceiling ranks; it is reported as a demonstrated failure of
  our own prior rule, not a strawman.
- **Ceiling-constant sensitivity:** the 0.95 ceiling and the >=3-fold rule are
  swept (0.90-0.98; >=2 vs >=3) and reported; verdicts are stable.

## Composition-aware recipe (Sec. V)

- **Hypothesis:** at a fixed budget, a source-balanced retained-source sampler
  raises held-out AUROC on below-ceiling folds relative to the naive hash
  sampler.
- **Test:** per-fold hash vs. source-balanced delta, hierarchical
  seed+utterance bootstrap 95% CI, Bonferroni-corrected within each corpus's
  fold family. Report every fold, including nulls and negative deltas.
- **Decision:** an effect is "significant" only if its 95% CI excludes zero;
  "Bonferroni-robust" only if the family-adjusted interval excludes zero.

## MLAAD-en TORTOISE seed extension (Sec. V)

- **Pre-registered before the extension runs:** the base MLAAD sampler sweep used
  5 seeds {7, 42, 99, 123, 2024}. TORTOISE is the only genuinely hard
  (below-ceiling) non-reference MLAAD fold, so it was pre-designated for a
  power extension to 15 seeds. The added 10 seeds {11, 17, 29, 31, 47, 59, 71,
  83, 101, 127} were fixed in advance.
- **Decision rule (fixed before running):** report the combined 15-seed paired
  CI whatever it shows; report the base-5 result alongside; report all seeds.
  No fold was extended conditional on its base-5 outcome, and no other fold's
  seed count was changed.
- **Base-5 vs. 15-seed (both reported):** base-5 delta +0.085 (95% CI includes
  zero); 15-seed delta +0.109 [0.018, 0.203].

## ASVspoof5 second-instrument composition sweep (Sec. VI)

- **Pre-registered:** under frozen WavLM the ASVspoof5 re-slice is testable
  (4 below-ceiling folds) and its full->MRD hardest fold is budget-stable (A24).
  Hypothesis: does composition at the fixed MRD budget change below-ceiling-fold
  AUROC and/or reshuffle the ranking?
- **Decision rule (fixed before running):** any below-ceiling fold's
  balanced-vs-hash CI excluding zero, or a hardest-fold change across samplers,
  counts as composition sensitivity replicating; otherwise the ranking is
  composition-robust. Either outcome is reported. Outcome: composition helps
  (A24/A22 Bonferroni-robust, A23 at 95%) while the ranking stays stable
  (tau = 0.86) -- the audit's "identifiable" outcome class.
