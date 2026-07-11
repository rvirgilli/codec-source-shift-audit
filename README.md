# Codec Source-Holdout Audit

Score-level artifact for:

> **Auditing Source Holdout in Codec-Based Audio Deepfake Detection: Budget,
> Composition, and Rank Uncertainty**

This is an evaluation-methodology artifact, not a detector release. It regenerates
the v9 statistical claims from released per-utterance scores. It does **not** contain
raw audio, training code, checkpoints, or the concrete selected training-utterance
manifests, so it does not reproduce model training. `src/recipe_sampler.py` is the
reference implementation of the two sampling policies, not proof of the exact rows
used by every completed run.

## Scope

The primary analysis uses a custom leave-one-source-out (LOSO) split over the
CodecFake+ codec-generated-speech (CoSG) partition. It is not the official CodecFake+
benchmark protocol. The ASVspoof5 and MLAAD-en analyses are also custom diagnostic
re-slices and are reported only as scope checks.

## V9 claim map

| V9 result | Authoritative script | Machine-readable output |
|---|---|---|
| Fixed-budget hash vs source-balanced effects; crossed seed/item CIs; eight-fold multiplicity; fixed-source macro; budget-policy factorial | `src/crossed_global_analysis.py` | `outputs/crossed_global_analysis.json` |
| Rank uncertainty and all 28 pairwise source orders | `src/rank_crossed_audit.py` | `outputs/rank_crossed_audit_200k.json` |
| ASVspoof5 and MLAAD scope checks | `src/external_crossed_effects.py` | `outputs/external_crossed_effects_200k.json` |
| Seed-only, item-only, and crossed uncertainty sensitivity | `src/uncertainty_components.py` | `outputs/uncertainty_components_200k.json` |
| Tie-correct AUROC validation | `src/validate_ties_threshold_bootstrap.py` | `outputs/ties_threshold_bootstrap_summary.json` and `outputs/tie_correct_*.csv` |
| Deterministic reference sampler | `src/recipe_sampler.py` | source code only; selected training IDs are not released |

Older scripts and figures remain as supporting/legacy analyses, but they are not
authoritative for v9 claims. In particular, use the tie-correct Mann--Whitney AUROC in
the v9 scripts rather than the legacy ROC-trapezoid helper in `src/audit_lib.py`.

## Quickstart

```bash
python -m pip install -r requirements.txt
python src/crossed_global_analysis.py > outputs/crossed_global_analysis.stdout
python src/rank_crossed_audit.py --reps 200000 --output outputs/rank_crossed_audit_200k.json
python src/external_crossed_effects.py --reps 200000 --output outputs/external_crossed_effects_200k.json
python src/uncertainty_components.py > outputs/uncertainty_components_200k.stdout
```

The main scripts use NumPy. Matplotlib supports legacy figure regeneration, and
scikit-learn is used only by the optional tie-validation script. The crossed analyses
were validated with Python 3.13.2 and NumPy 2.3.2.

## Expected v9 results

Primary CodecFake+ family, five paired seeds and 200,000 crossed draws:

```text
SimpleSpeech2: +0.208; ordinary 95% CI [ 0.086, 0.361]
                         Bonferroni m=8 [ 0.048, 0.432]
SimpleSpeech1: +0.157; ordinary 95% CI [ 0.012, 0.308]
                         Bonferroni m=8 [-0.049, 0.383]
```

Both deltas are positive in all five paired seeds on the fixed released test folds.
Only SimpleSpeech2 is familywise positive under the crossed seed-and-item analysis.
Seven of eight point effects are positive, but the fixed-source macro is unresolved:
`+0.080 [-0.026, 0.195]`. The global max-T test for any primary effect gives
`p=0.0326`.

Rank audit:

```text
hash policy:            0/28 pairwise source orders resolved
source-balanced policy: 0/28 pairwise source orders resolved
hash vs balanced tau-b: 0.071 [-0.357, 0.643]
```

The `0/28` conclusion holds under both single-step max-T and Bonferroni percentile
intervals and under the independent 100,000-draw sensitivity run.

External scope checks do not establish replication:

```text
ASVspoof5 / frozen WavLM: global max-T p=0.217; none familywise positive
MLAAD-en / XLS-R adapter:  global max-T p=0.520; none familywise positive
```

## Data layout

```text
data/
  scores/<experiment>/seed_<n>/<fold>.jsonl
  factorial/<fold>/budget_<b>/<sampler>/seed_<s>.jsonl
  crosscorpus/<corpus>/<block>/<condition>/seed_<n>/<fold>.jsonl
  collision_details.csv
  source_holdout_split.json
  EXPORT_MANIFEST.json
outputs/
  crossed_global_analysis.json
  rank_crossed_audit_200k.json
  rank_crossed_audit_100k_sensitivity.json
  external_crossed_effects_200k.json
  uncertainty_components_200k.json
src/
  crossed_global_analysis.py
  rank_crossed_audit.py
  external_crossed_effects.py
  uncertainty_components.py
  validate_ties_threshold_bootstrap.py
  recipe_sampler.py
```

Score JSONL rows contain sanitized `{utterance_id, label, source_model, score}` fields.
All primary test IDs align across policy arms and seeds after the released SHA-256
collision exclusions. Underlying audio remains governed by the upstream dataset
licenses.

## Analysis provenance

The crossed-bootstrap, simultaneous-rank, multiplicity, and tie-correct analyses are
revision-stage robustness analyses; they were not preregistered. See
`ANALYSIS_PROVENANCE.md` for the repository-backed chronology. Only the uniform
nine-fold MLAAD extension has a public pre-run rule commit in this repository, and it
is not used as confirmatory support in v9.

## Reproducibility boundary

This release supports score-level statistical reproduction. It does not support
end-to-end training reproduction because training code, model checkpoints, and exact
selected training IDs are absent. If the selection manifests are available privately,
the release should add one JSONL per fold, budget, policy, and seed before submission.

## Upstream data and license

CodecFake+ scores derive from the CodecFake+ dataset:
https://github.com/ResponsibleGenAI/CodecFake-Plus-Dataset. External scores derive
from ASVspoof5, MLAAD, and LibriSpeech. Code is MIT licensed. Derived scores are CC BY
4.0; raw audio is not redistributed here.
