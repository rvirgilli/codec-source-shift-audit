# Source-Shift Audit & Composition-Aware Recipe for Codec-Based Audio Deepfake Detection

Reproduction code and per-utterance score files for the paper
**"Composition-Aware Training and Ceiling-Robust Evaluation for Source Shift in
Codec-Based Audio Deepfake Detection."**

This is an **evaluation-and-training** artifact, not a detector release. It regenerates
every claim-bearing number in the paper **from saved per-utterance scores** — no raw
audio, no model weights, and no training required.

> **Scope caveat.** The primary evaluation uses a *custom* leave-one-source-out (LOSO)
> split over the CodecFake+ codec-generated-speech (CoSG) partition as a **diagnostic
> source-holdout protocol** — **not** the official CodecFake+ benchmark. The three
> control corpora (ASVspoof2019-LA, ASVspoof5, MLAAD-en) are likewise custom diagnostic
> re-slices with a matched imbalance profile, not official protocols. Nothing here is a
> detector or SOTA claim.

## What it reproduces (v8 paper mapping)

| Paper artifact | Script |
|---|---|
| Table I + Fig. 1 — composition-aware recipe: naive (hash) vs source-balanced AUROC per fold @558, with paired CIs; cross-composition Kendall tau (with seed-bootstrap CI) and rank moves (Sec. IV–V) | `src/imbalance_rankdistort.py` |
| Table II — ceiling-robust scoping across four corpora; below-ceiling-gated mean full→MRD collapse; the naive-tau metric trap on saturated ASVspoof2019-LA; ceiling sensitivity 0.90–0.98; second-instrument (frozen WavLM) audit (Sec. VI) | `src/ceiling_robust.py` |
| Sec. VI — second-corpus composition test: ASVspoof5 under frozen WavLM (below-ceiling folds gain up to +0.129 with CIs excluding zero; ranking composition-stable, tau 0.86) | `src/asvspoof5_composition.py` |
| Sec. IV — full vs MRD source-holdout AUROCs, CIs, vulnerability-rank intervals, both collapse-mean definitions; SS2 drop bootstrap; SS2-vs-SS1 seed-paired tie | `src/reproduce.py` |
| Sec. V — budget-vs-composition decomposition (SS2/SS1/NS3 × budgets 558/1000/full) | `src/factorial_decompose.py` |
| Sec. V — second-corpus recipe check on the MLAAD-en re-slice (TORTOISE et al.) | `src/mlaad_recipe.py` |
| Sec. VII — proxy-codec coverage matrix incl. the second scalar quantizer (Spectral Codecs) | `src/proxy_coverage.py` |
| Sec. VII — operating-point transfer per fold + DET curves (oracle vs transferred threshold) | `src/operating_point.py` |
| Sec. IV — external LA-trained wav2vec2/AASIST check (pooled EER on CoSG) | `src/external_check.py` |
| Score-distribution figures (compression vs displacement) | `src/figures.py` |

## Quickstart

```bash
pip install -r requirements.txt
python src/reproduce.py             # source-holdout tables + statistics (~15 s)
python src/imbalance_rankdistort.py # recipe table (paper Table I) + tau CIs
python src/ceiling_robust.py        # paper Table II + metric trap + ceiling sweep
python src/factorial_decompose.py   # budget-vs-composition decomposition
python src/mlaad_recipe.py          # MLAAD-en recipe replication check
python src/asvspoof5_composition.py # ASVspoof5/WavLM second-corpus composition test
python src/proxy_coverage.py        # proxy x SimpleSpeech-slice coverage matrix
python src/operating_point.py       # per-fold operating points + DET curves
python src/external_check.py        # external-detector pooled EER
python src/figures.py               # figures -> outputs/*.pdf
```

Expected headline numbers (leak-free, deterministic at the default 2000 reps):

```
recipe @558:            SIMPLESPEECH2 0.673 -> 0.881 (+0.208 [0.135, 0.287])
                        SIMPLESPEECH1 0.594 -> 0.752 (+0.157 [0.072, 0.275])
                        NS3           0.805 -> 0.756 (-0.049, n.s.)
non-MASKGCT mean AUROC: full 0.909  ->  MRD 0.767
mean full->MRD collapse: below-ceiling-gated 0.146 | all-non-MASKGCT 0.142
SIMPLESPEECH2:          full 0.888 (rank 4/9)  ->  MRD 0.666 (rank 1)   delta 0.222
  hierarchical seed+utterance bootstrap of the drop: 95% CI [0.1563, 0.2869] (excludes 0)
  paired vs SIMPLESPEECH1: SS2 below SS1 in exactly 5/10 seeds (mean delta -0.013) -> tie
proxy coverage: FACodec 0.719 > SQ-Codec 0.691 (SS2 slice); Spectral Codecs SS1 0.100 / SS2 0.451
external LA-trained wav2vec2/AASIST on CoSG: pooled EER 27.8% (leak-free rows)
```

## Contents

```
data/
  scores/<experiment>/seed_<n>/<fold>.jsonl   sanitized per-utterance scores:
                                              {utterance_id, label, source_model, score}
  factorial/<fold>/budget_<b>/<sampler>/seed_<s>.jsonl
                                              budget x composition factorial: all 8
                                              non-MASKGCT folds @558; SS2/SS1/NS3 also
                                              @1000 and (SS2/NS3) full
  crosscorpus/<corpus>/<block>/<condition>/seed_<n>/<fold>.jsonl
                                              ASVspoof2019-LA / ASVspoof5 / MLAAD-en
                                              re-slices (full, mrd, samp_* blocks)
  external_anchor/scores.jsonl                inference-only LA-trained wav2vec2/AASIST
  collision_details.csv                       SHA-256 train/test byte-identical pairs
  source_holdout_split.json                   the custom CoSG source-holdout split plan
  EXPORT_MANIFEST.json                        counts of the exported score files
src/
  recipe_sampler.py       reference implementation of the hash / source-balanced /
                          source-proportional samplers (the training recipe)
  audit_lib.py           loader + metrics (AUROC, EER, pAUC, bootstrap, rank intervals)
  crosscorpus_lib.py     loader for the cross-corpus re-slices
  reproduce.py           source-holdout tables + statistics
  imbalance_rankdistort.py  recipe table + composition ranking distortion
  ceiling_robust.py      multi-corpus scoping + metric trap + ceiling sensitivity
  factorial_decompose.py budget-vs-composition decomposition
  mlaad_recipe.py        MLAAD-en recipe replication
  proxy_coverage.py      proxy x slice coverage matrix
  operating_point.py     per-fold operating points + DET curves
  external_check.py      external-detector pooled EER
  figures.py             figures
outputs/                 generated tables (CSV), DET points, figures (PDF)
```

CodecFake+ `experiment` is one of: `full_budget_loso`, `budget_matched_loso` (MRD),
`wavlm_frozen_loso`, `wavlm_budget_matched_loso`, and the CoRS proxy anchors
`cors_facodec` / `cors_sqcodec` / `cors_vocos` / `cors_spectralcodecs`
(held in the `COSG_POOLED` "fold").

### Seed coverage notes (no silent exclusions)

- The MRD regime is 558 train (251 bona-fide) / 87 validation utterances.
- The factorial focus cells (SS2/SS1/NS3 × {558, 1000} × 3 samplers and SS2/NS3 full)
  carry seeds {7, 11, 17, 29, 31}; SIMPLESPEECH1 has no `budget_full` cell of its own —
  its full-budget reference is the 10-seed `full_budget_loso` experiment.
- The ASVspoof5 wavlm_frozen_backend sampler blocks (A18-A25) carry seeds {7, 42, 99,
  123, 2024}; xlsr sampler blocks likewise.
- The MLAAD-en sampler blocks carry seeds {7, 42, 99, 123, 2024}; the TORTOISE
  hash/source-balanced cells carry 10 additional seeds (a pre-registered power
  extension for the only hard non-reference fold; all seeds are reported).

## Provenance and leak-freeness

All primary CodecFake+ numbers exclude every test record byte-identical (SHA-256) to a
training record (all bona-fide; 13 of 62 for SIMPLESPEECH2, 0 for MASKGCT), listed in
`data/collision_details.csv`. These duplicates reflect source-dataset reuse in the
upstream corpus, not contamination introduced here. Cross-corpus re-slices need no such
pruning (independent bona-fide pools). The scores are post-hoc outputs of two diagnostic
instruments — an XLS-R + AASIST adapter detector and a frozen WavLM-Base+ backend — used
as instruments, not as detector contributions.

## Upstream data

CodecFake+ scores derive from the **CodecFake+** dataset
(https://github.com/ResponsibleGenAI/CodecFake-Plus-Dataset). The control re-slices
derive from ASVspoof2019-LA, ASVspoof5, MLAAD, and LibriSpeech. Please cite the upstream
datasets and observe their licenses for any use of the underlying audio.

## License

Code is released under the MIT License (see `LICENSE`). The derived score files are
released for reproducibility under CC BY 4.0; the underlying audio remains governed by
the upstream dataset licenses.
