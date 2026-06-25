# Source-Shift Attribution Audit for Codec-Based Audio Deepfake Detection

Reproduction code and per-utterance score files for the paper
**"Decomposing Source-Shift Failure in Codec-Based Audio Deepfake Detection."**

This is a **diagnostic / evaluation** artifact, not a detector release. It lets you
regenerate every claim-bearing number and figure in the paper **from saved
per-utterance scores** — no raw audio, no model weights, and no training required.

> **Scope caveat.** The evaluation uses a *custom* leave-one-source-out (LOSO) split
> over the CodecFake+ codec-generated-speech (CoSG) partition as a **diagnostic
> source-holdout protocol**. It is **not** the official CodecFake+ benchmark, and the
> numbers here are evaluation-attribution diagnostics, not a detector or SOTA claim.

## What it reproduces

| Artifact | Script |
|---|---|
| Table I — full-budget vs. Matched Remaining-Data (MRD) source-holdout AUROCs, CIs, vulnerability-rank intervals; non-MASKGCT means | `src/reproduce.py` |
| Table IV — exact-codec proxy recovery (SimpleSpeech→SQ-Codec, NaturalSpeech3→FACodec) | `src/reproduce.py` |
| Stats (A) — hierarchical seed+utterance bootstrap of the SIMPLESPEECH2 full→MRD AUROC drop | `src/reproduce.py` |
| Stats (B) — paired-by-seed SIMPLESPEECH2 vs. SIMPLESPEECH1 under MRD | `src/reproduce.py` |
| Figure 1 — four-panel SS2 diagnostic trajectory | `src/figures.py` |
| Figure 2 — label-wise score distributions (compression vs. displacement) | `src/figures.py` |
| Table II — Exp 3 budget×composition decomposition (composition lever is codec-architecture-dependent) | `src/factorial_decompose.py` |

## Quickstart

```bash
pip install -r requirements.txt
python src/reproduce.py            # tables + statistics  (~15 s at default 2000 reps)
python src/figures.py             # Figure 1 + Figure 2 -> outputs/*.pdf
python src/factorial_decompose.py # Table II: Exp 3 budget-vs-composition decomposition
```

Expected headline numbers (leak-free):

```
non-MASKGCT mean AUROC: full 0.909  ->  MRD 0.767
SIMPLESPEECH2:          full 0.888 (rank 4/9)  ->  MRD 0.666 (rank 1)   delta 0.222
  hierarchical seed+utterance bootstrap of the drop: 95% CI [0.155, 0.291]  (excludes 0)
  paired vs SIMPLESPEECH1: SS2 below SS1 in exactly 5/10 seeds (mean delta -0.013)  -> a statistical tie
exact-codec proxy recovery:  SS2 (SQ-Codec) 0.691 -> no ;  NS3 (FACodec) 0.942 -> yes
```

## Contents

```
data/
  scores/<experiment>/seed_<n>/<fold>.jsonl   sanitized per-utterance scores:
                                              {utterance_id, label, source_model, score}
  collision_details.csv                       SHA-256 train/test byte-identical pairs (leak audit)
  source_holdout_split.json                   the custom CoSG source-holdout split plan
  EXPORT_MANIFEST.json                        counts of the exported score files
  factorial/<fold>/budget_<b>/<sampler>/seed_<s>.jsonl   Exp 3 budget×composition factorial (92 cells)
src/
  audit_lib.py     loader + metrics (AUROC, EER, pAUC, stratified bootstrap, rank intervals)
  reproduce.py     regenerates Tables I & IV and statistics (A) and (B)
  figures.py       regenerates Figure 1 and Figure 2
outputs/           generated tables (CSV) and figures (PDF)
```

`experiment` is one of: `full_budget_loso`, `budget_matched_loso` (the MRD / hash-matched
regime), `wavlm_frozen_loso`, `wavlm_budget_matched_loso`, and the CoRS proxy anchors
`cors_facodec` / `cors_sqcodec` / `cors_vocos` (held in the `COSG_POOLED` "fold").

## Provenance and leak-freeness

All primary numbers exclude every test record byte-identical (SHA-256) to a training
record (all bona-fide; 13 of 62 for SIMPLESPEECH2, 0 for MASKGCT), listed in
`data/collision_details.csv`. These duplicates reflect source-dataset reuse in the
upstream corpus, not contamination introduced here. The scores are the post-hoc outputs
of two diagnostic instruments — an XLS-R + AASIST adapter detector and a frozen
WavLM-Base+ backend — used as instruments, not as detector contributions.

## Upstream data

Scores are derived from the **CodecFake+** dataset
(https://github.com/ResponsibleGenAI/CodecFake-Plus-Dataset). Please cite CodecFake+ and
observe its license for any use of the underlying audio.

## License

Code is released under the MIT License (see `LICENSE`). The derived score files are
released for reproducibility under CC BY 4.0; the underlying audio remains governed by
the CodecFake+ license.
