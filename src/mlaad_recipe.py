"""Second-corpus recipe check: composition-aware sampler on the MLAAD-en re-slice.

Reproduces the paper's cross-corpus recipe sentence (Sec. V): per held-out fold,
seed-mean AUROC under the naive (hash) vs source-balanced retained-source sampler
at the fixed MRD budget, with a paired-by-seed bootstrap CI of the difference
(2000 reps, fixed rng). JENNY is the reference fold (largest source). The
TORTOISE cells carry extra seeds by design (power for the only genuinely hard
non-reference fold); all seeds present in the data are used and reported.

Custom diagnostic TTS-model-holdout re-slice with LibriSpeech bona-fide; NOT an
official MLAAD protocol.
"""
from __future__ import annotations
import numpy as np

from audit_lib import auroc, hierarchical_paired_delta
import crosscorpus_lib as cc

REPS, RNG_SEED = 2000, 7
BONF_M = 9  # folds in the MLAAD family


def per_seed_auroc(loaded):
    return {fold: {seed: auroc(l, s) for seed, (l, s) in d.items()}
            for fold, d in loaded.items()}


def hier_ci(corpus, fold):
    a = cc.load(corpus, "samp_hash", "xlsr_peft_adapter", with_utts=True).get(fold, {})
    b = cc.load(corpus, "samp_source-balanced", "xlsr_peft_adapter", with_utts=True).get(fold, {})
    if not (set(a) & set(b)):
        return None
    point, (lo, hi), arr = hierarchical_paired_delta(a, b)
    q = 100 * 0.05 / (2 * BONF_M)
    blo, bhi = np.percentile(arr, q), np.percentile(arr, 100 - q)
    return (round(point, 3), round(lo, 3), round(hi, 3), len(set(a) & set(b)),
            bool(blo > 0 or bhi < 0))


def main():
    hash_ = per_seed_auroc(cc.load("mlaad", "samp_hash"))
    bal = per_seed_auroc(cc.load("mlaad", "samp_source-balanced"))
    full = per_seed_auroc(cc.load("mlaad", "full"))
    ref = cc.REFERENCE["mlaad"]
    print("== MLAAD-en recipe check: hash vs source-balanced @ fixed MRD budget ==")
    print("   (hierarchical seed+utterance bootstrap CIs; Bonferroni family m=9)")
    print(f"{'fold':14s} {'full':>6} {'hash':>6} {'bal':>6} {'delta':>7} {'95% CI':>18} {'n':>3}")
    for fold in sorted(hash_):
        h, b = hash_[fold], bal.get(fold, {})
        res = hier_ci("mlaad", fold)
        if res is None:
            continue
        mean, lo, hi, n, bonf = res
        fb = np.mean(list(full[fold].values())) if fold in full else float("nan")
        flags = " [REF]" if fold == ref else ""
        if fb < 0.95:
            flags += " [below-ceiling]"
        if bonf:
            flags += " Bonf-robust"
        print(f"{fold:14s} {fb:>6.3f} {np.mean(list(h.values())):>6.3f} "
              f"{np.mean(list(b.values())):>6.3f} {mean:>+7.3f} "
              f"{'[' + str(lo) + ', ' + str(hi) + ']':>18} {n:>3}{flags}")


if __name__ == "__main__":
    main()
