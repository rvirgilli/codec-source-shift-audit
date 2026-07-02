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
import random
import numpy as np

from audit_lib import auroc
import crosscorpus_lib as cc

REPS, RNG_SEED = 2000, 7


def per_seed_auroc(loaded):
    return {fold: {seed: auroc(l, s) for seed, (l, s) in d.items()}
            for fold, d in loaded.items()}


def paired_ci(c1, c2):
    common = sorted(set(c1) & set(c2))
    diffs = [c2[s] - c1[s] for s in common]
    if not diffs:
        return None
    r = random.Random(RNG_SEED); n = len(diffs)
    bs = [np.mean([diffs[r.randrange(n)] for _ in range(n)]) for _ in range(REPS)]
    return (round(float(np.mean(diffs)), 3),
            round(float(np.percentile(bs, 2.5)), 3),
            round(float(np.percentile(bs, 97.5)), 3), n)


def main():
    hash_ = per_seed_auroc(cc.load("mlaad", "samp_hash"))
    bal = per_seed_auroc(cc.load("mlaad", "samp_source-balanced"))
    full = per_seed_auroc(cc.load("mlaad", "full"))
    ref = cc.REFERENCE["mlaad"]
    print("== MLAAD-en recipe check: hash vs source-balanced @ fixed MRD budget ==")
    print(f"{'fold':14s} {'full':>6} {'hash':>6} {'bal':>6} {'delta':>7} {'95% CI':>18} {'n':>3}")
    for fold in sorted(hash_):
        h, b = hash_[fold], bal.get(fold, {})
        res = paired_ci(h, b)
        if res is None:
            continue
        mean, lo, hi, n = res
        fb = np.mean(list(full[fold].values())) if fold in full else float("nan")
        flags = " [REF]" if fold == ref else ""
        if fb < 0.95:
            flags += " [below-ceiling]"
        print(f"{fold:14s} {fb:>6.3f} {np.mean(list(h.values())):>6.3f} "
              f"{np.mean(list(b.values())):>6.3f} {mean:>+7.3f} "
              f"{'[' + str(lo) + ', ' + str(hi) + ']':>18} {n:>3}{flags}")


if __name__ == "__main__":
    main()
