"""Bona-fide displacement diagnostic (paper Sec. VII).

Frozen WavLM on the held-out MASKGCT fold scores below chance (AUROC < 0.5).
This script shows it is bona-fide displacement (held-out bona-fide scored more
spoof-like than spoof), not label inversion: AUROC, d-prime, and the median
spoof-minus-bonafide score gap, all seed-mean over the 10 wavlm_frozen_loso
seeds. d-prime uses the standard pooled-variance estimator
d' = (mean_spoof - mean_bona) / sqrt(0.5 (var_spoof + var_bona)).
Reproduces the paper's 0.422 / -0.345 / -0.046.
"""
from __future__ import annotations
import numpy as np
from audit_lib import load, load_exclude, auroc


def d_prime(spoof, bona):
    return (spoof.mean() - bona.mean()) / np.sqrt(0.5 * (spoof.var() + bona.var()))


def main():
    items = load(load_exclude())
    a, d, g = [], [], []
    for it in items:
        if it.experiment == "wavlm_frozen_loso" and it.fold == "MASKGCT":
            sp, bo = it.scores[it.labels == 1], it.scores[it.labels == 0]
            lab = np.r_[np.ones(len(sp)), np.zeros(len(bo))].astype(int)
            a.append(auroc(lab, np.r_[sp, bo]))
            d.append(d_prime(sp, bo))
            g.append(np.median(sp) - np.median(bo))
    print("== Frozen WavLM on MASKGCT: bona-fide displacement (seed-mean, n={} seeds) ==".format(len(a)))
    print(f"  AUROC            {np.mean(a):.3f}   (< 0.5 = below chance)")
    print(f"  d-prime          {np.mean(d):.3f}   (pooled-variance estimator)")
    print(f"  median spoof-bona {np.mean(g):+.3f}  (negative = bona-fide scored more spoof-like)")


if __name__ == "__main__":
    main()
