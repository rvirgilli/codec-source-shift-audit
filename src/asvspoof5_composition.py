"""Second-corpus composition test: ASVspoof5 re-slice under frozen WavLM.

Pre-registered follow-up to the second-instrument audit (see ceiling_robust.py):
under frozen WavLM the ASVspoof5 re-slice is testable (4 below-ceiling folds)
and the full->MRD hardest fold is budget-stable (A24). This script asks whether
retained-source COMPOSITION at the fixed MRD budget moves that corpus: per-fold
hash vs source-balanced (and source-proportional) seed-mean AUROC with
paired-by-seed bootstrap CIs, plus the cross-composition Kendall tau and rank
moves over the non-reference folds.

Decision rule (pre-registered before the runs): any below-ceiling fold's
balanced-hash CI excluding 0, or a hardest-fold change across samplers, counts
as composition sensitivity replicating on a second corpus; otherwise the
ranking there is composition-robust and the result is reported as the audit's
third outcome class. Either way is reported.

Reads data/crosscorpus/asvspoof5/samp_<sampler>/wavlm_frozen_backend/.
"""
from __future__ import annotations
import itertools, random
import numpy as np

from audit_lib import auroc, hierarchical_paired_delta
import crosscorpus_lib as cc

COND = "wavlm_frozen_backend"
SAMPLERS = ["hash", "source-balanced", "source-proportional"]
REPS, RNG_SEED = 2000, 7


def kendall_tau(a, b):
    n = len(a); c = d = 0
    for i, j in itertools.combinations(range(n), 2):
        s = np.sign((a[i] - a[j]) * (b[i] - b[j]))
        if s > 0: c += 1
        elif s < 0: d += 1
    return (c - d) / (0.5 * n * (n - 1))


def per_seed(loaded):
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
    cells = {s: per_seed(cc.load("asvspoof5", f"samp_{s}", COND)) for s in SAMPLERS}
    full = per_seed(cc.load("asvspoof5", "full", COND))
    folds = sorted(cells["hash"])
    if not folds:
        print("no ASVspoof5 wavlm sampler data in the bundle"); return
    print("== ASVspoof5 / frozen WavLM: composition at fixed MRD budget ==")
    print("   (hierarchical seed+utterance bootstrap CIs; Bonferroni family m=8)")
    print(f"{'fold':6} {'full':>6} {'hash':>6} {'bal':>6} {'prop':>6} {'d(bal-hash)':>12} {'95% CI':>17} {'n':>3}")
    M = {s: {} for s in SAMPLERS}
    q = 100 * 0.05 / (2 * 8)
    for fold in folds:
        for s in SAMPLERS:
            M[s][fold] = float(np.mean(list(cells[s][fold].values()))) if fold in cells[s] else float("nan")
        a = cc.load("asvspoof5", "samp_hash", COND, with_utts=True).get(fold, {})
        b = cc.load("asvspoof5", "samp_source-balanced", COND, with_utts=True).get(fold, {})
        point, (lo, hi), arr = hierarchical_paired_delta(a, b)
        n = len(set(a) & set(b))
        blo, bhi = np.percentile(arr, q), np.percentile(arr, 100 - q)
        fb = float(np.mean(list(full[fold].values()))) if fold in full else float("nan")
        flag = " *below-ceiling*" if fb < 0.95 else ""
        ex = " CI-excl-0" if (lo > 0 or hi < 0) else ""
        if blo > 0 or bhi < 0:
            ex += " Bonf-robust"
        print(f"{fold:6} {fb:>6.3f} {M['hash'][fold]:>6.3f} {M['source-balanced'][fold]:>6.3f} "
              f"{M['source-proportional'][fold]:>6.3f} {point:>+12.3f} "
              f"{'[' + format(lo, '.3f') + ', ' + format(hi, '.3f') + ']':>17} {n:>3}{flag}{ex}")

    print("\n== Cross-composition agreement / hardest fold ==")
    import random
    seeds = sorted(set.intersection(*[set(cells[s][f]) for s in SAMPLERS for f in folds]))
    r = random.Random(7)
    for x, y in itertools.combinations(SAMPLERS, 2):
        common = [f for f in folds if not np.isnan(M[x][f]) and not np.isnan(M[y][f])]
        t = kendall_tau([M[x][f] for f in common], [M[y][f] for f in common])
        bs = []
        for _ in range(2000):
            draw = [seeds[r.randrange(len(seeds))] for _ in seeds]
            av = [np.mean([cells[x][f][sd] for sd in draw]) for f in common]
            bv = [np.mean([cells[y][f][sd] for sd in draw]) for f in common]
            bs.append(kendall_tau(av, bv))
        lo, hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
        print(f"  tau({x} vs {y}) = {t:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  ({len(common)} folds)")
    for s in SAMPLERS:
        valid = {f: v for f, v in M[s].items() if not np.isnan(v)}
        if valid:
            print(f"  hardest under {s:20s}: {min(valid, key=valid.get)}")


if __name__ == "__main__":
    main()
