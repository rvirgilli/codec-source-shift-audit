"""Composition-induced ranking distortion at fixed budget (paper Sec. IV/V).

For each retained-source composition {hash, source-balanced, source-proportional}
at the fixed 558 budget, rank all 8 non-MASKGCT folds by leak-free seed-mean
held-out AUROC, then measure how much the vulnerability ranking changes across
compositions (Kendall tau with a seed-paired bootstrap CI, plus per-fold rank
shifts). Also prints the recipe table (naive hash vs source-balanced AUROC and
delta per fold — paper Table I).

Reads data/factorial/<fold>/budget_558/<sampler>/seed_<n>.jsonl.
"""
from __future__ import annotations
import itertools, json, random
import numpy as np

from audit_lib import ROOT, load_exclude, auroc

TREE = ROOT / "data" / "factorial"
FOLDS = ["CLAMTTS", "GPST", "NS2", "UNIAUDIO", "VALLE",
         "SIMPLESPEECH1", "SIMPLESPEECH2", "NS3"]
SAMPLERS = ["hash", "source-balanced", "source-proportional"]
QUANT = {"SIMPLESPEECH1": "Scq", "SIMPLESPEECH2": "Scq"}
REPS, RNG_SEED = 2000, 7


def kendall_tau(a, b):
    n = len(a); c = d = 0
    for i, j in itertools.combinations(range(n), 2):
        s = np.sign((a[i] - a[j]) * (b[i] - b[j]))
        if s > 0: c += 1
        elif s < 0: d += 1
    return (c - d) / (0.5 * n * (n - 1))


def per_seed_auroc(fold, samp, excl):
    d = {}
    for f in sorted((TREE / fold / "budget_558" / samp).glob("seed_*.jsonl")):
        seed = int(f.stem.replace("seed_", ""))
        lab, sc = [], []
        for ln in f.open():
            o = json.loads(ln)
            if str(o.get("utterance_id", "")) in excl:
                continue
            lab.append(1 if o.get("label") == "spoof" else 0)
            sc.append(float(o["score"]))
        if sum(lab) and len(lab) - sum(lab):
            d[seed] = auroc(np.asarray(lab), np.asarray(sc))
    return d


def paired_ci(c1, c2):
    common = sorted(set(c1) & set(c2))
    diffs = [c2[s] - c1[s] for s in common]
    r = random.Random(RNG_SEED); n = len(diffs)
    bs = [np.mean([diffs[r.randrange(n)] for _ in range(n)]) for _ in range(REPS)]
    return (round(float(np.mean(diffs)), 3),
            round(float(np.percentile(bs, 2.5)), 3),
            round(float(np.percentile(bs, 97.5)), 3))


def main():
    excl = load_exclude()
    cells = {(f, s): per_seed_auroc(f, s, excl) for f in FOLDS for s in SAMPLERS}
    seeds = sorted(set.intersection(*(set(c) for c in cells.values())))
    M = {s: {f: float(np.mean(list(cells[(f, s)].values()))) for f in FOLDS} for s in SAMPLERS}

    print("== Recipe table (paper Table I): naive (hash) vs source-balanced @558, leak-free ==")
    print(f"{'fold':14s} {'Q.':>4} {'naive':>7} {'balanced':>9} {'delta':>7} {'95% CI':>18}")
    for f in sorted(FOLDS, key=lambda x: -(M['source-balanced'][x] - M['hash'][x])):
        mean, lo, hi = paired_ci(cells[(f, "hash")], cells[(f, "source-balanced")])
        print(f"{f:14s} {QUANT.get(f, 'Mvq'):>4} {M['hash'][f]:>7.3f} {M['source-balanced'][f]:>9.3f} "
              f"{mean:>+7.3f} {'[' + str(lo) + ', ' + str(hi) + ']':>18}")

    def ranks(sampdict):
        order = sorted(FOLDS, key=lambda f: sampdict[f])
        return {f: i + 1 for i, f in enumerate(order)}
    R = {s: ranks(M[s]) for s in SAMPLERS}

    print("\n== Cross-composition ranking agreement (8 folds, seed-paired bootstrap CI) ==")
    r = random.Random(RNG_SEED)
    for x, y in itertools.combinations(SAMPLERS, 2):
        point = kendall_tau([M[x][f] for f in FOLDS], [M[y][f] for f in FOLDS])
        bs = []
        for _ in range(REPS):
            draw = [seeds[r.randrange(len(seeds))] for _ in seeds]
            av = [np.mean([cells[(f, x)][sd] for sd in draw]) for f in FOLDS]
            bv = [np.mean([cells[(f, y)][sd] for sd in draw]) for f in FOLDS]
            bs.append(kendall_tau(av, bv))
        lo, hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
        print(f"  {x:20s} vs {y:20s} tau {point:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")

    print("\n== Rank moves (1 = most vulnerable) ==")
    for f in FOLDS:
        rs = [R[s][f] for s in SAMPLERS]
        print(f"  {f:14s} hash/balanced/proportional ranks {rs}  shift {max(rs) - min(rs)}")


if __name__ == "__main__":
    main()
