"""Operating-point transfer baseline (paper Sec. VII) + per-fold DET points.

For each corpus and each SEED separately (thresholds do not transfer across
differently calibrated models), find the REFERENCE fold's EER threshold on that
seed's scores, transfer it to every other fold of the same seed, and take the
balanced-accuracy gap vs that fold's own oracle threshold; then average gaps
over seeds. CodecFake+ uses the leak-pruned full_budget_loso scores (10 seeds);
cross-corpus re-slices use their full blocks (5 seeds).

Writes outputs/operating_points.csv (per-fold seed-mean oracle/transferred BAcc
+ gap) and outputs/det_points/<corpus>__<fold>.csv (FPR/FNR, pooled for plotting).
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np

from audit_lib import load, load_exclude, auroc, NON_MASK, ROOT
import crosscorpus_lib as cc

OUTDIR = ROOT / "outputs"
DETDIR = OUTDIR / "det_points"


def eer_threshold(spoof, bona):
    cand = np.unique(np.concatenate([spoof, bona]))
    ns, nb = max(len(spoof), 1), max(len(bona), 1)
    best_t, best_gap = 0.5, 9.9
    for t in cand:
        far = float((bona >= t).sum()) / nb
        miss = float((spoof < t).sum()) / ns
        if abs(far - miss) < best_gap:
            best_gap, best_t = abs(far - miss), float(t)
    return best_t


def bacc(spoof, bona, t):
    ns, nb = max(len(spoof), 1), max(len(bona), 1)
    tpr = 1 - float((spoof < t).sum()) / ns
    tnr = 1 - float((bona >= t).sum()) / nb
    return 0.5 * (tpr + tnr)


def best_bacc(spoof, bona):
    cand = np.unique(np.concatenate([spoof, bona]))
    return max(bacc(spoof, bona, float(t)) for t in cand)


def det_points(spoof, bona):
    cand = np.unique(np.concatenate([spoof, bona]))
    ns, nb = max(len(spoof), 1), max(len(bona), 1)
    rows = []
    for t in cand:
        rows.append((float((bona >= t).sum()) / nb, float((spoof < t).sum()) / ns, float(t)))
    return rows


def by_seed_codecfake(items):
    """fold -> {seed: (spoof_scores, bona_scores)}."""
    out = {}
    for it in items:
        if it.experiment != "full_budget_loso":
            continue
        out.setdefault(it.fold, {})[it.seed] = (it.scores[it.labels == 1],
                                                it.scores[it.labels == 0])
    return out


def by_seed_cross(corpus):
    out = {}
    for fold, d in cc.load(corpus, "full").items():
        for seed, (labels, scores) in d.items():
            out.setdefault(fold, {})[seed] = (scores[labels == 1], scores[labels == 0])
    return out


def main():
    OUTDIR.mkdir(exist_ok=True)
    DETDIR.mkdir(exist_ok=True)
    excl = load_exclude()
    items = load(excl)
    corpora = {
        "CodecFake+": (by_seed_codecfake(items), "MASKGCT"),
        "asvspoof2019la": (by_seed_cross("asvspoof2019la"), cc.REFERENCE["asvspoof2019la"]),
        "asvspoof5": (by_seed_cross("asvspoof5"), cc.REFERENCE["asvspoof5"]),
        "mlaad": (by_seed_cross("mlaad"), cc.REFERENCE["mlaad"]),
    }
    all_rows = []
    print(f"{'corpus':14s} {'fold':14s} {'AUROC':>6} {'BAcc_oracle':>11} {'BAcc_xfer':>9} {'gap':>6}")
    for corpus, (folds, ref) in corpora.items():
        if ref not in folds:
            print(f"{corpus}: reference {ref} missing"); continue
        ref_t_by_seed = {s: eer_threshold(sp, bo) for s, (sp, bo) in folds[ref].items()}
        fold_gaps = []
        for fold in sorted(folds):
            # pooled DET points (plotting only)
            sp_all = np.concatenate([sp for sp, _ in folds[fold].values()])
            bo_all = np.concatenate([bo for _, bo in folds[fold].values()])
            with (DETDIR / f"{corpus}__{fold}.csv").open("w", newline="") as w:
                cw = csv.writer(w); cw.writerow(["fpr", "fnr", "threshold"])
                cw.writerows(det_points(sp_all, bo_all))
            if fold == ref:
                continue
            seeds = sorted(set(folds[fold]) & set(ref_t_by_seed))
            aucs, orcs_eer, orcs_best, xfers = [], [], [], []
            for s in seeds:
                sp, bo = folds[fold][s]
                lab = np.concatenate([np.ones(len(sp), dtype=int), np.zeros(len(bo), dtype=int)])
                aucs.append(auroc(lab, np.concatenate([sp, bo])))
                # symmetric oracle: the fold's OWN EER threshold (same estimator as
                # the transferred one, so the gap isolates threshold mis-transfer
                # and is not inflated by a max-BAcc oracle overfit to small folds)
                orcs_eer.append(bacc(sp, bo, eer_threshold(sp, bo)))
                orcs_best.append(best_bacc(sp, bo))
                xfers.append(bacc(sp, bo, ref_t_by_seed[s]))
            a, orc, orcb, xfer = map(lambda v: float(np.mean(v)),
                                     (aucs, orcs_eer, orcs_best, xfers))
            fold_gaps.append(orc - xfer)
            all_rows.append([corpus, fold, round(a, 4), round(orc, 4), round(orcb, 4),
                             round(xfer, 4), round(orc - xfer, 4), len(seeds)])
            print(f"{corpus:14s} {fold:14s} {a:>6.3f} {orc:>11.3f} {xfer:>9.3f} {orc - xfer:>6.3f}")
        print(f"{corpus:14s} {'MEAN':14s} {'':>6} {'':>11} {'':>9} {np.mean(fold_gaps):>6.3f}\n")
    with (OUTDIR / "operating_points.csv").open("w", newline="") as w:
        cw = csv.writer(w)
        cw.writerow(["corpus", "fold", "auroc", "bacc_oracle_eer", "bacc_oracle_best", "bacc_transferred", "gap_vs_oracle_eer", "n_seeds"])
        cw.writerows(all_rows)
    print(f"wrote {OUTDIR / 'operating_points.csv'} and DET curves under {DETDIR}/")


if __name__ == "__main__":
    main()
