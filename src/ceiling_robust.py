"""Ceiling-robust multi-corpus scoping + the naive-metric trap (paper Sec. VI).

Reproduces:
  - Table II: per-corpus reference AUROC, #below-ceiling folds (full-budget
    AUROC < 0.95), testability (>= 3 below-ceiling folds).
  - The below-ceiling-gated mean full->MRD collapse on CodecFake+ (0.146; the
    unconditional all-8-fold mean 0.142 is printed alongside for clarity).
  - The metric trap: on the saturated ASVspoof2019-LA re-slice the naive
    Kendall-tau criterion (tau < 0.5 => "ranking moves") fires although every
    fold is at ceiling, whereas the ceiling-robust gate returns a null.
  - Ceiling sensitivity: testability verdicts across ceilings 0.90-0.98.

Cross-corpus data are custom diagnostic re-slices, NOT official protocols.
"""
from __future__ import annotations
import itertools
import numpy as np

from audit_lib import load, load_exclude, auroc, by, fold_mean_auroc, NON_MASK
import crosscorpus_lib as cc

CEIL = 0.95
MIN_BELOW = 3
CEILINGS = [0.90, 0.92, 0.95, 0.97, 0.98]


def kendall_tau(a, b):
    n = len(a); c = d = 0
    for i, j in itertools.combinations(range(n), 2):
        s = np.sign((a[i] - a[j]) * (b[i] - b[j]))
        if s > 0: c += 1
        elif s < 0: d += 1
    return (c - d) / (0.5 * n * (n - 1))


def codecfake_folds(items):
    full = {f: fold_mean_auroc(items, "full_budget_loso", f)
            for f in NON_MASK + ["MASKGCT"]}
    mrd = {f: fold_mean_auroc(items, "budget_matched_loso", f) for f in NON_MASK}
    return full, mrd


def corpus_row(name, full, mrd, ref):
    below = [f for f, v in full.items() if v is not None and v < CEIL]
    below_nonref = [f for f in below if f != ref and mrd.get(f) is not None]
    collapses = [full[f] - mrd[f] for f in below_nonref]
    row = {
        "corpus": name, "reference": ref,
        "ref_full_auroc": round(full[ref], 3) if full.get(ref) is not None else None,
        "n_below_ceiling": len(below), "below_ceiling_folds": sorted(below),
        "testable": len(below) >= MIN_BELOW,
        "mean_collapse_below_ceiling": round(float(np.mean(collapses)), 3) if collapses else None,
    }
    all_nonref = [f for f in full if f != ref and mrd.get(f) is not None]
    all_collapses = [full[f] - mrd[f] for f in all_nonref]
    row["mean_collapse_all_nonref"] = round(float(np.mean(all_collapses)), 3) if all_collapses else None
    return row


def main():
    excl = load_exclude()
    items = load(excl)

    print("== Table II: ceiling-robust scoping (ceiling 0.95, testable >= 3 below) ==")
    rows = []

    cf_full, cf_mrd = codecfake_folds(items)
    rows.append(corpus_row("CodecFake+ (CoSG)", cf_full, cf_mrd, "MASKGCT"))

    cross_fold_aurocs = {}
    for corpus in ("asvspoof2019la", "asvspoof5", "mlaad"):
        full = cc.seed_mean_auroc(cc.load(corpus, "full"), auroc)
        mrd = cc.seed_mean_auroc(cc.load(corpus, "mrd"), auroc)
        cross_fold_aurocs[corpus] = (full, mrd)
        rows.append(corpus_row(corpus, full, mrd, cc.REFERENCE[corpus]))

    for r in rows:
        print(f"  {r['corpus']:20s} ref {r['reference']:8s} {r['ref_full_auroc']}"
              f"  below-ceiling {r['n_below_ceiling']}  testable {'yes' if r['testable'] else 'no'}")
    cf = rows[0]
    print(f"\nCodecFake+ mean full->MRD collapse: below-ceiling-gated {cf['mean_collapse_below_ceiling']}"
          f" (paper Sec. IV); unconditional all-8-non-reference {cf['mean_collapse_all_nonref']}")

    # metric trap on the saturated LA re-slice
    la_full, la_mrd = cross_fold_aurocs["asvspoof2019la"]
    common = sorted(set(la_full) & set(la_mrd))
    tau = kendall_tau([la_full[f] for f in common], [la_mrd[f] for f in common])
    n_below = sum(1 for v in la_full.values() if v < CEIL)
    naive_fires = tau < 0.5
    print("\n== Metric trap (ASVspoof2019-LA re-slice) ==")
    print(f"  every fold full-budget AUROC >= {min(la_full.values()):.3f}; "
          f"naive full-vs-MRD Kendall tau = {tau:+.3f} -> naive criterion (tau<0.5) "
          f"{'FIRES (false positive)' if naive_fires else 'does not fire'}")
    print(f"  ceiling-robust gate: {n_below} below-ceiling folds (<{MIN_BELOW}) -> correctly NOT testable")

    # ceiling sensitivity
    print("\n== Ceiling sensitivity (testable at ceiling c?) ==")
    print(f"{'corpus':20s}" + "".join(f"  c={c:<5}" for c in CEILINGS))
    for r, (name, full) in zip(rows, [("CodecFake+ (CoSG)", cf_full)] +
                               [(c, cross_fold_aurocs[c][0]) for c in ("asvspoof2019la", "asvspoof5", "mlaad")]):
        counts = [sum(1 for v in full.values() if v is not None and v < c) for c in CEILINGS]
        print(f"{name:20s}" + "".join(f"  {n:>2}{'T' if n >= MIN_BELOW else '-'}   " for n in counts))
    print("(verdicts are identical across ceilings 0.90-0.98: only CodecFake+ is testable)")


if __name__ == "__main__":
    main()
