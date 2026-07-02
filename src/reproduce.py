#!/usr/bin/env python3
"""Reproduce the paper's claim-bearing numbers from the bundled per-utterance scores.

Regenerates:
  - Sec. IV  : full-budget vs Matched Remaining-Data (MRD) source-holdout AUROCs,
               95% CIs, and bootstrap vulnerability-rank intervals; non-MASKGCT means.
  - Sec. VII : exact-codec proxy recovery (SimpleSpeech->SQ-Codec, NaturalSpeech3->FACodec).
  - Stats (A): hierarchical seed+utterance bootstrap of the SS2 full->MRD AUROC drop.
  - Stats (B): paired-by-seed SIMPLESPEECH2 vs SIMPLESPEECH1 under MRD.

Usage:  python src/reproduce.py [--reps 2000]
Outputs printed to stdout and written to outputs/.
"""
from __future__ import annotations
import argparse, csv, json, random
from pathlib import Path
import numpy as np
import audit_lib as A

OUT = A.ROOT / "outputs"
EXACT = {"SIMPLESPEECH1": "cors_sqcodec", "SIMPLESPEECH2": "cors_sqcodec", "NS3": "cors_facodec"}
ANCHORS = {"cors_facodec": "FACodec", "cors_sqcodec": "SQ-Codec", "cors_vocos": "Vocos"}


def slice_auroc(items, anchor_exp, fold):
    """AUROC of one source's spoof rows vs all bona-fide, in a CoRS anchor, mean over seeds."""
    vals = []
    for it in items:
        if it.experiment != anchor_exp:
            continue
        mask = np.array([(s == fold and lab == 1) or lab == 0
                         for s, lab in zip(it.sources, it.labels)])
        if mask.sum() < 3 or len(set(it.labels[mask].tolist())) < 2:
            continue
        vals.append(A.auroc(it.labels[mask], it.scores[mask]))
    return float(np.mean(vals)) if vals else None


def per_seed_auroc(items, exp, fold):
    return {it.seed: A.auroc(it.labels, it.scores)
            for it in items if it.experiment == exp and it.fold == fold}


def hierarchical_delta(items, full_exp, mrd_exp, fold, reps, rng):
    full = {it.seed: it for it in items if it.experiment == full_exp and it.fold == fold}
    mrd = {it.seed: it for it in items if it.experiment == mrd_exp and it.fold == fold}
    seeds = sorted(set(full) & set(mrd))
    data = {}
    for s in seeds:
        fm = {u: (l, sc) for u, l, sc in zip(full[s].utts, full[s].labels, full[s].scores)}
        mm = {u: (l, sc) for u, l, sc in zip(mrd[s].utts, mrd[s].labels, mrd[s].scores)}
        common = sorted(set(fm) & set(mm))
        lab = np.array([fm[u][0] for u in common])
        data[s] = (lab, np.array([fm[u][1] for u in common]), np.array([mm[u][1] for u in common]),
                   np.where(lab == 1)[0], np.where(lab == 0)[0])
    deltas = []
    for _ in range(reps):
        drawn = [seeds[rng.randrange(len(seeds))] for _ in seeds]
        fa, ma = [], []
        for s in drawn:
            lab, fsc, msc, pos, neg = data[s]
            if len(pos) < 2 or len(neg) < 2:
                continue
            idx = np.concatenate([np.array([pos[rng.randrange(len(pos))] for _ in range(len(pos))]),
                                  np.array([neg[rng.randrange(len(neg))] for _ in range(len(neg))])])
            L = lab[idx]
            fa.append(A.auroc(L, fsc[idx])); ma.append(A.auroc(L, msc[idx]))
        if fa and ma:
            deltas.append(np.mean(fa) - np.mean(ma))
    arr = np.asarray(deltas)
    point = (np.mean([A.auroc(data[s][0], data[s][1]) for s in seeds])
             - np.mean([A.auroc(data[s][0], data[s][2]) for s in seeds]))
    return round(float(point), 4), [round(float(np.percentile(arr, 2.5)), 4),
                                     round(float(np.percentile(arr, 97.5)), 4)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    excl = A.load_exclude()
    items = A.load(exclude=excl)
    rng = random.Random(17062026)
    print(f"loaded {len(items)} leak-free score sets ({len(excl)} excluded utt-ids)\n")

    # ---- Table I ----
    ri_full = A.rank_intervals(items, "full_budget_loso", args.reps)
    ri_mrd = A.rank_intervals(items, "budget_matched_loso", args.reps)

    def ci(ri, fold):
        v = ri.get(fold)
        return f"{v[4]:.3f}-{v[5]:.3f}" if v else None

    def rk(ri, fold):
        v = ri.get(fold)
        return f"{v[0]:.0f} [{v[1]:.0f}-{v[2]:.0f}]" if v else None

    rows = []
    for fold in A.FOLDS:
        full = A.fold_mean_auroc(items, "full_budget_loso", fold)
        mrd = A.fold_mean_auroc(items, "budget_matched_loso", fold)
        r = {"fold": fold,
             "auroc_full": round(full, 3) if full else None,
             "ci_full": ci(ri_full, fold),
             "auroc_mrd": round(mrd, 3) if mrd else None,
             "ci_mrd": ci(ri_mrd, fold),
             "delta": round(full - mrd, 3) if (full and mrd) else None,
             "rank_full": rk(ri_full, fold),
             "rank_mrd": rk(ri_mrd, fold)}
        rows.append(r)
    nm_full = np.mean([r["auroc_full"] for r in rows if r["fold"] != "MASKGCT" and r["auroc_full"]])
    nm_mrd = np.mean([r["auroc_mrd"] for r in rows if r["fold"] != "MASKGCT" and r["auroc_mrd"]])

    print("== Source-holdout AUROC, full vs MRD (leak-free; paper Sec. IV) ==")
    print(f"{'fold':14} {'full':>6} {'CI_full':>13} {'MRD':>6} {'CI_mrd':>13} {'delta':>6} {'rank_full':>11} {'rank_mrd':>11}")
    for r in rows:
        print(f"{r['fold']:14} {str(r['auroc_full']):>6} {str(r['ci_full']):>13} {str(r['auroc_mrd']):>6} "
              f"{str(r['ci_mrd']):>13} {str(r['delta']):>6} {str(r['rank_full']):>11} {str(r['rank_mrd']):>11}")
    print(f"non-MASKGCT mean: full {nm_full:.3f}  MRD {nm_mrd:.3f}")
    # Two collapse definitions (the paper's Sec. IV number is the gated one):
    #   gated  = mean over below-ceiling (full AUROC < 0.95) non-reference folds
    #   all    = mean over all 8 non-MASKGCT folds (includes NS2, above ceiling)
    deltas_all = [r["delta"] for r in rows if r["fold"] != "MASKGCT" and r["delta"] is not None]
    deltas_gated = [r["delta"] for r in rows
                    if r["fold"] != "MASKGCT" and r["delta"] is not None and r["auroc_full"] < 0.95]
    print(f"mean full->MRD collapse: below-ceiling-gated {np.mean(deltas_gated):.3f} "
          f"({len(deltas_gated)} folds; paper Sec. IV)  |  all-non-MASKGCT {np.mean(deltas_all):.3f} "
          f"({len(deltas_all)} folds)\n")

    # ---- Table IV: exact-codec proxy recovery ----
    print("== Exact-codec proxy recovery (paper Sec. VII) ==")
    print(f"{'fold':14} {'codec':9} {'full':>6} {'MRD':>6} {'exact_proxy':>11} {'recovers':>8}")
    t4 = []
    for fold, anchor in EXACT.items():
        full = A.fold_mean_auroc(items, "full_budget_loso", fold)
        mrd = A.fold_mean_auroc(items, "budget_matched_loso", fold)
        proxy = slice_auroc(items, anchor, fold)
        rec = "yes" if (proxy and full and proxy >= 0.95 * full) else "no"
        t4.append({"fold": fold, "anchor": ANCHORS[anchor], "full": round(full, 3),
                   "mrd": round(mrd, 3), "exact_proxy": round(proxy, 3), "recovers": rec})
        print(f"{fold:14} {ANCHORS[anchor]:9} {full:6.3f} {mrd:6.3f} {proxy:11.3f} {rec:>8}")
    print()

    # ---- Stats (A): hierarchical bootstrap of SS2 full->MRD drop ----
    a_x = hierarchical_delta(items, "full_budget_loso", "budget_matched_loso", "SIMPLESPEECH2", args.reps, rng)
    a_w = hierarchical_delta(items, "wavlm_frozen_loso", "wavlm_budget_matched_loso", "SIMPLESPEECH2", args.reps, rng)
    print("== Stats (A): hierarchical seed+utterance bootstrap, SS2 full->MRD AUROC drop ==")
    print(f"  XLS-R: delta {a_x[0]}  95% CI {a_x[1]}")
    print(f"  WavLM: delta {a_w[0]}  95% CI {a_w[1]}\n")

    # ---- Stats (B): paired SS2 vs SS1 under MRD ----
    s2 = per_seed_auroc(items, "budget_matched_loso", "SIMPLESPEECH2")
    s1 = per_seed_auroc(items, "budget_matched_loso", "SIMPLESPEECH1")
    seeds = sorted(set(s2) & set(s1))
    d = np.array([s2[s] - s1[s] for s in seeds])
    brng = random.Random(17062027)
    boot = [float(np.mean([d[brng.randrange(len(d))] for _ in d])) for _ in range(args.reps)]
    print("== Stats (B): paired-by-seed SS2 vs SS1 (MRD) ==")
    print(f"  SS2 below SS1 in {int((d < 0).sum())}/{len(seeds)} seeds; "
          f"paired mean SS2-SS1 = {d.mean():.3f}  95% CI [{np.percentile(boot,2.5):.3f}, {np.percentile(boot,97.5):.3f}]")
    print(f"  means: SS2 {np.mean(list(s2.values())):.3f}  SS1 {np.mean(list(s1.values())):.3f}\n")

    # write machine-readable outputs
    with (OUT / "table1_source_holdout.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with (OUT / "table4_proxy_recovery.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(t4[0].keys())); w.writeheader(); w.writerows(t4)
    (OUT / "stats_rigor.json").write_text(json.dumps(
        {"non_mask_mean": {"full": round(float(nm_full), 3), "mrd": round(float(nm_mrd), 3)},
         "A_hierarchical_ss2_drop": {"xlsr": {"delta": a_x[0], "ci95": a_x[1]},
                                     "wavlm": {"delta": a_w[0], "ci95": a_w[1]}},
         "B_paired_ss2_vs_ss1": {"n_ss2_below_ss1": int((d < 0).sum()), "n_seeds": len(seeds),
                                 "paired_mean": round(float(d.mean()), 4)}}, indent=2) + "\n")
    print("wrote outputs/ (table1_source_holdout.csv, table4_proxy_recovery.csv, stats_rigor.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
