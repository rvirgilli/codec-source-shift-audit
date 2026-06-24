#!/usr/bin/env python3
"""Regenerate Figure 1 (4-panel SS2 diagnostic trajectory) and Figure 2 (label-wise
score distributions) from the bundled per-utterance scores. Output: outputs/*.pdf.

Usage:  python src/figures.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # avoid Type-3 fonts (IEEE requirement)
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import audit_lib as A

OUT = A.ROOT / "outputs"
XL_F, XL_M = "full_budget_loso", "budget_matched_loso"
WV_F, WV_M = "wavlm_frozen_loso", "wavlm_budget_matched_loso"


def per_seed_auroc(items, exp, fold):
    return {it.seed: A.auroc(it.labels, it.scores) for it in items if it.experiment == exp and it.fold == fold}


def per_seed_gap(items, exp, fold):
    return {it.seed: A.median_gap(it.labels, it.scores) for it in items if it.experiment == exp and it.fold == fold}


def slice_per_seed(items, anchor_exp, fold):
    out = {}
    for it in items:
        if it.experiment != anchor_exp:
            continue
        m = np.array([(s == fold and lab == 1) or lab == 0 for s, lab in zip(it.sources, it.labels)])
        if m.sum() < 3 or len(set(it.labels[m].tolist())) < 2:
            continue
        out[it.seed] = A.auroc(it.labels[m], it.scores[m])
    return out


def pool(items, exp, fold):
    bona, spoof = [], []
    for it in items:
        if it.experiment == exp and it.fold == fold:
            bona.append(it.scores[it.labels == 0]); spoof.append(it.scores[it.labels == 1])
    return np.concatenate(bona), np.concatenate(spoof)


def fig1(items):
    fm = {f: A.fold_mean_auroc(items, XL_F, f) for f in A.FOLDS}
    fm = {k: v for k, v in sorted(fm.items(), key=lambda x: x[1])}
    xf = per_seed_auroc(items, XL_F, "SIMPLESPEECH2"); xm = per_seed_auroc(items, XL_M, "SIMPLESPEECH2")
    wf = per_seed_auroc(items, WV_F, "SIMPLESPEECH2"); wm = per_seed_auroc(items, WV_M, "SIMPLESPEECH2")

    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5})
    fig, ax = plt.subplots(2, 2, figsize=(7.16, 4.5))

    a = ax[0, 0]
    cols = ["#c44e52" if f == "SIMPLESPEECH2" else ("#dddddd" if f == "MASKGCT" else "#bbbbbb") for f in fm]
    bars = a.bar(range(len(fm)), list(fm.values()), color=cols, edgecolor="black", linewidth=0.5)
    bars[list(fm).index("SIMPLESPEECH2")].set_hatch("//")
    a.set_xticks(range(len(fm)))
    a.set_xticklabels([f.replace("SIMPLESPEECH", "SS") for f in fm], rotation=45, ha="right", fontsize=7)
    a.set_ylim(0.5, 1.0); a.set_ylabel("AUROC"); a.set_title("A. Naive full-budget XLS-R LOSO ranking")
    rk = list(fm).index("SIMPLESPEECH2") + 1
    a.annotate(f"SS2: AUROC {fm['SIMPLESPEECH2']:.3f},\nrank {rk}/{len(fm)} (lower-middle)",
               xy=(rk - 1, fm["SIMPLESPEECH2"]), xytext=(1.5, 0.62), fontsize=7, ha="left",
               arrowprops=dict(arrowstyle="->", lw=0.7))

    b = ax[0, 1]
    for d in (xf, xm):
        pass
    seeds = sorted(set(xf) & set(xm))
    for s in seeds:
        b.plot([0, 1], [xf[s], xm[s]], color="#9ecae1", lw=0.6, alpha=0.8, zorder=1)
    for s in sorted(set(wf) & set(wm)):
        b.plot([0, 1], [wf[s], wm[s]], color="#d9b3b3", lw=0.6, alpha=0.8, zorder=1)
    b.plot([0, 1], [np.mean(list(xf.values())), np.mean(list(xm.values()))], "-o", color="#1f4e79", lw=2, ms=6, label="XLS-R (mean)")
    b.plot([0, 1], [np.mean(list(wf.values())), np.mean(list(wm.values()))], "--s", color="#7a1f1f", lw=2, ms=6, label="WavLM (mean)")
    b.set_xticks([0, 1]); b.set_xticklabels(["Full", "MRD"]); b.set_xlim(-0.4, 1.4); b.set_ylim(0.4, 1.0)
    b.set_ylabel("SS2 AUROC"); b.set_title("B. Full-to-MRD collapse (gray = seeds)"); b.legend(fontsize=7, loc="lower left")

    c = ax[1, 0]
    names = [("FACodec", "cors_facodec"), ("SQ-Codec", "cors_sqcodec"), ("Vocos", "cors_vocos")]
    means, sds = [], []
    for _, exp in names:
        v = list(slice_per_seed(items, exp, "SIMPLESPEECH2").values())
        means.append(np.mean(v)); sds.append(np.std(v))
    c.bar(range(3), means, yerr=sds, color="#7fa8c9", edgecolor="black", linewidth=0.5, capsize=3)
    xfm, xmm = np.mean(list(xf.values())), np.mean(list(xm.values()))
    c.axhline(xfm, ls="--", color="#2e7d32", lw=1); c.axhline(xmm, ls="--", color="#c44e52", lw=1)
    c.text(2.4, xfm + 0.005, f"LOSO full {xfm:.3f}", color="#2e7d32", fontsize=6.5, ha="right")
    c.text(2.4, xmm - 0.03, f"LOSO MRD {xmm:.3f}", color="#c44e52", fontsize=6.5, ha="right")
    for i, m in enumerate(means):
        c.text(i, m + sds[i] + 0.01, f"{m:.3f}", ha="center", fontsize=7)
    c.set_xticks(range(3)); c.set_xticklabels([n for n, _ in names]); c.set_ylim(0.5, 1.0)
    c.set_ylabel("SS2-slice AUROC"); c.set_title("C. CoRS proxy SS2-slice AUROC")

    d = ax[1, 1]
    conds = [("XLS-R\nfull", XL_F), ("XLS-R\nMRD", XL_M), ("WavLM\nfull", WV_F), ("WavLM\nMRD", WV_M)]
    gm, gs = [], []
    for _, exp in conds:
        v = list(per_seed_gap(items, exp, "SIMPLESPEECH2").values())
        gm.append(np.mean(v)); gs.append(np.std(v))
    d.bar(range(4), gm, yerr=gs, color=["#4c4c4c", "#bdbdbd", "#4c4c4c", "#bdbdbd"], edgecolor="black", linewidth=0.5, capsize=3)
    d.axhline(0, color="black", lw=0.8)
    for i, m in enumerate(gm):
        d.text(i, m + gs[i] + 0.02, f"{m:.3f}", ha="center", fontsize=7)
    d.set_xticks(range(4)); d.set_xticklabels([n for n, _ in conds], fontsize=7); d.set_ylim(0, 1.0)
    d.set_ylabel("median spoof - bona-fide"); d.set_title("D. Label-wise gap (positive = compression)")

    fig.tight_layout(pad=0.6)
    fig.savefig(OUT / "fig1_diagnosis_panel.pdf", bbox_inches="tight")
    print("wrote outputs/fig1_diagnosis_panel.pdf")


def fig2(items):
    mb, ms = pool(items, WV_F, "MASKGCT")
    sb, ss = pool(items, XL_M, "SIMPLESPEECH2")
    plt.rcParams.update({"font.size": 9})
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    fig.suptitle("Label-wise score placement distinguishes displacement from compression", fontsize=9.5)
    bins = np.linspace(0, 1, 31)
    for a, (b, s, title) in zip(ax, [(mb, ms, "WavLM / MASKGCT full"),
                                     (sb, ss, "XLS-R / SIMPLESPEECH2 matched")]):
        gap = np.median(s) - np.median(b)
        a.hist(b, bins=bins, density=True, color="#a9c0d6", alpha=0.85, label="bona-fide")
        a.hist(s, bins=bins, density=True, histtype="step", color="#b03a3a", lw=1.6, label="spoof")
        a.axvline(np.median(b), color="#3a5a99", ls="--", lw=1.2)
        a.axvline(np.median(s), color="#b03a3a", ls="--", lw=1.2)
        a.set_title(f"{title}\npooled median gap={gap:.3f}", fontsize=8.5)
        a.set_xlabel("spoof score"); a.set_ylabel("density")
    ax[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "fig_labelwise_distributions.pdf", bbox_inches="tight")
    print("wrote outputs/fig_labelwise_distributions.pdf")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    it = A.load(exclude=A.load_exclude())
    fig1(it)
    fig2(it)
