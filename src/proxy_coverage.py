"""Proxy-codec coverage matrix (paper Sec. VII).

For each CoRS proxy anchor (FACodec, SQ-Codec, Vocos, Spectral Codecs), compute
the AUROC of its pooled-CoSG scoring restricted to each SimpleSpeech slice
(spoof rows of that source vs all bona-fide rows), averaged over seeds.

Reproduces: FACodec 0.719 > SQ-Codec 0.691 on the SS2 slice (SQ-Codec is not
even the best SS2 proxy despite matching the generator's codec), and the second
architecturally-distinct scalar quantizer failing to recover SimpleSpeech
(Spectral Codecs: SS1 0.096, SS2 0.449).
"""
from __future__ import annotations
import numpy as np

from audit_lib import load, load_exclude, auroc

PROXIES = ["cors_facodec", "cors_sqcodec", "cors_vocos", "cors_spectralcodecs"]
SLICES = ["SIMPLESPEECH1", "SIMPLESPEECH2"]


def slice_auroc(it, source):
    mask = np.array([(lab == 0) or (src == source)
                     for lab, src in zip(it.labels, it.sources)])
    lab, sc = it.labels[mask], it.scores[mask]
    if lab.sum() == 0 or (lab == 0).sum() == 0:
        return None
    return auroc(lab, sc)


ALL_FOLDS = ["SIMPLESPEECH1", "SIMPLESPEECH2", "VALLE", "UNIAUDIO",
             "NS2", "NS3", "CLAMTTS", "GPST", "MASKGCT"]


def main():
    items = load(load_exclude())
    print("== CoRS proxy x SimpleSpeech-slice AUROC (seed means, pooled-CoSG scoring) ==")
    print(f"{'proxy':22s}" + "".join(f"{s:>16}" for s in SLICES) + f"{'pooled':>10}")
    for proxy in PROXIES:
        sets = [it for it in items if it.experiment == proxy]
        if not sets:
            print(f"{proxy:22s}  (no data)")
            continue
        cells = []
        for s in SLICES:
            vals = [v for it in sets if (v := slice_auroc(it, s)) is not None]
            cells.append(float(np.mean(vals)) if vals else float("nan"))
        pooled = float(np.mean([auroc(it.labels, it.scores) for it in sets]))
        print(f"{proxy:22s}" + "".join(f"{c:>16.3f}" for c in cells) + f"{pooled:>10.3f}"
              + f"   ({len(sets)} seeds)")

    # full fold x proxy matrix: verifies "multi-VQ generators are codec-recovered
    # while both SimpleSpeech folds are not" (paper Sec. VII) from released data.
    # recovery rule matches reproduce.py: best-proxy slice >= 0.95 x full-LOSO.
    print("\n== Full fold x proxy slice matrix (seed means) ==")
    from audit_lib import fold_mean_auroc
    print(f"{'fold':14s}" + "".join(f"{p.replace('cors_', ''):>16}" for p in PROXIES)
          + f"{'best':>8}{'full':>8}{'recovered':>10}")
    for fold in ALL_FOLDS:
        row = []
        for proxy in PROXIES:
            sets = [it for it in items if it.experiment == proxy]
            vals = [v for it in sets if (v := slice_auroc(it, fold)) is not None]
            row.append(float(np.mean(vals)) if vals else float("nan"))
        best = np.nanmax(row) if row else float("nan")
        full = fold_mean_auroc(items, "full_budget_loso", fold)
        rec = "yes" if (full and best >= 0.95 * full) else "no"
        print(f"{fold:14s}" + "".join(f"{c:>16.3f}" for c in row)
              + f"{best:>8.3f}{full:>8.3f}{rec:>10}")


if __name__ == "__main__":
    main()
