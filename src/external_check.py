"""External-detector check (paper Sec. IV): inference-only ASVspoof2019-LA-trained
wav2vec2/AASIST scored on the same leak-free CoSG rows -> pooled EER ~28.9%.

Confirms the CoSG difficulty is real and not specific to the paper's instruments.
Reads data/external_anchor/scores.jsonl (sanitized per-utterance scores).
"""
from __future__ import annotations
import json
import numpy as np

from audit_lib import ROOT, load_exclude, eer, auroc

SRC = ROOT / "data/external_anchor/scores.jsonl"


def main():
    excl = load_exclude()
    labels, scores = [], []
    for ln in SRC.open():
        ln = ln.strip()
        if not ln:
            continue
        o = json.loads(ln)
        if str(o.get("utterance_id", "")) in excl:
            continue
        labels.append(1 if o.get("label") == "spoof" else 0)
        scores.append(float(o["score"]))
    lab, sc = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    print(f"external LA-trained wav2vec2/AASIST on CoSG ({len(lab)} rows, leak-free):")
    print(f"  pooled EER  {eer(lab, sc) * 100:.1f}%")
    print(f"  pooled AUROC {auroc(lab, sc):.4f}")


if __name__ == "__main__":
    main()
