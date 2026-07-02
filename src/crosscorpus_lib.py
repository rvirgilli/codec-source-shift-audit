"""Loader for the cross-corpus re-slice scores (release v1.2.0).

Layout: data/crosscorpus/<corpus>/<block>/<condition>/seed_<n>/<fold>.jsonl,
each line {utterance_id, label, source_model, score}.

corpus    : asvspoof2019la | asvspoof5 | mlaad
block     : full | mrd | samp_hash | samp_source-balanced | samp_source-proportional
condition : xlsr_peft_adapter | wavlm_frozen_backend

These are CUSTOM diagnostic source-holdout re-slices (attack-system holdout for
the ASVspoof corpora, TTS-model holdout for MLAAD-en with LibriSpeech bona-fide),
NOT official protocols. No leak pruning applies: their bona-fide pools are
independent of the CodecFake+ corpus.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CROSS = ROOT / "data" / "crosscorpus"

REFERENCE = {"asvspoof2019la": "A07", "asvspoof5": "A17", "mlaad": "JENNY"}


def load(corpus: str, block: str, condition: str = "xlsr_peft_adapter", with_utts: bool = False):
    """Return {fold: {seed: (labels, scores)}} (or (labels, scores, utts))."""
    base = CROSS / corpus / block / condition
    out: dict = {}
    for f in sorted(base.glob("seed_*/*.jsonl")):
        fold = f.stem
        seed = int(f.parent.name.replace("seed_", ""))
        labels, scores, utts = [], [], []
        for ln in f.open():
            ln = ln.strip()
            if not ln:
                continue
            o = json.loads(ln)
            labels.append(1 if o.get("label") == "spoof" else 0)
            scores.append(float(o["score"]))
            utts.append(str(o.get("utterance_id", "")))
        if not scores or len(set(labels)) < 2:
            continue
        cell = (np.asarray(labels, dtype=int), np.asarray(scores, dtype=float))
        out.setdefault(fold, {})[seed] = cell + ((utts,) if with_utts else ())
    return out


def seed_mean_auroc(loaded, auroc_fn) -> dict[str, float]:
    return {fold: float(np.mean([auroc_fn(l, s) for l, s in d.values()]))
            for fold, d in loaded.items()}
