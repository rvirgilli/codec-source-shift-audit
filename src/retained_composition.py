"""Retained-source composition under each sampler (paper Sec. IV).

Shows that the naive (hash) and source-proportional samplers both leave the
dominant MASKGCT source at ~two-thirds of the retained training spoof, while the
source-balanced sampler cuts it several-fold. This is why proportional's rank
moves are within the seed-noise floor (its composition ~= hash) while the
balanced sampler is the real composition change.

Pool definition: each held-out fold's ACTUAL train_sources from
data/source_holdout_split.json (the full pool of all non-held-out CoSG sources,
including the small ineligible ones), subsampled to the MRD spoof budget (307)
by the released recipe_sampler. Reports the mean and range of the retained
MASKGCT share over the 8 non-MASKGCT held-out folds, per sampler.
"""
from __future__ import annotations
import json
from collections import Counter
import statistics as st
from pathlib import Path

from recipe_sampler import subsample_spoof_rows

ROOT = Path(__file__).resolve().parents[1]
SPOOF_BUDGET = 307  # MRD: 558 train - 251 bona-fide
SAMPLERS = ["hash", "source-balanced", "source-proportional"]


def main():
    plan = json.load(open(ROOT / "data/source_holdout_split.json"))
    sl = plan["source_labels"]
    res = {s: [] for s in SAMPLERS}
    for f in plan["folds"]:
        ho = f["heldout_source"]
        if ho == "MASKGCT":
            continue
        rows = [{"utterance_id": f"{s}_{i}", "source_model": s}
                for s in f["train_sources"] for i in range(sl[s]["spoof"])]
        for samp in SAMPLERS:
            kept = subsample_spoof_rows(rows, target=SPOOF_BUDGET, seed=7,
                                        heldout=ho, sampler=samp)
            share = Counter(r["source_model"] for r in kept)["MASKGCT"] / len(kept)
            res[samp].append(share)
    print("== Retained MASKGCT share of the training spoof set (8 non-MASKGCT folds) ==")
    for s in SAMPLERS:
        v = res[s]
        print(f"  {s:20s} mean {st.mean(v):.0%}   range {min(v):.0%}-{max(v):.0%}")
    print("  (hash ~= proportional keep the dominant source; balanced cuts it ~5x)")


if __name__ == "__main__":
    main()
