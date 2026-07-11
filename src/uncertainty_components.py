#!/usr/bin/env python3
"""Seed-only, item-only, and crossed uncertainty for CodecFake+ recipe effects.

This is a sensitivity decomposition rather than an additive random-effects
model.  AUROC is nonlinear and seed-by-item interactions are retained, so the
three bootstrap variances must not be reported as independent variance shares.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from crossed_global_analysis import (
    CHUNK,
    FOLDS,
    REPS,
    RNG_SEED,
    SEEDS,
    exclusions,
    load_kernels,
    percentile_ci,
)


ROOT = Path(__file__).resolve().parents[1]


def describe(point: float, samples: np.ndarray):
    return {
        "point": float(point),
        "bootstrap_se": float(np.std(samples, ddof=1)),
        "ci95_percentile": percentile_ci(samples),
    }


def main():
    excluded = exclusions()
    kernels = []
    sizes = []
    points = []
    seed_points = []
    for fold in FOLDS:
        arms, npos, nneg = load_kernels(
            fold,
            [("558", "hash"), ("558", "source-balanced")],
            excluded,
        )
        delta = arms[1] - arms[0]
        kernels.append(delta)
        sizes.append((npos, nneg))
        points.append(float(np.mean(delta)))
        seed_points.append(np.mean(delta, axis=(1, 2)))

    point = np.asarray(points)
    seed_points = np.asarray(seed_points).T  # seed x fold
    seed_only = np.empty((REPS, len(FOLDS)))
    item_only = np.empty_like(seed_only)
    crossed = np.empty_like(seed_only)

    # Separate deterministic streams prevent one regime's random-number
    # consumption from affecting another.  The crossed stream and loop order
    # match crossed_global_analysis.py exactly.
    rng_seed = np.random.default_rng(RNG_SEED + 101)
    rng_item = np.random.default_rng(RNG_SEED + 202)
    rng_crossed = np.random.default_rng(RNG_SEED)
    seed_prob = np.full(len(SEEDS), 1.0 / len(SEEDS))

    for start in range(0, REPS, CHUNK):
        stop = min(start + CHUNK, REPS)
        batch = stop - start

        seed_w_only = rng_seed.multinomial(
            len(SEEDS), seed_prob, size=batch
        ) / len(SEEDS)
        seed_only[start:stop] = seed_w_only @ seed_points

        seed_w_crossed = rng_crossed.multinomial(
            len(SEEDS), seed_prob, size=batch
        ) / len(SEEDS)
        for fi, (kernel, (npos, nneg)) in enumerate(zip(kernels, sizes)):
            # Item-only: average all five seed kernels first, then resample one
            # shared empirical item distribution for the fold.
            pos_w = rng_item.multinomial(
                npos, np.full(npos, 1.0 / npos), size=batch
            ) / npos
            neg_w = rng_item.multinomial(
                nneg, np.full(nneg, 1.0 / nneg), size=batch
            ) / nneg
            seed_avg_fixed = np.mean(kernel, axis=0)
            pos_avg_fixed = pos_w @ seed_avg_fixed
            item_only[start:stop, fi] = np.einsum(
                "bn,bn->b", pos_avg_fixed, neg_w
            )

            # Fully crossed: common resampled seed IDs plus a class-stratified
            # item draw shared across the selected seeds and both arms.
            pos_w = rng_crossed.multinomial(
                npos, np.full(npos, 1.0 / npos), size=batch
            ) / npos
            neg_w = rng_crossed.multinomial(
                nneg, np.full(nneg, 1.0 / nneg), size=batch
            ) / nneg
            seed_avg = np.einsum(
                "bs,spn->bpn", seed_w_crossed, kernel, optimize=True
            )
            pos_avg = np.einsum("bp,bpn->bn", pos_w, seed_avg, optimize=True)
            crossed[start:stop, fi] = np.einsum("bn,bn->b", pos_avg, neg_w)

    result = {
        "method": {
            "bootstrap_replicates": REPS,
            "folds_are_fixed": True,
            "seed_only": "paired seed IDs resampled; all test items fixed",
            "item_only": "all five seeds fixed; class-stratified item draw shared across seeds and arms",
            "crossed": "paired seeds and class-stratified items resampled; both draws shared across arms",
            "macro_seed_draw_shared_across_folds": True,
            "macro_item_draws_independent_across_disjoint_folds": True,
            "interpretation": "sensitivity decomposition, not additive variance shares",
        },
        "folds": {},
    }

    for fi, fold in enumerate(FOLDS):
        seed_summary = describe(point[fi], seed_only[:, fi])
        item_summary = describe(point[fi], item_only[:, fi])
        crossed_summary = describe(point[fi], crossed[:, fi])
        seed_se = seed_summary["bootstrap_se"]
        item_se = item_summary["bootstrap_se"]
        crossed_se = crossed_summary["bootstrap_se"]
        result["folds"][fold] = {
            "seed_only": seed_summary,
            "item_only": item_summary,
            "crossed": crossed_summary,
            "se_ratios": {
                "item_over_seed": item_se / seed_se,
                "crossed_over_seed": crossed_se / seed_se,
                "crossed_over_item": crossed_se / item_se,
            },
            "item_se_exceeds_seed_se": item_se > seed_se,
        }

    result["macro_all_8_fixed_sources"] = {}
    macro_point = float(np.mean(point))
    for name, samples in (
        ("seed_only", np.mean(seed_only, axis=1)),
        ("item_only", np.mean(item_only, axis=1)),
        ("crossed", np.mean(crossed, axis=1)),
    ):
        result["macro_all_8_fixed_sources"][name] = describe(macro_point, samples)
    seed_se = result["macro_all_8_fixed_sources"]["seed_only"]["bootstrap_se"]
    item_se = result["macro_all_8_fixed_sources"]["item_only"]["bootstrap_se"]
    crossed_se = result["macro_all_8_fixed_sources"]["crossed"]["bootstrap_se"]
    result["macro_all_8_fixed_sources"]["se_ratios"] = {
        "item_over_seed": item_se / seed_se,
        "crossed_over_seed": crossed_se / seed_se,
        "crossed_over_item": crossed_se / item_se,
    }
    result["macro_all_8_fixed_sources"]["item_se_exceeds_seed_se"] = item_se > seed_se

    output = ROOT / "outputs" / "uncertainty_components_200k.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
