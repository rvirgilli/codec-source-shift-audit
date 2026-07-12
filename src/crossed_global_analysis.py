#!/usr/bin/env python3
"""Crossed seed-by-utterance inference for the CodecFake+ recipe scores.

This analysis treats the eight named CodecFake+ sources as fixed.  Within each
bootstrap replicate it resamples the five paired training-seed IDs once and
shares that draw across folds and arms.  It independently resamples spoof and
bona-fide utterance IDs within each fold, sharing those item draws across every
seed and arm.  AUROC is the tie-correct Mann--Whitney statistic.

The script reports per-fold balanced-minus-hash effects, fixed-source macro
effects, leave-one-source-out sensitivity, centered bootstrap max-T intervals,
pairwise effect heterogeneity, and the three-fold budget/composition factorial.
"""
from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TREE = ROOT / "data" / "factorial"
FOLDS = [
    "CLAMTTS",
    "GPST",
    "NS2",
    "UNIAUDIO",
    "VALLE",
    "SIMPLESPEECH1",
    "SIMPLESPEECH2",
    "NS3",
]
SEEDS = [7, 11, 17, 29, 31]
REPS = 200_000
CHUNK = 2_000
RNG_SEED = 20_260_711


def exclusions() -> set[str]:
    with (ROOT / "data" / "collision_details.csv").open() as handle:
        return {
            row["test_utterance_id"]
            for row in csv.DictReader(handle)
            if row["reference_partition"] == "train"
        }


def load_cell(fold: str, budget: str, arm: str, seed: int, excluded: set[str]):
    path = TREE / fold / f"budget_{budget}" / arm / f"seed_{seed}.jsonl"
    rows = {}
    with path.open() as handle:
        for line in handle:
            obj = json.loads(line)
            uid = str(obj["utterance_id"])
            if uid in excluded:
                continue
            rows[uid] = (obj["label"] == "spoof", float(obj["score"]))
    return rows


def mann_whitney_kernel(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    return (pos[:, None] > neg[None, :]).astype(np.float64) + 0.5 * (
        pos[:, None] == neg[None, :]
    )


def load_kernels(fold: str, cells: list[tuple[str, str]], excluded: set[str]):
    loaded = {
        (budget, arm, seed): load_cell(fold, budget, arm, seed, excluded)
        for budget, arm in cells
        for seed in SEEDS
    }
    reference = loaded[(cells[0][0], cells[0][1], SEEDS[0])]
    ref_ids = set(reference)
    for key, rows in loaded.items():
        if set(rows) != ref_ids:
            raise RuntimeError(f"unaligned test IDs for {fold} {key}")
        for uid in ref_ids:
            if rows[uid][0] != reference[uid][0]:
                raise RuntimeError(f"label mismatch for {fold} {key} {uid}")
    pos_ids = sorted(uid for uid, (label, _) in reference.items() if label)
    neg_ids = sorted(uid for uid, (label, _) in reference.items() if not label)
    kernels = np.empty((len(cells), len(SEEDS), len(pos_ids), len(neg_ids)))
    for ci, (budget, arm) in enumerate(cells):
        for si, seed in enumerate(SEEDS):
            rows = loaded[(budget, arm, seed)]
            pos = np.asarray([rows[uid][1] for uid in pos_ids])
            neg = np.asarray([rows[uid][1] for uid in neg_ids])
            kernels[ci, si] = mann_whitney_kernel(pos, neg)
    return kernels, len(pos_ids), len(neg_ids)


def bootstrap_kernel_contrasts(
    contrast_kernels: list[np.ndarray],
    item_sizes: list[tuple[int, int]],
    reps: int,
    rng: np.random.Generator,
):
    """Joint bootstrap for fold-level contrasts.

    Each contrast kernel has shape (seed, positive_item, negative_item).  The
    training-seed multinomial draw is common to every contrast.  Item draws are
    common wherever contrasts have the same fold, represented here by passing
    them in fold order and generating one pair of weights for each contrast.
    For the main recipe, there is one contrast per fold.
    """
    out = np.empty((reps, len(contrast_kernels)), dtype=np.float64)
    seed_prob = np.full(len(SEEDS), 1.0 / len(SEEDS))
    for start in range(0, reps, CHUNK):
        stop = min(start + CHUNK, reps)
        batch = stop - start
        seed_w = rng.multinomial(len(SEEDS), seed_prob, size=batch) / len(SEEDS)
        for fi, (kernel, (npos, nneg)) in enumerate(zip(contrast_kernels, item_sizes)):
            pos_w = rng.multinomial(npos, np.full(npos, 1.0 / npos), size=batch) / npos
            neg_w = rng.multinomial(nneg, np.full(nneg, 1.0 / nneg), size=batch) / nneg
            # Average the per-seed Mann--Whitney kernels, then integrate over
            # the shared class-stratified empirical item distribution.
            seed_avg = np.einsum("bs,spn->bpn", seed_w, kernel, optimize=True)
            pos_avg = np.einsum("bp,bpn->bn", pos_w, seed_avg, optimize=True)
            out[start:stop, fi] = np.einsum("bn,bn->b", pos_avg, neg_w)
    return out


def percentile_ci(values: np.ndarray, alpha: float = 0.05):
    return np.quantile(values, [alpha / 2, 1 - alpha / 2]).tolist()


def bonferroni_ci(values: np.ndarray, m: int):
    return percentile_ci(values, alpha=0.05 / m)


def max_t_intervals(point: np.ndarray, boot: np.ndarray, alpha: float = 0.05):
    """Single-step centered-bootstrap max-T simultaneous intervals.

    Studentization uses each contrast's bootstrap SD.  The joint critical value
    is the (1-alpha) quantile of the maximum absolute centered standardized
    bootstrap error.  This preserves the bootstrap dependence across contrasts.
    """
    se = np.std(boot, axis=0, ddof=1)
    if np.any(se <= 0):
        raise RuntimeError("zero bootstrap standard error in max-T family")
    standardized_error = (boot - point) / se
    max_abs = np.max(np.abs(standardized_error), axis=1)
    critical = float(np.quantile(max_abs, 1 - alpha))
    intervals = np.column_stack((point - critical * se, point + critical * se))
    observed = np.max(np.abs(point / se))
    global_p = float((1 + np.sum(max_abs >= observed)) / (len(max_abs) + 1))
    return se, critical, intervals, global_p


def summarize(point: float, boot: np.ndarray, bonf_m: int | None = None):
    result = {
        "point": float(point),
        "ci95_percentile": percentile_ci(boot),
        "bootstrap_se": float(np.std(boot, ddof=1)),
        "bootstrap_probability_positive": float(np.mean(boot > 0)),
    }
    if bonf_m is not None:
        result["ci95_bonferroni_percentile"] = bonferroni_ci(boot, bonf_m)
    return result


def main():
    excluded = exclusions()
    rng = np.random.default_rng(RNG_SEED)

    recipe_contrasts = []
    recipe_sizes = []
    recipe_points = []
    fold_counts = {}
    for fold in FOLDS:
        kernels, npos, nneg = load_kernels(
            fold,
            [("558", "hash"), ("558", "source-balanced")],
            excluded,
        )
        contrast = kernels[1] - kernels[0]
        recipe_contrasts.append(contrast)
        recipe_sizes.append((npos, nneg))
        recipe_points.append(float(np.mean(contrast)))
        fold_counts[fold] = {"spoof": npos, "bonafide": nneg, "total": npos + nneg}

    recipe_point = np.asarray(recipe_points)
    recipe_boot = bootstrap_kernel_contrasts(recipe_contrasts, recipe_sizes, REPS, rng)
    fold_se, fold_crit, fold_maxt, fold_global_p = max_t_intervals(recipe_point, recipe_boot)

    per_fold = {}
    for i, fold in enumerate(FOLDS):
        row = summarize(recipe_point[i], recipe_boot[:, i], bonf_m=len(FOLDS))
        row["ci95_max_t"] = fold_maxt[i].tolist()
        per_fold[fold] = row

    macro_point = float(np.mean(recipe_point))
    macro_boot = np.mean(recipe_boot, axis=1)
    no_ss2 = [i for i, fold in enumerate(FOLDS) if fold != "SIMPLESPEECH2"]
    no_simple = [i for i, fold in enumerate(FOLDS) if not fold.startswith("SIMPLESPEECH")]
    macros = {
        "all_8_fixed_sources": summarize(macro_point, macro_boot),
        "excluding_SIMPLESPEECH2": summarize(
            float(np.mean(recipe_point[no_ss2])), np.mean(recipe_boot[:, no_ss2], axis=1)
        ),
        "excluding_both_SIMPLESPEECH_sources": summarize(
            float(np.mean(recipe_point[no_simple])), np.mean(recipe_boot[:, no_simple], axis=1)
        ),
    }

    leave_one_out = {}
    for i, fold in enumerate(FOLDS):
        keep = [j for j in range(len(FOLDS)) if j != i]
        leave_one_out[fold] = summarize(
            float(np.mean(recipe_point[keep])), np.mean(recipe_boot[:, keep], axis=1)
        )

    pair_names = []
    pair_points = []
    pair_boots = []
    for i, j in itertools.combinations(range(len(FOLDS)), 2):
        pair_names.append((FOLDS[i], FOLDS[j]))
        pair_points.append(recipe_point[i] - recipe_point[j])
        pair_boots.append(recipe_boot[:, i] - recipe_boot[:, j])
    pair_point = np.asarray(pair_points)
    pair_boot = np.column_stack(pair_boots)
    pair_se, pair_crit, pair_maxt, pair_global_p = max_t_intervals(pair_point, pair_boot)
    pairwise = {}
    for k, (left, right) in enumerate(pair_names):
        row = summarize(pair_point[k], pair_boot[:, k], bonf_m=len(pair_names))
        row["ci95_max_t"] = pair_maxt[k].tolist()
        pairwise[f"{left}_minus_{right}"] = row

    # Budget/composition factorial.  For each fold, item draws are common to all
    # three contrasts because they are computed together from the same kernels.
    factorial_folds = ["SIMPLESPEECH2", "SIMPLESPEECH1", "NS3"]
    factorial_points = np.empty((len(factorial_folds), 3))
    factorial_boot = np.empty((REPS, len(factorial_folds), 3))
    factorial_rng = np.random.default_rng(RNG_SEED + 1)
    seed_prob = np.full(len(SEEDS), 1.0 / len(SEEDS))
    factorial_data = []
    for fi, fold in enumerate(factorial_folds):
        kernels, npos, nneg = load_kernels(
            fold,
            [
                ("558", "hash"),
                ("1000", "hash"),
                ("558", "source-balanced"),
            ],
            excluded,
        )
        contrasts = np.stack(
            [
                kernels[1] - kernels[0],  # budget: hash1000 - hash558
                kernels[2] - kernels[0],  # composition: balanced558 - hash558
                kernels[2] - kernels[1],  # composition minus budget
            ]
        )
        factorial_data.append((contrasts, npos, nneg))
        factorial_points[fi] = np.mean(contrasts, axis=(1, 2, 3))

    for start in range(0, REPS, CHUNK):
        stop = min(start + CHUNK, REPS)
        batch = stop - start
        seed_w = factorial_rng.multinomial(len(SEEDS), seed_prob, size=batch) / len(SEEDS)
        for fi, (contrasts, npos, nneg) in enumerate(factorial_data):
            pos_w = factorial_rng.multinomial(
                npos, np.full(npos, 1.0 / npos), size=batch
            ) / npos
            neg_w = factorial_rng.multinomial(
                nneg, np.full(nneg, 1.0 / nneg), size=batch
            ) / nneg
            for ci in range(3):
                seed_avg = np.einsum(
                    "bs,spn->bpn", seed_w, contrasts[ci], optimize=True
                )
                pos_avg = np.einsum("bp,bpn->bn", pos_w, seed_avg, optimize=True)
                factorial_boot[start:stop, fi, ci] = np.einsum(
                    "bn,bn->b", pos_avg, neg_w
                )

    factorial_names = ["budget_hash1000_minus_hash558", "composition_bal558_minus_hash558", "composition_minus_budget"]
    factorial = {}
    factorial_family = {}
    for ci, contrast_name in enumerate(factorial_names):
        cpoint = factorial_points[:, ci]
        cboot = factorial_boot[:, :, ci]
        _, critical, maxt, global_p = max_t_intervals(cpoint, cboot)
        factorial_family[contrast_name] = {
            "max_t_critical": critical,
            "global_max_t_p": global_p,
        }
        for fi, fold in enumerate(factorial_folds):
            factorial.setdefault(fold, {})[contrast_name] = summarize(
                cpoint[fi], cboot[:, fi], bonf_m=len(factorial_folds)
            )
            factorial[fold][contrast_name]["ci95_max_t"] = maxt[fi].tolist()

    result = {
        "method": {
            "bootstrap_replicates": REPS,
            "rng_seed": RNG_SEED,
            "seed_ids": SEEDS,
            "folds_are_fixed": True,
            "seed_draw_shared_across_folds_and_arms": True,
            "class_stratified_item_draw_shared_across_seeds_and_arms_within_fold": True,
            "cross_fold_items_overlap": False,
            "auroc": "tie-correct Mann-Whitney; 0.5 credit for equal scores",
            "ci": "percentile crossed bootstrap",
            "bonferroni_family_size": len(FOLDS),
            "max_t": "single-step centered bootstrap, fixed bootstrap-SD studentization",
        },
        "fold_counts_after_collision_exclusions": fold_counts,
        "recipe_balanced_minus_hash": per_fold,
        "recipe_family": {
            "max_t_critical": fold_crit,
            "global_max_t_p_any_effect_nonzero": fold_global_p,
        },
        "fixed_source_macros": macros,
        "leave_one_source_out_macros": leave_one_out,
        "pairwise_effect_heterogeneity": pairwise,
        "pairwise_family": {
            "max_t_critical": pair_crit,
            "global_max_t_p_any_pair_differs": pair_global_p,
        },
        "budget_composition_factorial": factorial,
        "budget_composition_families": factorial_family,
    }
    output = ROOT / "outputs" / "crossed_global_analysis.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    display_output = output.relative_to(ROOT) if output.is_relative_to(ROOT) else output
    print(f"wrote {display_output}")


if __name__ == "__main__":
    main()
