#!/usr/bin/env python3
"""Crossed seed-by-item uncertainty audit for CodecFake+ source rankings.

This is a score-only reanalysis of the fixed-budget hash and source-balanced
factorial cells.  Within a fold, each bootstrap item draw is shared across all
training seeds and both sampling arms.  A single bootstrap seed draw is shared
across folds and arms.  AUROC is the tie-correct Mann--Whitney statistic.
"""
from __future__ import annotations

import argparse
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
CONDITIONS = ["hash", "source-balanced"]
PAIR_I, PAIR_J = np.asarray(
    list(itertools.combinations(range(len(FOLDS)), 2)), dtype=int
).T


def load_exclusions() -> set[str]:
    exclusions: set[str] = set()
    with (ROOT / "data" / "collision_details.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("reference_partition") == "train":
                exclusions.add(str(row["test_utterance_id"]))
    return exclusions


def read_score_map(path: Path, exclusions: set[str]) -> dict[str, tuple[int, float]]:
    result: dict[str, tuple[int, float]] = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            utterance_id = str(row.get("utterance_id", ""))
            if utterance_id in exclusions:
                continue
            label = 1 if row.get("label") == "spoof" else 0
            result[utterance_id] = (label, float(row["score"]))
    return result


def tie_correct_comparison(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    return (pos[:, None] > neg[None, :]).astype(float) + 0.5 * (
        pos[:, None] == neg[None, :]
    )


def tau_b_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Kendall tau-b for corresponding rows of two matrices."""
    if a.ndim == 1:
        a = a[None, :]
    if b.ndim == 1:
        b = b[None, :]
    da = np.sign(a[:, PAIR_I] - a[:, PAIR_J])
    db = np.sign(b[:, PAIR_I] - b[:, PAIR_J])
    numerator = np.sum(da * db, axis=1)
    denominator = np.sqrt(
        np.sum(da != 0, axis=1) * np.sum(db != 0, axis=1)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan, dtype=float),
        where=denominator > 0,
    )


def average_ranks_rows(values: np.ndarray) -> np.ndarray:
    """Ascending average ranks, with rank 1 denoting lowest AUROC."""
    lower = np.sum(values[:, None, :] < values[:, :, None], axis=2)
    equal = np.sum(values[:, None, :] == values[:, :, None], axis=2) - 1
    return 1.0 + lower + 0.5 * equal


def percentile_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.nanmean(values)),
        "median": float(np.nanmedian(values)),
        "lo95": float(np.nanpercentile(values, 2.5)),
        "hi95": float(np.nanpercentile(values, 97.5)),
    }


def simultaneous_max_t(
    boot: np.ndarray, point: np.ndarray, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """Single-step bootstrap max-t intervals and adjusted p-values.

    The bootstrap standard deviation is used as the fixed studentizer.  The
    reference distribution is the maximum absolute centered bootstrap error.
    """
    se = np.std(boot, axis=0, ddof=1)
    centered_t = np.divide(
        boot - point,
        se,
        out=np.zeros_like(boot, dtype=float),
        where=se > 0,
    )
    max_t = np.max(np.abs(centered_t), axis=1)
    critical = float(np.quantile(max_t, 1.0 - alpha))
    lo = point - critical * se
    hi = point + critical * se
    observed_t = np.divide(
        np.abs(point), se, out=np.full_like(point, np.inf), where=se > 0
    )
    adjusted_p = np.asarray(
        [(1.0 + np.sum(max_t >= value)) / (len(max_t) + 1.0) for value in observed_t]
    )
    return lo, hi, critical, adjusted_p, max_t


def pair_records(
    point_values: np.ndarray,
    boot_values: np.ndarray,
) -> tuple[list[dict[str, float | str | bool]], dict[str, float]]:
    point = point_values[PAIR_I] - point_values[PAIR_J]
    boot = boot_values[:, PAIR_I] - boot_values[:, PAIR_J]
    lo, hi, critical, adjusted_p, _ = simultaneous_max_t(boot, point)
    bonf_q = 0.05 / (2.0 * len(PAIR_I))
    bonf_lo = np.quantile(boot, bonf_q, axis=0)
    bonf_hi = np.quantile(boot, 1.0 - bonf_q, axis=0)
    records = []
    for k, (i, j) in enumerate(zip(PAIR_I, PAIR_J)):
        probability = float(np.mean(boot[:, k] > 0) + 0.5 * np.mean(boot[:, k] == 0))
        records.append(
            {
                "left": FOLDS[i],
                "right": FOLDS[j],
                "point_left_minus_right": float(point[k]),
                "p_left_gt_right": probability,
                "max_t_lo95": float(lo[k]),
                "max_t_hi95": float(hi[k]),
                "max_t_adjusted_p": float(adjusted_p[k]),
                "max_t_resolved": bool(lo[k] > 0 or hi[k] < 0),
                "bonferroni_percentile_lo95": float(bonf_lo[k]),
                "bonferroni_percentile_hi95": float(bonf_hi[k]),
                "bonferroni_percentile_resolved": bool(
                    bonf_lo[k] > 0 or bonf_hi[k] < 0
                ),
            }
        )
    metadata = {
        "max_t_critical": critical,
        "max_t_resolved_count": int(sum(r["max_t_resolved"] for r in records)),
        "bonferroni_percentile_resolved_count": int(
            sum(r["bonferroni_percentile_resolved"] for r in records)
        ),
    }
    return records, metadata


def split_perturbation(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average within/cross-composition tau over all 2/3 positional splits.

    values has shape (replicate, condition, fold, seed_position).
    """
    all_positions = set(range(values.shape[3]))
    within_terms, cross_terms = [], []
    for left_tuple in itertools.combinations(range(values.shape[3]), 2):
        left = list(left_tuple)
        right = sorted(all_positions - set(left))
        hash_left = np.mean(np.take(values[:, 0, :, :], left, axis=2), axis=2)
        hash_right = np.mean(np.take(values[:, 0, :, :], right, axis=2), axis=2)
        balanced_left = np.mean(
            np.take(values[:, 1, :, :], left, axis=2), axis=2
        )
        balanced_right = np.mean(
            np.take(values[:, 1, :, :], right, axis=2), axis=2
        )
        within_terms.append(
            0.5
            * (
                tau_b_rows(hash_left, hash_right)
                + tau_b_rows(balanced_left, balanced_right)
            )
        )
        cross_terms.append(
            0.5
            * (
                tau_b_rows(hash_left, balanced_right)
                + tau_b_rows(hash_right, balanced_left)
            )
        )
    within = np.nanmean(np.stack(within_terms, axis=1), axis=1)
    cross = np.nanmean(np.stack(cross_terms, axis=1), axis=1)
    return within, cross, cross - within


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=100_000)
    parser.add_argument("--batch", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs" / "rank_crossed_audit.json"
    )
    args = parser.parse_args()
    exclusions = load_exclusions()

    seed_sets = []
    for fold in FOLDS:
        for condition in CONDITIONS:
            seed_sets.append(
                {
                    int(path.stem.removeprefix("seed_"))
                    for path in (TREE / fold / "budget_558" / condition).glob("seed_*.jsonl")
                }
            )
    seeds = sorted(set.intersection(*seed_sets))
    if len(seeds) < 2:
        raise RuntimeError(f"Too few globally paired seeds: {seeds}")

    comparison: list[list[list[np.ndarray]]] = [
        [[None for _ in seeds] for _ in FOLDS] for _ in CONDITIONS
    ]
    alignment = {}
    point_seed = np.empty((len(CONDITIONS), len(FOLDS), len(seeds)), dtype=float)

    for fold_index, fold in enumerate(FOLDS):
        maps = {}
        for condition_index, condition in enumerate(CONDITIONS):
            for seed_index, seed in enumerate(seeds):
                path = TREE / fold / "budget_558" / condition / f"seed_{seed}.jsonl"
                maps[(condition_index, seed_index)] = read_score_map(path, exclusions)
        common = set.intersection(*(set(value) for value in maps.values()))
        union = set.union(*(set(value) for value in maps.values()))
        labels_by_id = {}
        for utterance_id in sorted(common):
            labels = {
                score_map[utterance_id][0] for score_map in maps.values()
            }
            if len(labels) != 1:
                raise RuntimeError(f"Inconsistent labels for {fold}/{utterance_id}: {labels}")
            labels_by_id[utterance_id] = labels.pop()
        positive_ids = sorted(u for u, label in labels_by_id.items() if label == 1)
        negative_ids = sorted(u for u, label in labels_by_id.items() if label == 0)
        if len(positive_ids) < 2 or len(negative_ids) < 2:
            raise RuntimeError(f"Too few class items for {fold}")
        alignment[fold] = {
            "union_items": len(union),
            "common_items": len(common),
            "dropped_noncommon_items": len(union - common),
            "spoof_items": len(positive_ids),
            "bonafide_items": len(negative_ids),
        }
        for condition_index in range(len(CONDITIONS)):
            for seed_index in range(len(seeds)):
                score_map = maps[(condition_index, seed_index)]
                positive_scores = np.asarray([score_map[u][1] for u in positive_ids])
                negative_scores = np.asarray([score_map[u][1] for u in negative_ids])
                matrix = tie_correct_comparison(positive_scores, negative_scores)
                comparison[condition_index][fold_index][seed_index] = matrix
                point_seed[condition_index, fold_index, seed_index] = np.mean(matrix)

    point = np.mean(point_seed, axis=2)
    item_seed_boot = np.empty(
        (args.reps, len(CONDITIONS), len(FOLDS), len(seeds)), dtype=np.float32
    )
    seed_draws = np.empty((args.reps, len(seeds)), dtype=np.int8)
    boot = np.empty((args.reps, len(CONDITIONS), len(FOLDS)), dtype=np.float32)
    rng = np.random.default_rng(args.seed)

    for start in range(0, args.reps, args.batch):
        stop = min(start + args.batch, args.reps)
        size = stop - start
        drawn_seed = rng.integers(0, len(seeds), size=(size, len(seeds)), dtype=np.int8)
        seed_draws[start:stop] = drawn_seed
        batch_seed_auc = np.empty(
            (size, len(CONDITIONS), len(FOLDS), len(seeds)), dtype=float
        )
        for fold_index, fold in enumerate(FOLDS):
            n_positive = alignment[fold]["spoof_items"]
            n_negative = alignment[fold]["bonafide_items"]
            positive_counts = rng.multinomial(
                n_positive, np.full(n_positive, 1.0 / n_positive), size=size
            )
            negative_counts = rng.multinomial(
                n_negative, np.full(n_negative, 1.0 / n_negative), size=size
            )
            denominator = float(n_positive * n_negative)
            for condition_index in range(len(CONDITIONS)):
                for seed_index in range(len(seeds)):
                    matrix = comparison[condition_index][fold_index][seed_index]
                    batch_seed_auc[:, condition_index, fold_index, seed_index] = (
                        np.einsum(
                            "bi,ij,bj->b",
                            positive_counts,
                            matrix,
                            negative_counts,
                            optimize=True,
                        )
                        / denominator
                    )
        item_seed_boot[start:stop] = batch_seed_auc
        gathered = np.take_along_axis(
            batch_seed_auc,
            np.broadcast_to(
                drawn_seed[:, None, None, :], batch_seed_auc.shape
            ),
            axis=3,
        )
        boot[start:stop] = np.mean(gathered, axis=3)

    result: dict[str, object] = {
        "method": {
            "replicates": args.reps,
            "rng_seed": args.seed,
            "seeds": seeds,
            "conditions": CONDITIONS,
            "fold_order": FOLDS,
            "auroc": "tie-correct Mann-Whitney; 0.5 credit for cross-class ties",
            "item_bootstrap": "class-stratified within each fold; one draw shared across all seeds and both arms",
            "seed_bootstrap": "one five-seed draw shared across all folds and both arms",
            "rank": "ascending average rank; rank 1 is lowest AUROC (most vulnerable)",
            "simultaneous_intervals": "single-step bootstrap max-t over all 28 source pairs, fixed bootstrap-SD studentizer",
        },
        "alignment": alignment,
        "point_seed_mean_auroc": {
            condition: {
                fold: float(point[condition_index, fold_index])
                for fold_index, fold in enumerate(FOLDS)
            }
            for condition_index, condition in enumerate(CONDITIONS)
        },
    }

    rank_section = {}
    for condition_index, condition in enumerate(CONDITIONS):
        ranks = average_ranks_rows(boot[:, condition_index, :])
        point_rank = average_ranks_rows(point[condition_index][None, :])[0]
        pairwise, pair_metadata = pair_records(
            point[condition_index], boot[:, condition_index, :]
        )
        rank_section[condition] = {
            "ranks": {
                fold: {
                    "point": float(point_rank[fold_index]),
                    "median": float(np.median(ranks[:, fold_index])),
                    "lo95": float(np.percentile(ranks[:, fold_index], 2.5)),
                    "hi95": float(np.percentile(ranks[:, fold_index], 97.5)),
                }
                for fold_index, fold in enumerate(FOLDS)
            },
            "pairwise": pairwise,
            **pair_metadata,
        }
    result["rank_uncertainty"] = rank_section

    tau_boot = tau_b_rows(boot[:, 0, :], boot[:, 1, :])
    tau_point = float(tau_b_rows(point[0], point[1])[0])
    result["hash_vs_balanced_tau_b"] = {
        "point": tau_point,
        **percentile_summary(tau_boot),
    }

    effect_point = point[1] - point[0]
    effect_boot = boot[:, 1, :] - boot[:, 0, :]
    interaction_point = effect_point[PAIR_I] - effect_point[PAIR_J]
    interaction_boot = effect_boot[:, PAIR_I] - effect_boot[:, PAIR_J]
    int_lo, int_hi, int_critical, int_p, int_max_t = simultaneous_max_t(
        interaction_boot, interaction_point
    )
    interaction_records = []
    for k, (i, j) in enumerate(zip(PAIR_I, PAIR_J)):
        interaction_records.append(
            {
                "left": FOLDS[i],
                "right": FOLDS[j],
                "point_delta_left_minus_delta_right": float(interaction_point[k]),
                "p_delta_left_gt_delta_right": float(
                    np.mean(interaction_boot[:, k] > 0)
                    + 0.5 * np.mean(interaction_boot[:, k] == 0)
                ),
                "max_t_lo95": float(int_lo[k]),
                "max_t_hi95": float(int_hi[k]),
                "max_t_adjusted_p": float(int_p[k]),
                "max_t_resolved": bool(int_lo[k] > 0 or int_hi[k] < 0),
            }
        )
    observed_global = float(
        np.max(
            np.divide(
                np.abs(interaction_point),
                np.std(interaction_boot, axis=0, ddof=1),
                out=np.full_like(interaction_point, np.inf),
                where=np.std(interaction_boot, axis=0, ddof=1) > 0,
            )
        )
    )
    result["composition_effect_heterogeneity"] = {
        "definition": "(balanced-hash)_left - (balanced-hash)_right",
        "max_t_critical": int_critical,
        "global_observed_max_abs_t": observed_global,
        "global_max_t_p": float(
            (1.0 + np.sum(int_max_t >= observed_global)) / (len(int_max_t) + 1.0)
        ),
        "resolved_pair_count": int(sum(r["max_t_resolved"] for r in interaction_records)),
        "pairwise": interaction_records,
    }

    observed_values = point_seed[None, :, :, :]
    observed_within, observed_cross, observed_difference = split_perturbation(
        observed_values
    )
    item_within, item_cross, item_difference = split_perturbation(item_seed_boot)
    gathered_all = np.take_along_axis(
        item_seed_boot,
        np.broadcast_to(seed_draws[:, None, None, :], item_seed_boot.shape),
        axis=3,
    )
    crossed_within, crossed_cross, crossed_difference = split_perturbation(gathered_all)
    result["composition_vs_seed_perturbation"] = {
        "definition": {
            "within": "mean over all 10 2/3 splits of [tau_b(hash_A,hash_B)+tau_b(balanced_A,balanced_B)]/2",
            "cross": "mean over all 10 2/3 splits of [tau_b(hash_A,balanced_B)+tau_b(hash_B,balanced_A)]/2",
            "difference": "cross - within; negative means cross-composition agreement is lower",
        },
        "observed": {
            "within": float(observed_within[0]),
            "cross": float(observed_cross[0]),
            "difference": float(observed_difference[0]),
        },
        "item_only_bootstrap_conditional_on_five_seeds": {
            "within": percentile_summary(item_within),
            "cross": percentile_summary(item_cross),
            "difference": percentile_summary(item_difference),
            "p_difference_below_zero": float(np.mean(item_difference < 0)),
        },
        "crossed_seed_and_item_bootstrap": {
            "within": percentile_summary(crossed_within),
            "cross": percentile_summary(crossed_cross),
            "difference": percentile_summary(crossed_difference),
            "p_difference_below_zero": float(np.mean(crossed_difference < 0)),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {args.output}")
    print(f"hash vs balanced tau-b: {tau_point:+.6f} "
          f"[{np.nanpercentile(tau_boot, 2.5):+.6f}, {np.nanpercentile(tau_boot, 97.5):+.6f}]")
    for condition in CONDITIONS:
        section = rank_section[condition]
        print(f"{condition}: max-t resolved {section['max_t_resolved_count']}/28; "
              f"Bonferroni-percentile resolved {section['bonferroni_percentile_resolved_count']}/28")
    print("composition-effect heterogeneity: "
          f"global max-t p={result['composition_effect_heterogeneity']['global_max_t_p']:.6g}, "
          f"resolved={result['composition_effect_heterogeneity']['resolved_pair_count']}/28")
    print("split perturbation observed/crossed-bootstrap difference: "
          f"{observed_difference[0]:+.6f} "
          f"[{np.nanpercentile(crossed_difference, 2.5):+.6f}, "
          f"{np.nanpercentile(crossed_difference, 97.5):+.6f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
