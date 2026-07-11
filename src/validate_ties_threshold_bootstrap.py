"""Independent AUROC-tie audit and crossed operating-point bootstrap.

This is a statistical validation script for the released score files. It does
not alter any manuscript result or score. The operating-point bootstrap matches
``operating_point.py``: the threshold is estimated on a designated reference
fold, not on a separately released validation set.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from audit_lib import ROOT, auroc, load, load_exclude
import crosscorpus_lib as cc


REPS = 100_000
BATCH = 500
RNG_SEED = 20_260_711
OUT = ROOT / "outputs"


def read_jsonl(path: Path, exclude: set[str] | None = None):
    exclude = exclude or set()
    labels, scores, utts = [], [], []
    for line in path.open():
        if not line.strip():
            continue
        row = json.loads(line)
        utt = str(row.get("utterance_id", ""))
        if utt in exclude:
            continue
        label = row.get("label")
        if label not in {"spoof", "bonafide"}:
            continue
        labels.append(1 if label == "spoof" else 0)
        scores.append(float(row["score"]))
        utts.append(utt)
    return np.asarray(labels, dtype=np.int8), np.asarray(scores), utts


def auc_tie_audit():
    rows = []
    eligible = 0
    for path in sorted((ROOT / "data").rglob("*.jsonl")):
        labels, scores, _ = read_jsonl(path)
        if len(scores) == 0 or len(np.unique(labels)) != 2:
            continue
        eligible += 1
        old = auroc(labels, scores)
        correct = float(roc_auc_score(labels, scores))
        diff = correct - old
        if abs(diff) > 1e-12:
            rows.append({
                "path": str(path.relative_to(ROOT)),
                "n": len(scores),
                "n_spoof": int(labels.sum()),
                "n_bonafide": int(len(labels) - labels.sum()),
                "n_unique_scores": int(len(np.unique(scores))),
                "artifact_auroc": old,
                "sklearn_auroc": correct,
                "sklearn_minus_artifact": diff,
            })
    rows.sort(key=lambda r: abs(r["sklearn_minus_artifact"]), reverse=True)

    path = OUT / "tie_correct_auc_differences.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["path"])
        writer.writeheader()
        writer.writerows(rows)

    return {
        "eligible_score_files": eligible,
        "files_changed": len(rows),
        "maximum_absolute_change": max((abs(r["sklearn_minus_artifact"]) for r in rows), default=0.0),
        "largest_changes": rows[:10],
    }


def recipe_tie_audit():
    """Check all claim-bearing composition cells after leak pruning."""
    exclude = load_exclude()
    groups = []

    factorial = ROOT / "data" / "factorial"
    folds = ["CLAMTTS", "GPST", "NS2", "UNIAUDIO", "VALLE",
             "SIMPLESPEECH1", "SIMPLESPEECH2", "NS3"]
    for fold in folds:
        for sampler in ["hash", "source-balanced"]:
            paths = sorted((factorial / fold / "budget_558" / sampler).glob("seed_*.jsonl"))
            groups.append(("CodecFake+ recipe", fold, sampler, paths, exclude))

    for corpus, condition, family in [
        ("asvspoof5", "wavlm_frozen_backend", "ASVspoof5 recipe"),
        ("mlaad", "xlsr_peft_adapter", "MLAAD recipe"),
    ]:
        base = ROOT / "data" / "crosscorpus" / corpus
        for sampler in ["hash", "source-balanced"]:
            block = base / f"samp_{sampler}" / condition
            for fold in sorted(p.stem for p in next(iter(sorted(block.glob("seed_*")))).glob("*.jsonl")):
                paths = sorted(block.glob(f"seed_*/{fold}.jsonl"))
                groups.append((family, fold, sampler, paths, set()))

    details = []
    for family, fold, sampler, paths, exclude in groups:
        old, correct = [], []
        for path in paths:
            labels, scores, _ = read_jsonl(path, exclude)
            old.append(auroc(labels, scores))
            correct.append(float(roc_auc_score(labels, scores)))
        details.append({
            "family": family,
            "fold": fold,
            "sampler": sampler,
            "n_seeds": len(paths),
            "artifact_mean": float(np.mean(old)),
            "tie_correct_mean": float(np.mean(correct)),
            "change": float(np.mean(correct) - np.mean(old)),
        })

    # Compare the actual balanced-hash deltas as well as individual cells.
    deltas = []
    keys = sorted({(r["family"], r["fold"]) for r in details})
    for family, fold in keys:
        cells = {(r["sampler"]): r for r in details if r["family"] == family and r["fold"] == fold}
        if set(cells) != {"hash", "source-balanced"}:
            continue
        old_delta = cells["source-balanced"]["artifact_mean"] - cells["hash"]["artifact_mean"]
        new_delta = cells["source-balanced"]["tie_correct_mean"] - cells["hash"]["tie_correct_mean"]
        deltas.append({
            "family": family,
            "fold": fold,
            "artifact_delta": old_delta,
            "tie_correct_delta": new_delta,
            "change": new_delta - old_delta,
        })

    with (OUT / "tie_correct_recipe_cells.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    with (OUT / "tie_correct_recipe_deltas.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(deltas[0]))
        writer.writeheader()
        writer.writerows(deltas)

    return {
        "n_cells": len(details),
        "max_absolute_cell_change": max(abs(r["change"]) for r in details),
        "max_absolute_delta_change": max(abs(r["change"]) for r in deltas),
        "changed_cells_at_1e-12": sum(abs(r["change"]) > 1e-12 for r in details),
        "changed_deltas_at_1e-12": sum(abs(r["change"]) > 1e-12 for r in deltas),
        "cell_details": details,
        "delta_details": deltas,
    }


def codec_full_with_ids():
    out = {}
    for item in load(load_exclude()):
        if item.experiment != "full_budget_loso":
            continue
        out.setdefault(item.fold, {})[item.seed] = (item.labels, item.scores, item.utts)
    return out


def aligned_arrays(seed_cells, seeds):
    """Return seed-by-item spoof/bona-fide scores aligned by utterance ID."""
    first = seed_cells[seeds[0]]
    label_by_id = {u: int(y) for y, u in zip(first[0], first[2])}
    ids_by_label = {
        label: sorted(u for u, y in label_by_id.items() if y == label)
        for label in [0, 1]
    }
    arrays = {}
    for label, name in [(1, "spoof"), (0, "bona")]:
        ids = ids_by_label[label]
        rows = []
        for seed in seeds:
            labels, scores, utts = seed_cells[seed]
            score_by_id = {u: float(x) for u, x in zip(utts, scores)}
            current_label = {u: int(y) for u, y in zip(utts, labels)}
            if set(score_by_id) != set(label_by_id):
                raise ValueError("Utterance IDs differ across seeds")
            if any(current_label[u] != label_by_id[u] for u in label_by_id):
                raise ValueError("Labels differ across seeds")
            rows.append([score_by_id[u] for u in ids])
        arrays[name] = np.asarray(rows)
    return arrays["spoof"], arrays["bona"]


def threshold_and_own_bacc(spoof_scores, bona_scores, spoof_weights, bona_weights):
    """Vectorized exact equivalent of eer_threshold + bacc for all seeds."""
    batch = spoof_weights.shape[0]
    n_seed = spoof_scores.shape[0]
    n_spoof = spoof_scores.shape[1]
    n_bona = bona_scores.shape[1]
    combined_weights = np.concatenate([spoof_weights, bona_weights], axis=1)
    labels = np.concatenate([np.ones(n_spoof, dtype=np.int8), np.zeros(n_bona, dtype=np.int8)])
    thresholds = np.empty((batch, n_seed))
    baccs = np.empty((batch, n_seed))

    for seed in range(n_seed):
        values = np.concatenate([spoof_scores[seed], bona_scores[seed]])
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        sorted_labels = labels[order]
        weights = combined_weights[:, order]
        pos = weights * sorted_labels
        neg = weights * (1 - sorted_labels)
        pos_before = np.cumsum(pos, axis=1) - pos
        neg_before = np.cumsum(neg, axis=1) - neg
        miss = pos_before / n_spoof
        far = (n_bona - neg_before) / n_bona
        objective = np.abs(far - miss)
        group_start = np.r_[True, sorted_values[1:] != sorted_values[:-1]]
        starts = np.flatnonzero(group_start)
        # A candidate threshold exists only if that score was drawn.
        group_weight = np.add.reduceat(weights, starts, axis=1)
        candidate_objective = objective[:, starts]
        candidate_objective[group_weight == 0] = np.inf
        index = starts[np.argmin(candidate_objective, axis=1)]
        row = np.arange(batch)
        thresholds[:, seed] = sorted_values[index]
        baccs[:, seed] = 1.0 - 0.5 * (miss[row, index] + far[row, index])
    return thresholds, baccs


def bacc_at_threshold(spoof_scores, bona_scores, spoof_weights, bona_weights, thresholds):
    batch, n_seed = thresholds.shape
    n_spoof = spoof_scores.shape[1]
    n_bona = bona_scores.shape[1]
    out = np.empty((batch, n_seed))
    for seed in range(n_seed):
        pos_order = np.argsort(spoof_scores[seed], kind="stable")
        neg_order = np.argsort(bona_scores[seed], kind="stable")
        pos_values = spoof_scores[seed, pos_order]
        neg_values = bona_scores[seed, neg_order]
        pos_cum = np.concatenate([
            np.zeros((batch, 1), dtype=np.int32),
            np.cumsum(spoof_weights[:, pos_order], axis=1),
        ], axis=1)
        neg_cum = np.concatenate([
            np.zeros((batch, 1), dtype=np.int32),
            np.cumsum(bona_weights[:, neg_order], axis=1),
        ], axis=1)
        pos_index = np.searchsorted(pos_values, thresholds[:, seed], side="left")
        neg_index = np.searchsorted(neg_values, thresholds[:, seed], side="left")
        row = np.arange(batch)
        miss = pos_cum[row, pos_index] / n_spoof
        far = (n_bona - neg_cum[row, neg_index]) / n_bona
        out[:, seed] = 1.0 - 0.5 * (miss + far)
    return out


def point_gap(reference, target):
    n_seed = reference[0].shape[0]
    ref_spoof, ref_bona = reference
    tar_spoof, tar_bona = target
    ref_w_spoof = np.ones((1, ref_spoof.shape[1]), dtype=np.int32)
    ref_w_bona = np.ones((1, ref_bona.shape[1]), dtype=np.int32)
    tar_w_spoof = np.ones((1, tar_spoof.shape[1]), dtype=np.int32)
    tar_w_bona = np.ones((1, tar_bona.shape[1]), dtype=np.int32)
    ref_threshold, _ = threshold_and_own_bacc(
        ref_spoof, ref_bona, ref_w_spoof, ref_w_bona
    )
    _, oracle = threshold_and_own_bacc(
        tar_spoof, tar_bona, tar_w_spoof, tar_w_bona
    )
    transferred = bacc_at_threshold(
        tar_spoof, tar_bona, tar_w_spoof, tar_w_bona, ref_threshold
    )
    if oracle.shape[1] != n_seed:
        raise AssertionError("Seed count mismatch")
    return float(np.mean(oracle - transferred))


def multinomial_weights(rng, batch, n):
    return rng.multinomial(n, np.full(n, 1.0 / n), size=batch).astype(np.int32)


def bootstrap_corpus(corpus, folds, reference_name, reps=REPS):
    targets = sorted(f for f in folds if f != reference_name)
    seed_sets = [set(folds[f]) for f in [reference_name] + targets]
    seeds = sorted(set.intersection(*seed_sets))
    reference = aligned_arrays(folds[reference_name], seeds)
    target_arrays = {fold: aligned_arrays(folds[fold], seeds) for fold in targets}
    points = {fold: point_gap(reference, target_arrays[fold]) for fold in targets}
    samples = np.empty((reps, len(targets)))
    rng = np.random.default_rng(RNG_SEED + sum(ord(c) for c in corpus))

    done = 0
    while done < reps:
        batch = min(BATCH, reps - done)
        seed_weights = multinomial_weights(rng, batch, len(seeds))
        ref_spoof_w = multinomial_weights(rng, batch, reference[0].shape[1])
        ref_bona_w = multinomial_weights(rng, batch, reference[1].shape[1])
        ref_threshold, _ = threshold_and_own_bacc(
            reference[0], reference[1], ref_spoof_w, ref_bona_w
        )
        for column, fold in enumerate(targets):
            target = target_arrays[fold]
            spoof_w = multinomial_weights(rng, batch, target[0].shape[1])
            bona_w = multinomial_weights(rng, batch, target[1].shape[1])
            _, oracle = threshold_and_own_bacc(
                target[0], target[1], spoof_w, bona_w
            )
            transferred = bacc_at_threshold(
                target[0], target[1], spoof_w, bona_w, ref_threshold
            )
            by_seed = oracle - transferred
            samples[done:done + batch, column] = (
                np.sum(by_seed * seed_weights, axis=1) / len(seeds)
            )
        done += batch
        if done % 10_000 == 0:
            print(f"  {corpus}: {done:,}/{reps:,}", flush=True)

    rows = []
    m = len(targets)
    global_m = 32
    for column, fold in enumerate(targets):
        values = samples[:, column]
        lo, hi = np.percentile(values, [2.5, 97.5])
        blo, bhi = np.percentile(values, [100 * 0.05 / (2 * m), 100 * (1 - 0.05 / (2 * m))])
        glo, ghi = np.percentile(values, [100 * 0.05 / (2 * global_m), 100 * (1 - 0.05 / (2 * global_m))])
        rows.append({
            "corpus": corpus,
            "fold": fold,
            "n_seeds": len(seeds),
            "n_reference_spoof": reference[0].shape[1],
            "n_reference_bonafide": reference[1].shape[1],
            "n_target_spoof": target_arrays[fold][0].shape[1],
            "n_target_bonafide": target_arrays[fold][1].shape[1],
            "point_gap": points[fold],
            "ci95_low": float(lo),
            "ci95_high": float(hi),
            "bonferroni_within_corpus_low": float(blo),
            "bonferroni_within_corpus_high": float(bhi),
            "bonferroni_global32_low": float(glo),
            "bonferroni_global32_high": float(ghi),
        })

    macro = np.mean(samples, axis=1)
    macro_row = {
        "corpus": corpus,
        "fixed_fold_mean_point": float(np.mean(list(points.values()))),
        "ci95_low": float(np.percentile(macro, 2.5)),
        "ci95_high": float(np.percentile(macro, 97.5)),
        "n_target_folds": len(targets),
    }
    return rows, macro_row


def threshold_audit(reps=REPS):
    corpora = {
        "CodecFake+": (codec_full_with_ids(), "MASKGCT"),
        "asvspoof2019la": (cc.load("asvspoof2019la", "full", with_utts=True), "A07"),
        "asvspoof5": (cc.load("asvspoof5", "full", with_utts=True), "A17"),
        "mlaad": (cc.load("mlaad", "full", with_utts=True), "JENNY"),
    }
    rows, macros = [], []
    for corpus, (folds, reference) in corpora.items():
        print(f"Bootstrapping {corpus} operating-point gaps", flush=True)
        corpus_rows, macro = bootstrap_corpus(corpus, folds, reference, reps)
        rows.extend(corpus_rows)
        macros.append(macro)

    with (OUT / "operating_point_crossed_bootstrap.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (OUT / "operating_point_crossed_bootstrap_macro.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(macros[0]))
        writer.writeheader()
        writer.writerows(macros)
    return rows, macros


def main():
    OUT.mkdir(exist_ok=True)
    print("Auditing AUROC ties", flush=True)
    tie = auc_tie_audit()
    recipe = recipe_tie_audit()
    rows, macros = threshold_audit()
    summary = {
        "repetitions": REPS,
        "rng_seed": RNG_SEED,
        "auc_tie_audit": tie,
        "recipe_tie_audit": {k: v for k, v in recipe.items() if not k.endswith("details")},
        "threshold_fold_rows": rows,
        "threshold_macro_rows": macros,
        "notes": [
            "Threshold-source and target utterances were resampled independently and stratified by class.",
            "One item resample per fold was shared across all seeds; seed resampling was paired across threshold source and target.",
            "There is no separately released validation-score set; the threshold source is the designated reference fold, matching operating_point.py.",
            "Fold CIs are percentile crossed-bootstrap intervals; familywise intervals are two-sided Bonferroni intervals.",
        ],
    }
    with (OUT / "ties_threshold_bootstrap_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
