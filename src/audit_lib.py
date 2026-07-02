"""Core library for the source-shift attribution audit (self-contained).

Reads sanitized per-utterance scores from data/scores/<experiment>/seed_<n>/<fold>.jsonl,
each line {utterance_id, label, source_model, score}. stdlib + numpy only.

All numbers in the paper's primary tables/figures are leak-free: test records
byte-identical (SHA-256) to a training record are excluded via data/collision_details.csv.
"""
from __future__ import annotations
import csv, json, math, random
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SCORES = DATA / "scores"

FOLDS = ["CLAMTTS", "GPST", "MASKGCT", "NS2", "NS3",
         "SIMPLESPEECH1", "SIMPLESPEECH2", "UNIAUDIO", "VALLE"]
NON_MASK = [f for f in FOLDS if f != "MASKGCT"]


def load_exclude() -> set[str]:
    """Test utterance IDs byte-identical (SHA-256) to a training record (leak-free)."""
    excl: set[str] = set()
    p = DATA / "collision_details.csv"
    if p.exists():
        for r in csv.DictReader(p.open()):
            if r.get("reference_partition") == "train":
                excl.add(str(r["test_utterance_id"]))
    return excl


class ScoreSet:
    __slots__ = ("experiment", "seed", "fold", "labels", "scores", "sources", "utts")

    def __init__(self, experiment, seed, fold, labels, scores, sources, utts):
        self.experiment = experiment
        self.seed = seed
        self.fold = fold
        self.labels = labels
        self.scores = scores
        self.sources = sources
        self.utts = utts


def load(exclude: set[str] | None = None) -> list[ScoreSet]:
    exclude = exclude or set()
    out: list[ScoreSet] = []
    for f in sorted(SCORES.rglob("*.jsonl")):
        fold = f.stem
        seed = int(f.parent.name.replace("seed_", ""))
        experiment = f.parent.parent.name
        labels, scores, sources, utts = [], [], [], []
        for ln in f.open():
            ln = ln.strip()
            if not ln:
                continue
            o = json.loads(ln)
            u = str(o.get("utterance_id", ""))
            if u in exclude:
                continue
            labels.append(1 if o.get("label") == "spoof" else 0)
            scores.append(float(o["score"]))
            sources.append(str(o.get("source_model", "UNKNOWN")))
            utts.append(u)
        if len(scores) == 0 or len(set(labels)) < 2:
            continue
        out.append(ScoreSet(experiment, seed, fold,
                            np.asarray(labels, dtype=int), np.asarray(scores, dtype=float), sources, utts))
    return out


# ---- metrics ----
def roc_points(labels, scores):
    order = np.argsort(-scores)
    y = labels[order]
    pos = int(y.sum()); neg = len(y) - pos
    if pos == 0 or neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    tp = np.cumsum(y == 1); fp = np.cumsum(y == 0)
    return (np.concatenate([[0.0], fp / neg, [1.0]]),
            np.concatenate([[0.0], tp / pos, [1.0]]))


def auroc(labels, scores):
    fpr, tpr = roc_points(labels, scores)
    return float(np.trapezoid(tpr, fpr))


def pauc(labels, scores, max_fpr=0.05):
    fpr, tpr = roc_points(labels, scores)
    if max_fpr <= 0:
        return float("nan")
    xs, ys = [0.0], [0.0]
    for x, y in zip(fpr, tpr):
        if 0 < x < max_fpr:
            xs.append(float(x)); ys.append(float(y))
    xs.append(max_fpr); ys.append(float(np.interp(max_fpr, fpr, tpr)))
    return float(np.trapezoid(np.asarray(ys), np.asarray(xs))) / max_fpr


def eer(labels, scores):
    pos = int((labels == 1).sum()); neg = int((labels == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ss, sl = scores[order], labels[order]
    tp = np.cumsum(sl == 1); fp = np.cumsum(sl == 0)
    be = np.r_[np.flatnonzero(ss[:-1] != ss[1:]), len(scores) - 1]
    miss = np.r_[1.0, 1.0 - tp[be] / pos, 0.0]
    fa = np.r_[0.0, fp[be] / neg, 1.0]
    i = int(np.argmin(np.abs(miss - fa)))
    return float((miss[i] + fa[i]) / 2)


def median_gap(labels, scores):
    bona = scores[labels == 0]; spoof = scores[labels == 1]
    if not len(bona) or not len(spoof):
        return float("nan")
    return float(np.median(spoof) - np.median(bona))


def stratified_bootstrap_auroc(labels, scores, reps, rng):
    ip = np.where(labels == 1)[0]; ineg = np.where(labels == 0)[0]
    if len(ip) < 2 or len(ineg) < 2:
        return (float("nan"), float("nan"))
    vals = []
    for _ in range(reps):
        idx = np.concatenate([
            np.array([ip[rng.randrange(len(ip))] for _ in range(len(ip))]),
            np.array([ineg[rng.randrange(len(ineg))] for _ in range(len(ineg))])])
        vals.append(auroc(labels[idx], scores[idx]))
    a = np.asarray(vals)
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def by(items, experiment, fold):
    return {it.seed: it for it in items if it.experiment == experiment and it.fold == fold}


def fold_mean_auroc(items, experiment, fold):
    d = by(items, experiment, fold)
    if not d:
        return None
    return float(np.mean([auroc(it.labels, it.scores) for it in d.values()]))


def rank_intervals(items, experiment, reps=2000, seed=17062026):
    """Bootstrap vulnerability-rank AND seed-averaged-AUROC intervals.

    Each replicate resamples test utterances within each seed (stratified) and
    averages AUROC across seeds, then ranks folds (rank 1 = lowest AUROC). The
    AUROC CI is therefore the CI of the seed-mean AUROC (tight), matching the
    paper's "2000 bootstrap reps within each seed and averaged across seeds".
    Returns fold -> (rank_med, rank_lo, rank_hi, auroc_mean, auroc_lo, auroc_hi).
    """
    rng = random.Random(seed)
    sets = {(it.fold, it.seed): it for it in items if it.experiment == experiment and it.fold != "COSG_POOLED"}
    folds = sorted({f for f, _ in sets})
    seeds_by = {f: sorted(s for ff, s in sets if ff == f) for f in folds}
    rank_s = {f: [] for f in folds}
    auroc_s = {f: [] for f in folds}
    for _ in range(reps):
        fa = {}
        for f in folds:
            vals = []
            for s in seeds_by[f]:
                it = sets[(f, s)]
                ip = np.where(it.labels == 1)[0]; ineg = np.where(it.labels == 0)[0]
                if len(ip) < 2 or len(ineg) < 2:
                    continue
                idx = np.concatenate([
                    np.array([ip[rng.randrange(len(ip))] for _ in range(len(ip))]),
                    np.array([ineg[rng.randrange(len(ineg))] for _ in range(len(ineg))])])
                vals.append(auroc(it.labels[idx], it.scores[idx]))
            if vals:
                fa[f] = float(np.mean(vals))
        for rank, f in enumerate(sorted(fa, key=lambda x: fa[x]), start=1):
            rank_s[f].append(rank)
        for f, v in fa.items():
            auroc_s[f].append(v)
    out = {}
    for f in folds:
        if not rank_s[f]:
            continue
        rv, av = rank_s[f], auroc_s[f]
        out[f] = (float(np.median(rv)), float(np.percentile(rv, 2.5)), float(np.percentile(rv, 97.5)),
                  float(np.mean(av)), float(np.percentile(av, 2.5)), float(np.percentile(av, 97.5)))
    return out


def hierarchical_paired_delta(cells_a, cells_b, reps=10000, seed=7):
    """Hierarchical seed+utterance bootstrap CI for mean_b - mean_a AUROC.

    cells_x: {seed: (labels, scores, utts)} for the SAME test fold under two
    training conditions; pairing is at both levels (same resampled seeds, same
    resampled test utterances scored under both conditions). Returns
    (point, [lo95, hi95], bootstrap_samples) — callers derive family-adjusted
    (Bonferroni) intervals from the returned samples.
    """
    rng = random.Random(seed)
    seeds = sorted(set(cells_a) & set(cells_b))
    data = {}
    for s_ in seeds:
        la, sa, ua = cells_a[s_]
        lb, sb, ub = cells_b[s_]
        ma = {u: (l, sc) for u, l, sc in zip(ua, la, sa)}
        mb = {u: (l, sc) for u, l, sc in zip(ub, lb, sb)}
        common = sorted(set(ma) & set(mb))
        lab = np.array([ma[u][0] for u in common])
        data[s_] = (lab, np.array([ma[u][1] for u in common]),
                    np.array([mb[u][1] for u in common]),
                    np.where(lab == 1)[0], np.where(lab == 0)[0])
    def point_delta():
        va, vb = [], []
        for s_ in seeds:
            lab, xa, xb, _, _ = data[s_]
            va.append(auroc(lab, xa)); vb.append(auroc(lab, xb))
        return float(np.mean(vb) - np.mean(va))
    samples = []
    for _ in range(reps):
        drawn = [seeds[rng.randrange(len(seeds))] for _ in seeds]
        da, db = [], []
        for s_ in drawn:
            lab, xa, xb, pos, neg = data[s_]
            if len(pos) < 2 or len(neg) < 2:
                continue
            idx = np.concatenate([
                np.array([pos[rng.randrange(len(pos))] for _ in range(len(pos))]),
                np.array([neg[rng.randrange(len(neg))] for _ in range(len(neg))])])
            L = lab[idx]
            da.append(auroc(L, xa[idx])); db.append(auroc(L, xb[idx]))
        if da:
            samples.append(np.mean(db) - np.mean(da))
    arr = np.asarray(samples)
    return (round(point_delta(), 4),
            [round(float(np.percentile(arr, 2.5)), 4), round(float(np.percentile(arr, 97.5)), 4)],
            arr)
