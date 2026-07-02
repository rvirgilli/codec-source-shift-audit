"""Reference implementation of the composition-aware training samplers.

This is the exact selection algorithm behind the paper's recipe (Sec. V):
when capping a training set to a fixed budget, choose the retained SPOOF rows
with an explicit retained-source allocation instead of source-agnostic hashing.

Samplers (spoof rows only; bona-fide always uses the `hash` rule):
  hash                : source-agnostic — rank all rows by a stable SHA-256 of
                        (seed, heldout, utterance_id) and keep the first N.
                        This is the naive default the paper compares against.
  source-balanced     : equal per-source quota via water-filling — repeatedly
                        split the remaining budget equally over sources that
                        still have capacity (small sources are kept whole,
                        the dominant source is capped). The paper's recipe.
  source-proportional : largest-remainder allocation proportional to each
                        source's available count.

Selection within a source is by the same stable SHA-256 ranking, so the whole
procedure is deterministic given (seed, heldout fold). Mirrors the training
pipeline used for every sampler cell in data/factorial/ and the MLAAD samp_*
blocks; no external dependencies.

Example
-------
    from recipe_sampler import subsample_spoof_rows
    kept = subsample_spoof_rows(rows, target=307, seed=7, heldout="SIMPLESPEECH2",
                                sampler="source-balanced")
    # rows: [{"utterance_id": ..., "source_model": ...}, ...]
"""
from __future__ import annotations
import hashlib
from collections import defaultdict


def stable_rank(seed: int, heldout: str, key: str) -> str:
    return hashlib.sha256(f"{seed}:{heldout}:{key}".encode()).hexdigest()


def balanced_allocations(counts: dict[str, int], target: int) -> dict[str, int]:
    """Water-filling: equal shares over sources with remaining capacity."""
    allocations = dict.fromkeys(counts, 0)
    remaining = target
    active = sorted(counts)
    while remaining > 0 and active:
        base, extra = divmod(remaining, len(active))
        next_active: list[str] = []
        progressed = False
        for index, source in enumerate(active):
            requested = base + (1 if index < extra else 0)
            capacity = counts[source] - allocations[source]
            take = min(requested, capacity)
            allocations[source] += take
            remaining -= take
            progressed = progressed or take > 0
            if allocations[source] < counts[source]:
                next_active.append(source)
        if not progressed:
            break
        active = next_active
    if remaining != 0:
        raise ValueError(f"balanced allocation left {remaining} rows unallocated")
    return allocations


def proportional_allocations(counts: dict[str, int], target: int) -> dict[str, int]:
    """Largest-remainder allocation proportional to per-source availability."""
    total = sum(counts.values())
    allocations: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for source in sorted(counts):
        exact = target * counts[source] / total
        allocations[source] = min(counts[source], int(exact))
        remainders.append((exact - allocations[source], source))
    remaining = target - sum(allocations.values())
    while remaining > 0:
        progressed = False
        for _, source in sorted(remainders, reverse=True):
            if allocations[source] >= counts[source]:
                continue
            allocations[source] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            raise ValueError("proportional allocation could not place all rows")
    return allocations


def subsample_spoof_rows(rows: list[dict], *, target: int, seed: int,
                         heldout: str, sampler: str,
                         partition: str = "train") -> list[dict]:
    """Return the deterministic spoof subset for the given sampler."""
    if sampler == "hash":
        ranked = sorted(rows, key=lambda r: stable_rank(
            seed, heldout, f"subsample:{partition}:spoof:{r['utterance_id']}"))
        return ranked[:target]
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[str(r["source_model"])].append(r)
    counts = {s: len(g) for s, g in groups.items()}
    if sampler == "source-balanced":
        allocations = balanced_allocations(counts, target)
    elif sampler == "source-proportional":
        allocations = proportional_allocations(counts, target)
    else:
        raise ValueError(f"unknown sampler: {sampler}")
    selected: list[dict] = []
    for source in sorted(groups):
        quota = allocations.get(source, 0)
        if quota <= 0:
            continue
        ranked = sorted(groups[source], key=lambda r: stable_rank(
            seed, heldout,
            f"subsample:{partition}:{sampler}:spoof:{source}:{r['utterance_id']}"))
        selected.extend(ranked[:quota])
    if len(selected) != target:
        raise ValueError(f"{sampler} selected {len(selected)} rows, expected {target}")
    return selected


if __name__ == "__main__":
    # tiny self-check: 558-budget style split over an unbalanced toy corpus
    toy = [{"utterance_id": f"u{i}", "source_model": s}
           for s, n in {"BIG": 300, "MID": 60, "SMALL": 20}.items() for i in range(n)]
    for sampler in ("hash", "source-balanced", "source-proportional"):
        kept = subsample_spoof_rows(toy, target=120, seed=7, heldout="DEMO", sampler=sampler)
        by = defaultdict(int)
        for r in kept:
            by[r["source_model"]] += 1
        print(f"{sampler:22s} -> {dict(sorted(by.items()))}")
