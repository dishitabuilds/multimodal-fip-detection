"""Group-aware train / validation / test splitting.

The rule this module enforces
-----------------------------
Split by **claim cluster**, never by record. A cluster is the output of
`curation.dedup.cluster_records` — every record describing the same underlying
viral item. If one cluster's records are spread across train and test, the model
has seen the test answer during training and every reported number is inflated.

This is the most common silent failure in misinformation-detection papers, and
it is invisible in the metrics: leakage makes results better, not worse, so
nothing looks wrong. The only defence is to assert it cannot happen, which
`verify_no_leakage` does and the test suite checks.

Stratification
--------------
Clusters are bucketed by (label, dominant source) and each bucket is dealt out
across the splits in proportion. This keeps two things true:

  * label balance is comparable across splits, so validation F1 means the same
    thing as test F1
  * no split is accidentally all-BOOM, which would make the test set a measure
    of one publisher's house style rather than of fraud

Determinism
-----------
Assignment is driven by a seeded RNG over sorted cluster ids, so the same input
plus the same seed always produces the same split. The manifest written by
`save_manifest` records the seed, the counts and a hash of the assignment, so a
result can be traced back to the exact split that produced it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

log = logging.getLogger(__name__)

SPLIT_NAMES = ("train", "val", "test")


@dataclass
class SplitReport:
    seed: int = 0
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)
    n_records: int = 0
    n_clusters: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    cluster_counts: dict[str, int] = field(default_factory=dict)
    label_balance: dict[str, dict[str, int]] = field(default_factory=dict)
    source_balance: dict[str, dict[str, int]] = field(default_factory=dict)
    leakage_ok: bool = False
    assignment_hash: str = ""

    def summary(self) -> str:
        lines = [
            f"seed {self.seed}   ratios {self.ratios}",
            f"{self.n_records} records in {self.n_clusters} clusters",
            "",
            f"{'split':<8}{'records':>9}{'clusters':>10}{'  label balance'}",
            "-" * 58,
        ]
        for s in SPLIT_NAMES:
            lb = self.label_balance.get(s, {})
            lb_txt = "  ".join(f"{k}={v}" for k, v in sorted(lb.items()))
            lines.append(
                f"{s:<8}{self.counts.get(s, 0):>9}{self.cluster_counts.get(s, 0):>10}  {lb_txt}"
            )
        lines.append("-" * 58)
        lines.append(f"leakage check: {'PASS' if self.leakage_ok else 'FAIL'}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------


def _stratum(records_in_cluster: Sequence) -> tuple[str, str]:
    """Bucket key for a cluster: its majority label and majority source."""
    label = Counter(r.label for r in records_in_cluster).most_common(1)[0][0]
    source = Counter(
        (r.source or "").split(":")[-1] for r in records_in_cluster
    ).most_common(1)[0][0]
    return label, source


def split_by_cluster(
    records: Sequence,
    assignment: dict[str, str],
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    stratify: bool = True,
) -> tuple[dict[str, str], SplitReport]:
    """Return (uid -> split name, report).

    `assignment` maps uid -> cluster id, as produced by dedup.cluster_records.
    Records whose uid is absent from `assignment` are treated as singleton
    clusters, so this is safe to call before deduplication has been run.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {ratios} summing to {sum(ratios)}")
    if any(r < 0 for r in ratios):
        raise ValueError(f"ratios must be non-negative, got {ratios}")

    by_cluster: dict[str, list] = defaultdict(list)
    for r in records:
        by_cluster[assignment.get(r.uid, r.uid)].append(r)

    # Bucket clusters by stratum. Sorted keys + seeded RNG = reproducible.
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for cid, members in by_cluster.items():
        key = _stratum(members) if stratify else ("all", "all")
        strata[key].append(cid)

    rng = random.Random(seed)
    split_of_cluster: dict[str, str] = {}

    for key in sorted(strata):
        cids = sorted(strata[key])
        rng.shuffle(cids)
        n = len(cids)

        # Largest-remainder allocation, so small strata do not silently lose
        # their val/test share to floor().
        exact = [n * r for r in ratios]
        base = [int(x) for x in exact]
        for _ in range(n - sum(base)):
            frac = [e - b for e, b in zip(exact, base)]
            base[frac.index(max(frac))] += 1

        idx = 0
        for name, take in zip(SPLIT_NAMES, base):
            for cid in cids[idx: idx + take]:
                split_of_cluster[cid] = name
            idx += take

    uid_split = {
        r.uid: split_of_cluster[assignment.get(r.uid, r.uid)] for r in records
    }

    # --- report --------------------------------------------------------
    rep = SplitReport(seed=seed, ratios=ratios, n_records=len(records),
                      n_clusters=len(by_cluster))
    rep.counts = dict(Counter(uid_split.values()))
    rep.cluster_counts = dict(Counter(split_of_cluster.values()))

    lb: dict[str, Counter] = defaultdict(Counter)
    sb: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        s = uid_split[r.uid]
        lb[s][r.label] += 1
        sb[s][(r.source or "").split(":")[-1]] += 1
    rep.label_balance = {k: dict(v) for k, v in lb.items()}
    rep.source_balance = {k: dict(v) for k, v in sb.items()}

    rep.leakage_ok = verify_no_leakage(uid_split, assignment, records)[0]
    rep.assignment_hash = hashlib.sha256(
        json.dumps(sorted(uid_split.items()), sort_keys=True).encode()
    ).hexdigest()[:16]

    return uid_split, rep


# ---------------------------------------------------------------------------


def verify_no_leakage(
    uid_split: dict[str, str],
    assignment: dict[str, str],
    records: Sequence,
) -> tuple[bool, list[str]]:
    """Assert every cluster lives entirely in one split.

    Returns (ok, problems). Call this before trusting any evaluation number.
    """
    cluster_splits: dict[str, set[str]] = defaultdict(set)
    for r in records:
        cid = assignment.get(r.uid, r.uid)
        s = uid_split.get(r.uid)
        if s is not None:
            cluster_splits[cid].add(s)

    problems = [
        f"cluster {cid} spans splits {sorted(splits)}"
        for cid, splits in cluster_splits.items()
        if len(splits) > 1
    ]
    return (not problems), problems


def save_manifest(
    uid_split: dict[str, str],
    report: SplitReport,
    out_dir: str | Path,
    assignment: dict[str, str] | None = None,
) -> Path:
    """Freeze the split to disk. Commit this file — results are traceable to it."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "split_manifest.json"

    payload = {
        "seed": report.seed,
        "ratios": list(report.ratios),
        "n_records": report.n_records,
        "n_clusters": report.n_clusters,
        "counts": report.counts,
        "cluster_counts": report.cluster_counts,
        "label_balance": report.label_balance,
        "source_balance": report.source_balance,
        "leakage_ok": report.leakage_ok,
        "assignment_hash": report.assignment_hash,
        "splits": uid_split,
        "clusters": assignment or {},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    log.info("wrote split manifest -> %s", path)
    return path


def load_manifest(path: str | Path) -> tuple[dict[str, str], dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["splits"], data
