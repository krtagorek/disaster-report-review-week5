"""Reproducible pre-split inspection; never invokes a model or changes labels."""
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path
from audit_source import normalize

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "data/source-manifest.json").read_text())
    source = ROOT / manifest["local_file"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["sha256"], "Source checksum mismatch"
    with gzip.open(source, "rt", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels = manifest["initial_classes"]
    candidates = [r for r in rows if r["EVENT_TYPE"] in labels and 80 <= len(r["EVENT_NARRATIVE"].strip()) <= 2000]
    groups = collections.defaultdict(list)
    for row in candidates:
        groups[normalize(row["EVENT_NARRATIVE"])].append(row)
    conflicts = [g for g in groups.values() if len({r["EVENT_TYPE"] for r in g}) > 1]
    excluded = {r["EVENT_ID"] for g in conflicts for r in g}
    def record(r):
        return {k: r[k] for k in ("EVENT_ID", "EPISODE_ID", "EVENT_TYPE", "EVENT_NARRATIVE")}
    report = {
        "source_sha256": manifest["sha256"],
        "stage": "pre-split inspection; no predictions or semantic approval of all rows",
        "candidate_count": len(candidates),
        "conflicting_groups": [[record(r) for r in sorted(g, key=lambda r: r["EVENT_ID"])] for g in sorted(conflicts, key=lambda g: min(r["EVENT_ID"] for r in g))],
        "predeclared_exclusions": [{"event_id": i, "reason": "identical normalized narrative has multiple selected reference labels"} for i in sorted(excluded)],
        "classes": {label: {
            "candidate_count": sum(r["EVENT_TYPE"] == label for r in candidates),
            "remaining_after_conflict_exclusion": sum(r["EVENT_TYPE"] == label and r["EVENT_ID"] not in excluded for r in candidates),
            "inspection_examples": [record(r) for r in sorted((r for r in candidates if r["EVENT_TYPE"] == label), key=lambda r: hashlib.sha256(r["EVENT_ID"].encode()).hexdigest())[:5]],
        } for label in labels},
        "limitations": ["Five deterministic examples per class are a spot check, not an exhaustive semantic audit.", "No train/development/final assignment exists yet.", "Near duplicates and episode connectivity remain to be computed before splitting."],
    }
    destination = ROOT / "data/audits/candidate-inspection.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(candidates)} candidates; {len(conflicts)} conflicting groups; {len(excluded)} excluded rows")


if __name__ == "__main__":
    main()
