"""Standard-library audit of the pinned source; no model calls or training."""
import collections
import csv
import gzip
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def normalize(value):
    return re.sub(r"\s+", " ", value.casefold()).strip()


def main():
    manifest = json.loads((ROOT / "data/source-manifest.json").read_text())
    source = ROOT / manifest["local_file"]
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != manifest["sha256"]:
        raise SystemExit("Source checksum changed; do not silently use a different snapshot.")
    with gzip.open(source, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"EVENT_ID", "EPISODE_ID", "EVENT_TYPE", "EVENT_NARRATIVE"}
        if not required.issubset(reader.fieldnames or []):
            raise SystemExit("Source schema is missing required fields.")
        rows = list(reader)
    labels = manifest["initial_classes"]
    by_label = {}
    narrative_labels = collections.defaultdict(set)
    for label in labels:
        selected = [row for row in rows if row["EVENT_TYPE"] == label]
        usable = [row for row in selected if 80 <= len(row["EVENT_NARRATIVE"].strip()) <= 2000]
        for row in usable:
            narrative_labels[normalize(row["EVENT_NARRATIVE"])].add(label)
        by_label[label] = {
            "all_rows": len(selected),
            "candidate_rows_by_length_only": len(usable),
            "unique_normalized_candidate_narratives": len({normalize(row["EVENT_NARRATIVE"]) for row in usable}),
            "candidate_episodes": len({row["EPISODE_ID"] for row in usable}),
        }
    report = {
        "source_sha256": digest,
        "source_rows": len(rows),
        "duplicate_event_ids": len(rows) - len({row["EVENT_ID"] for row in rows}),
        "source_episodes": len({row["EPISODE_ID"] for row in rows}),
        "classes": by_label,
        "candidate_rows_by_length_only": sum(value["candidate_rows_by_length_only"] for value in by_label.values()),
        "identical_candidate_narratives_with_multiple_labels": sum(len(value) > 1 for value in narrative_labels.values()),
        "limits": "Length/structure audit only. Not an approved training set, semantic label audit, dataset split or classifier evaluation.",
        "model_calls": 0,
        "training": False,
    }
    destination = ROOT / "data/audits/source-audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
