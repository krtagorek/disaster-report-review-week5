"""Shared narrative-only contract and development-only evaluation utilities."""
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = json.loads((ROOT / 'data/source-manifest.json').read_text())['initial_classes']


def sha(data):
    return hashlib.sha256(data).hexdigest()


def prompt(text):
    return ('Classify this historical disaster report. Choose exactly one category from: '
            + ', '.join(LABELS) + '.\nReturn only the category name, with no explanation.\n'
            + 'Report: ' + json.dumps(text, ensure_ascii=False) + '\nCategory:\n')


def development_rows():
    # Deliberately no split argument: this milestone cannot evaluate final.
    freeze = json.loads((ROOT / 'data/splits/v1/freeze.json').read_text())
    data = (ROOT / 'data/splits/v1/development.jsonl').read_bytes()
    assert sha(data) == freeze['artifact_sha256']['development.jsonl']
    records = [json.loads(line) for line in data.splitlines()]
    wanted = {r['event_id'] for r in records}
    source_manifest = json.loads((ROOT / 'data/source-manifest.json').read_text())
    source = ROOT / source_manifest['local_file']
    assert sha(source.read_bytes()) == source_manifest['sha256']
    with gzip.open(source, 'rt', encoding='utf-8-sig', newline='') as f:
        selected = {r['EVENT_ID']: r for r in csv.DictReader(f) if r['EVENT_ID'] in wanted}
    result = []
    for r in records:
        source_row = selected[r['event_id']]
        assert sha(source_row['EVENT_NARRATIVE'].encode()) == r['narrative_sha256']
        assert source_row['EVENT_TYPE'] == r['reference_label']
        assert source_row['EPISODE_ID'] == r['episode_id']
        result.append({**r, 'text': source_row['EVENT_NARRATIVE']})
    return result


def parse_completion(text):
    # No first-line extraction, fuzzy matching or invalid-to-label fallback.
    value = text.strip()
    return value if value in LABELS else None


def metrics(records):
    columns = LABELS + ['INVALID']
    matrix = {c: {p: 0 for p in columns} for c in LABELS}
    for r in records:
        matrix[r['reference_label']][r['prediction'] or 'INVALID'] += 1
    per_class = {}
    for c in LABELS:
        tp = matrix[c][c]
        support = sum(matrix[c].values())
        predicted = sum(matrix[t][c] for t in LABELS)
        precision = tp / predicted if predicted else 0.
        recall = tp / support if support else 0.
        per_class[c] = {'precision': precision, 'recall': recall, 'f1': 2*precision*recall/(precision+recall) if precision+recall else 0., 'support': support}
    n = len(records)
    return {'n': n, 'accuracy': sum(matrix[c][c] for c in LABELS)/n, 'macro_f1': sum(v['f1'] for v in per_class.values())/len(LABELS), 'invalid_count': sum(matrix[c]['INVALID'] for c in LABELS), 'per_class': per_class, 'confusion_matrix': matrix, 'confusion_orientation': 'rows=reference; columns=prediction; INVALID counts as incorrect'}


# Predeclared before predictions; simple keyword rules, no dev-driven revisions.
KEYWORDS = {
    'Flash Flood': r'\b(?:flash flood\w*|flood\w*|inundat\w*)\b',
    'Tornado': r'\b(?:tornado\w*|twister\w*|EF[0-5])\b',
    'Thunderstorm Wind': r'\b(?:wind\w*|gust\w*|tree\w*|power ?line\w*)\b',
    'Hail': r'\bhail\w*\b',
    'Winter Storm': r'\b(?:snow\w*|blizzard\w*|sleet|freezing rain|ice storm)\b',
    'Wildfire': r'\b(?:wildfire\w*|fire\w*|burn\w*|contain\w*)\b',
}


def keyword(text):
    import re
    scores = {c: len(re.findall(KEYWORDS[c], text, flags=re.I)) for c in LABELS}
    # Ties in declared label order; no-hit fallback is training-majority Wind.
    if not any(scores.values()):
        return 'Thunderstorm Wind'
    return max(LABELS, key=lambda c: scores[c])
