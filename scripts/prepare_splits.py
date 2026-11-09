"""Deterministic, prediction-free grouped split construction and freeze verification."""
import argparse
import collections
import csv
import gzip
import hashlib
import json
import math
import re
from pathlib import Path
from audit_source import normalize

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'episode-shingle-v1'
SPLITS = {'train': .70, 'development': .15, 'final': .15}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def shingles(text):
    # Mask numeric changes in repeated reports, preserving words and hazard names.
    tokens = re.findall(r'[a-z]+|#', re.sub(r'\d+(?:[.,]\d+)*', '#', normalize(text)))
    return set(zip(tokens, tokens[1:], tokens[2:]))


def near_pairs(sets):
    """Exact Jaccard >= 0.8 search with complete frequency-ordered prefix filter."""
    freq = collections.Counter(t for s in sets for t in s)
    index = collections.defaultdict(list)
    for i, s in enumerate(sets):
        if not s:
            continue
        ordered = sorted(s, key=lambda t: (freq[t], t))
        prefix = ordered[:len(s) - math.ceil(.8 * len(s)) + 1]
        possible = set(j for t in prefix for j in index[t])
        for j in sorted(possible):
            other = sets[j]
            if min(len(s), len(other)) * 5 < max(len(s), len(other)) * 4:
                continue
            intersection = len(s & other)
            if intersection * 5 >= len(s | other) * 4:
                yield j, i
        for t in prefix:
            index[t].append(i)


class UnionFind:
    def __init__(self, n):
        self.parents = list(range(n))

    def find(self, i):
        while self.parents[i] != i:
            self.parents[i] = self.parents[self.parents[i]]
            i = self.parents[i]
        return i

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parents[max(a, b)] = min(a, b)


def build():
    manifest = json.loads((ROOT / 'data/source-manifest.json').read_text())
    raw = (ROOT / manifest['local_file']).read_bytes()
    if digest(raw) != manifest['sha256']:
        raise ValueError('Source checksum mismatch')
    with gzip.open(ROOT / manifest['local_file'], 'rt', encoding='utf-8-sig', newline='') as f:
        source = list(csv.DictReader(f))
    labels = manifest['initial_classes']
    candidates = [r for r in source if r['EVENT_TYPE'] in labels and 80 <= len(r['EVENT_NARRATIVE'].strip()) <= 2000]
    texts = collections.defaultdict(set)
    for r in candidates:
        texts[normalize(r['EVENT_NARRATIVE'])].add(r['EVENT_TYPE'])
    excluded = sorted(r['EVENT_ID'] for r in candidates if len(texts[normalize(r['EVENT_NARRATIVE'])]) > 1)
    rows = sorted((r for r in candidates if r['EVENT_ID'] not in set(excluded)), key=lambda r: r['EVENT_ID'])
    if len({r['EVENT_ID'] for r in rows}) != len(rows) or any(not r['EPISODE_ID'].strip() for r in rows):
        raise ValueError('Duplicate event IDs or missing episode IDs')
    uf = UnionFind(len(rows))
    episodes, exact = {}, {}
    for i, r in enumerate(rows):
        for table, key in ((episodes, r['EPISODE_ID']), (exact, normalize(r['EVENT_NARRATIVE']))):
            if key in table:
                uf.union(i, table[key])
            else:
                table[key] = i
    print('Finding near duplicates...', flush=True)
    edges = list(near_pairs([shingles(r['EVENT_NARRATIVE']) for r in rows]))
    for a, b in edges:
        uf.union(a, b)
    groups = collections.defaultdict(list)
    for i in range(len(rows)):
        groups[uf.find(i)].append(i)
    totals = collections.Counter(r['EVENT_TYPE'] for r in rows)
    counts = {s: collections.Counter() for s in SPLITS}
    assignment = {}
    group_ids = {}
    for members in sorted(groups.values(), key=lambda g: (-len(g), digest('|'.join(rows[i]['EVENT_ID'] for i in g).encode()))):
        histogram = collections.Counter(rows[i]['EVENT_TYPE'] for i in members)
        def cost(s):
            return sum(((counts[s][c] + n - totals[c]*SPLITS[s])**2 - (counts[s][c] - totals[c]*SPLITS[s])**2) / (totals[c]*SPLITS[s]) for c, n in histogram.items())
        split = min(SPLITS, key=lambda s: (cost(s), s))
        counts[split].update(histogram)
        gid = digest('|'.join(rows[i]['EVENT_ID'] for i in members).encode())
        for i in members:
            assignment[i], group_ids[i] = split, gid
    # Independent invariants on actual row assignments, not just group construction.
    for field in ('EPISODE_ID',):
        seen = collections.defaultdict(set)
        for i, r in enumerate(rows):
            seen[r[field]].add(assignment[i])
        assert all(len(s) == 1 for s in seen.values())
    seen = collections.defaultdict(set)
    for i, r in enumerate(rows):
        seen[normalize(r['EVENT_NARRATIVE'])].add(assignment[i])
    assert all(len(s) == 1 for s in seen.values())
    assert all(assignment[a] == assignment[b] for a, b in edges)
    for s in SPLITS:
        for c in labels:
            assert counts[s][c] >= (50 if s == 'train' else 20), (s, c, counts[s][c])
            assert len({group_ids[i] for i,r in enumerate(rows) if assignment[i] == s and r['EVENT_TYPE'] == c}) >= 5, (s, c, 'too few groups')
    files = {}
    for s in SPLITS:
        records = [{'event_id': r['EVENT_ID'], 'episode_id': r['EPISODE_ID'], 'group_id': group_ids[i], 'reference_label': r['EVENT_TYPE'], 'narrative_sha256': digest(r['EVENT_NARRATIVE'].encode())} for i,r in enumerate(rows) if assignment[i] == s]
        files[s + '.jsonl'] = ''.join(json.dumps(r, sort_keys=True) + '\n' for r in records).encode()
    files['near-duplicate-edges.jsonl'] = ''.join(json.dumps({'a': rows[a]['EVENT_ID'], 'b': rows[b]['EVENT_ID']}, sort_keys=True)+'\n' for a,b in edges).encode()
    report = {'version': VERSION, 'source_sha256': manifest['sha256'], 'target_fractions': SPLITS, 'excluded_event_ids': excluded, 'rows': len(rows), 'groups': len(groups), 'largest_group_rows': max(map(len, groups.values())), 'near_duplicate_edges': len(edges), 'cross_episode_near_duplicate_edges': sum(rows[a]['EPISODE_ID'] != rows[b]['EPISODE_ID'] for a,b in edges), 'counts': {s: dict(counts[s]) for s in SPLITS}, 'group_counts': {s: len({group_ids[i] for i in assignment if assignment[i] == s}) for s in SPLITS}, 'class_group_counts': {s: {c: len({group_ids[i] for i,r in enumerate(rows) if assignment[i] == s and r['EVENT_TYPE'] == c}) for c in labels} for s in SPLITS}, 'validation': {'episode_overlap': 0, 'exact_narrative_overlap': 0, 'near_duplicate_edge_overlap': 0, 'complete_unique_assignment': True, 'minimum_class_support_passed': True}, 'artifact_sha256': {k: digest(v) for k,v in files.items()}, 'builder_sha256': digest(Path(__file__).read_bytes())}
    files['freeze.json'] = (json.dumps(report, indent=2, sort_keys=True)+'\n').encode()
    return files, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true', help='Create frozen artifacts only if none exist')
    args = parser.parse_args()
    files, report = build()
    out = ROOT / 'data/splits/v1'
    if args.freeze:
        if out.exists():
            raise SystemExit('Refusing to overwrite existing frozen split directory; use verification mode.')
        out.mkdir(parents=True)
        for name, data in files.items():
            (out/name).write_bytes(data)
    else:
        for name, data in files.items():
            if (out/name).read_bytes() != data:
                raise SystemExit(f'Frozen artifact mismatch: {name}')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
