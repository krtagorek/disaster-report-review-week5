"""Frozen simple keyword baseline on all development rows only."""
import json
from baseline_common import ROOT, KEYWORDS, development_rows, keyword, metrics, sha

out = ROOT / 'data/evaluations/keyword-development-v1'
if out.exists():
    raise SystemExit('Refusing to overwrite existing baseline')
rows = development_rows()
records = [{k: r[k] for k in ('event_id', 'reference_label')} | {'prediction': keyword(r['text'])} for r in rows]
out.mkdir(parents=True)
data = ''.join(json.dumps(r, sort_keys=True)+'\n' for r in records).encode()
(out / 'predictions.jsonl').write_bytes(data)
report = metrics(records) | {'rules': KEYWORDS, 'tie_policy': 'manifest label order', 'no_hit_fallback': 'Thunderstorm Wind (training majority)', 'predictions_sha256': sha(data), 'split': 'development', 'source_code_sha256': sha((ROOT / 'scripts/baseline_common.py').read_bytes())}
(out / 'metrics.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
