"""One greedy raw-completion baseline, development only, checkpointed by batch."""
import json
import os
import time
from baseline_common import ROOT, LABELS, development_rows, metrics, parse_completion, prompt, sha
from verify_local_model import verified_load
import mlx.core as mx
from mlx_lm.generate import batch_generate
from mlx_lm.sample_utils import make_sampler


def main():
    out = ROOT / 'data/evaluations/untuned-development-v1'
    plan = json.loads((ROOT/'data/baseline-plan-v1.json').read_text())
    verification = json.loads((ROOT/'data/evaluations/local-model-verification.json').read_text())
    assert verification['boundary']['passed'] and verification['finite_logits']
    signature = {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in [ROOT/'scripts/baseline_common.py', ROOT/'scripts/run_model_baseline.py', ROOT/'scripts/verify_local_model.py', ROOT/'data/baseline-plan-v1.json', ROOT/'data/model-manifest.json', ROOT/'requirements-model.lock']}
    if out.exists():
        assert json.loads((out/'run-signature.json').read_text()) == signature, 'Cannot resume changed code/contract'
        if (out/'metrics.json').exists():
            raise SystemExit('Baseline already complete; refusing overwrite')
    else:
        out.mkdir(parents=True)
        (out/'run-signature.json').write_text(json.dumps(signature, indent=2)+'\n')
    rows = development_rows()
    path = out/'predictions.jsonl'
    records = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    assert [r['event_id'] for r in records] == [r['event_id'] for r in rows[:len(records)]]
    assert all(r['reference_label'] == rows[i]['reference_label'] for i,r in enumerate(records))
    model, tokenizer, pin, load_seconds = verified_load()
    mx.random.seed(0)
    batch = plan['decoding']['batch_size']
    start = time.monotonic()
    for offset in range(len(records), len(rows), batch):
        selected = rows[offset:offset+batch]
        prompts = [tokenizer.encode(prompt(r['text']), add_special_tokens=False) for r in selected]
        response = batch_generate(model, tokenizer, prompts, max_tokens=plan['decoding']['max_tokens'], sampler=make_sampler(temp=0), completion_batch_size=batch, prefill_batch_size=batch, return_token_ids=True)
        new = []
        for r,text,ids in zip(selected, response.texts, response.token_ids):
            new.append({'event_id': r['event_id'], 'reference_label': r['reference_label'], 'prediction': parse_completion(text), 'raw_completion': text, 'token_ids': ids, 'output_tokens': len(ids)})
        assert len(new) == len(selected)
        with path.open('a') as f:
            for r in new:
                f.write(json.dumps(r, sort_keys=True)+'\n')
            f.flush()
            os.fsync(f.fileno())
        records.extend(new)
        if offset == 0 or len(records) % 80 == 0 or len(records) == len(rows):
            print(f'{len(records)}/{len(rows)} complete; session {time.monotonic()-start:.1f}s; peak {mx.get_peak_memory()/1e9:.2f} GB', flush=True)
    assert len(records) == len(rows) == 2449
    report = metrics(records) | {'split': 'development', 'model_revision': pin['revision'], 'model_manifest_sha256': sha((ROOT/'data/model-manifest.json').read_bytes()), 'plan_sha256': sha((ROOT/'data/baseline-plan-v1.json').read_bytes()), 'predictions_sha256': sha(path.read_bytes()), 'max_token_budget_completions': sum(r['output_tokens'] == plan['decoding']['max_tokens'] for r in records), 'peak_memory_gb': mx.get_peak_memory()/1e9, 'last_session_seconds': time.monotonic()-start, 'load_seconds': load_seconds, 'note': 'Unconstrained greedy raw generation; no label scoring, chat template, adapter, quantization or prompt sweep.'}
    (out/'metrics.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
