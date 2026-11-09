"""Development-only handout-style baseline; no fine-tuning or generated-label fallback."""
import json
import os
import time
from baseline_common import ROOT, LABELS, development_rows, metrics, sha
from letter_choice import CHOICES, choice_ids, letter_prompt
from verify_local_model import verified_load
import mlx.core as mx


def main():
    out = ROOT/'data/evaluations/letter-development-v1'
    plan_path = ROOT/'data/letter-baseline-plan-v1.json'
    plan = json.loads(plan_path.read_text())
    signature = {str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in [plan_path, ROOT/'scripts/letter_choice.py', ROOT/'scripts/run_letter_baseline.py', ROOT/'scripts/baseline_common.py', ROOT/'scripts/verify_local_model.py', ROOT/'data/model-manifest.json', ROOT/'requirements-model.lock']}
    if out.exists():
        assert json.loads((out/'run-signature.json').read_text()) == signature
        if (out/'metrics.json').exists():
            raise SystemExit('Refusing to overwrite completed baseline')
    else:
        out.mkdir(parents=True)
        (out/'run-signature.json').write_text(json.dumps(signature, indent=2)+'\n')
    rows = development_rows()
    path = out/'predictions.jsonl'
    records = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    assert [r['event_id'] for r in records] == [r['event_id'] for r in rows[:len(records)]]
    model, tokenizer, pin, load_time = verified_load()
    ids, forms = choice_ids(tokenizer)
    prompts = []
    for r in rows:
        p = letter_prompt(tokenizer, r['text'])
        tokens = tokenizer.encode(p, add_special_tokens=False)
        for i,form in enumerate(forms):
            assert tokenizer.encode(p+form, add_special_tokens=False) == tokens+[ids[i]], 'Choice boundary mismatch'
        prompts.append(tokens)
    mx.random.seed(0)
    start = time.monotonic()
    batch = plan['batch_size']
    for offset in range(len(records), len(rows), batch):
        selected = prompts[offset:offset+batch]
        width = max(map(len, selected))
        padded = [p+[tokenizer.eos_token_id]*(width-len(p)) for p in selected]
        # Right padding is strictly after each scored position, so causal attention
        # cannot see it. Score the last real token at its original position.
        logits = model(mx.array(padded))
        scores = logits[mx.arange(len(selected)), mx.array([len(p)-1 for p in selected])][:, mx.array(ids)].astype(mx.float32)
        probs = mx.softmax(scores, axis=-1)
        mx.eval(scores, probs)
        assert bool(mx.all(mx.isfinite(scores)).item())
        values, probabilities = scores.tolist(), probs.tolist()
        new = []
        for r,ls,ps in zip(rows[offset:offset+batch], values, probabilities):
            best = max(range(6), key=lambda i: ls[i])
            new.append({'event_id':r['event_id'], 'reference_label':r['reference_label'], 'prediction':LABELS[best], 'choice':CHOICES[best], 'choice_logits':ls, 'choice_normalized_probabilities':ps})
        with path.open('a') as f:
            for r in new:
                f.write(json.dumps(r, sort_keys=True)+'\n')
            f.flush()
            os.fsync(f.fileno())
        records.extend(new)
        del logits, scores, probs
        if len(records) % 160 == 0 or offset == 0 or len(records) == len(rows):
            print(f'{len(records)}/{len(rows)} complete; session {time.monotonic()-start:.1f}s; peak {mx.get_peak_memory()/1e9:.2f} GB', flush=True)
    assert len(records) == len(rows) == 2449
    report = metrics(records) | {'split':'development', 'model_revision':pin['revision'], 'plan_sha256':sha(plan_path.read_bytes()), 'model_manifest_sha256':sha((ROOT/'data/model-manifest.json').read_bytes()), 'predictions_sha256':sha(path.read_bytes()), 'choice_ids':dict(zip(CHOICES,ids)), 'choice_forms':dict(zip(CHOICES,forms)), 'choice_label_mapping':dict(zip(CHOICES,LABELS)), 'choice_boundary_checks':len(rows)*6, 'prompt_tokens_min':min(map(len,prompts)), 'prompt_tokens_max':max(map(len,prompts)), 'last_session_seconds':time.monotonic()-start, 'peak_memory_gb':mx.get_peak_memory()/1e9, 'load_seconds':load_time, 'method':'handout constrained first-step letter logits; explicit chat template; no adapter or quantization'}
    (out/'metrics.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
