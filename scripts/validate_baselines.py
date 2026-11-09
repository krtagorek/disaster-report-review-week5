"""Validate saved development predictions/metrics and letter batching behavior."""
import argparse
import json
from pathlib import Path
from baseline_common import ROOT, LABELS, development_rows, metrics, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-letter-batching', action='store_true')
    args = parser.parse_args()
    rows = development_rows()
    expected = [(r['event_id'],r['reference_label']) for r in rows]
    evidence = {}
    letter_records = None
    for name in ['keyword-development-v1','untuned-development-v1','letter-development-v1']:
        out = ROOT/'data/evaluations'/name
        data = (out/'predictions.jsonl').read_bytes()
        records = [json.loads(l) for l in data.splitlines()]
        saved = json.loads((out/'metrics.json').read_text())
        assert [(r['event_id'],r['reference_label']) for r in records] == expected
        assert len(set(r['event_id'] for r in records)) == 2449
        assert saved['predictions_sha256'] == sha(data)
        recomputed = metrics(records)
        for key,value in recomputed.items():
            assert saved[key] == value, (name,key)
        if (out/'run-signature.json').exists():
            signature = json.loads((out/'run-signature.json').read_text())
            assert all(sha((ROOT/p).read_bytes()) == d for p,d in signature.items()), name
        evidence[name] = {'rows':len(records), 'unique_ids_and_order_match_development':True, 'metrics_recomputed_exactly':True, 'predictions_sha256':sha(data), 'metrics_sha256':sha((out/'metrics.json').read_bytes())}
        if name == 'letter-development-v1':
            letter_records = records
    report = {'development_only':True, 'evaluations':evidence}
    if args.check_letter_batching:
        from verify_local_model import verified_load
        from letter_choice import choice_ids, letter_prompt
        import mlx.core as mx
        model, tokenizer, _, _ = verified_load()
        ids, _ = choice_ids(tokenizer)
        differences = []
        for row, saved in zip(rows[:8], letter_records[:8]):
            tokens = tokenizer.encode(letter_prompt(tokenizer,row['text']), add_special_tokens=False)
            scores = model(mx.array([tokens]))[0,-1,mx.array(ids)].astype(mx.float32)
            mx.eval(scores)
            values = scores.tolist()
            differences.append(max(abs(a-b) for a,b in zip(values,saved['choice_logits'])))
            best = max(range(6),key=lambda i:values[i])
            assert LABELS[best] == saved['prediction'], 'Batch/unbatched prediction mismatch'
            assert bool(mx.allclose(scores,mx.array(saved['choice_logits']),atol=.5,rtol=.01).item()), 'Unexpected batch score drift'
        report['letter_batch_check'] = {'rows':8,'unbatched_predictions_match':True, 'maximum_absolute_logit_difference':max(differences), 'logit_tolerance':{'atol':.5,'rtol':.01},'note':'BF16 kernels can differ slightly with batch shape; predictions must match on this check.'}
    out = ROOT/'data/evaluations/validation-v1.json'
    if out.exists():
        raise SystemExit('Refusing to overwrite validation evidence')
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
