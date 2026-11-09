"""Five fixed merged smoke cases, then matched development-only evaluation."""
import argparse
import json
import os
import time
from baseline_common import ROOT,LABELS,development_rows,prompt,parse_completion,metrics,sha
from letter_choice import CHOICES,choice_ids,letter_prompt
from lora_artifacts import RUN,load_candidate
import mlx.core as mx
from mlx_lm.generate import batch_generate
from mlx_lm.sample_utils import make_sampler


def raw_predict(model,tok,texts):
    response=batch_generate(model,tok,[tok.encode(prompt(text),add_special_tokens=False) for text in texts],max_tokens=16,sampler=make_sampler(temp=0),prefill_batch_size=8,completion_batch_size=8,return_token_ids=True)
    return [{'prediction':parse_completion(text),'raw_completion':text,'token_ids':ids,'output_tokens':len(ids)} for text,ids in zip(response.texts,response.token_ids)]


def letter_predict(model,tok,texts):
    ids,forms=choice_ids(tok)
    inputs=[]
    for text in texts:
        p=letter_prompt(tok,text)
        encoded=tok.encode(p,add_special_tokens=False)
        for form,id_ in zip(forms,ids):
            assert tok.encode(p+form,add_special_tokens=False)==encoded+[id_]
        inputs.append(encoded)
    width=max(map(len,inputs))
    x=mx.array([p+[tok.eos_token_id]*(width-len(p)) for p in inputs])
    logits=model(x)
    scores=logits[mx.arange(len(inputs)),mx.array([len(p)-1 for p in inputs])][:,mx.array(ids)].astype(mx.float32)
    probs=mx.softmax(scores,axis=-1)
    mx.eval(scores,probs)
    assert bool(mx.all(mx.isfinite(scores)).item())
    result=[]
    for values,probabilities in zip(scores.tolist(),probs.tolist()):
        best=max(range(6),key=lambda i:values[i])
        result.append({'prediction':LABELS[best],'choice':CHOICES[best],'choice_logits':values,'choice_normalized_probabilities':probabilities})
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--artifact',choices=['adapter','merged'],required=True)
    parser.add_argument('--protocol',choices=['raw','letter'],default='raw')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.smoke:
        if args.protocol!='raw': raise SystemExit('Fixed smoke protocol is raw generation')
        output=RUN/f'{args.artifact}-smoke.json'
        if output.exists(): raise SystemExit('Refusing smoke overwrite')
    else:
        gate=json.loads((RUN/'merged-smoke.json').read_text())
        assert gate['passed']==5 and gate['total']==5,'Merged smoke gate failed'
        out=ROOT/f'data/evaluations/{args.artifact}-lora-v1-{args.protocol}'
        output=out/'metrics.json'
        if output.exists(): raise SystemExit('Refusing evaluation overwrite')
    model,tok,fingerprint=load_candidate(args.artifact)
    mx.random.seed(0)
    if args.smoke:
        cases=json.loads((ROOT/'data/baseline-plan-v1.json').read_text())['smoke_cases_for_future_merged_artifact']
        predictions=raw_predict(model,tok,[c['text'] for c in cases])
        results=[dict(case,**pred,correct=case['expected']==pred['prediction']) for case,pred in zip(cases,predictions)]
        report={'artifact':args.artifact,'artifact_fingerprint':fingerprint,'total':5,'passed':sum(r['correct'] for r in results),'protocol':'identical raw prompt, greedy16, exact label parser','cases':results}
        output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
        if report['passed']!=5: raise SystemExit('Smoke failure; do not start full merged evaluation')
        return
    paths=[ROOT/'scripts/evaluate_lora.py',ROOT/'scripts/lora_artifacts.py',ROOT/'scripts/baseline_common.py',ROOT/'scripts/letter_choice.py',ROOT/'requirements-model.lock',RUN/'plan.json']
    signature={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in paths}|{'artifact_fingerprint':fingerprint}
    if out.exists():
        assert json.loads((out/'run-signature.json').read_text())==signature
    else:
        out.mkdir(parents=True)
        (out/'run-signature.json').write_text(json.dumps(signature,indent=2)+'\n')
    rows=development_rows()
    path=out/'predictions.jsonl'
    records=[json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    assert [(r['event_id'],r['reference_label']) for r in records]==[(r['event_id'],r['reference_label']) for r in rows[:len(records)]]
    predictor=raw_predict if args.protocol=='raw' else letter_predict
    start=time.monotonic()
    for offset in range(len(records),len(rows),8):
        selected=rows[offset:offset+8]
        predictions=predictor(model,tok,[r['text'] for r in selected])
        assert len(predictions)==len(selected)
        new=[{'event_id':r['event_id'],'reference_label':r['reference_label'],**pred} for r,pred in zip(selected,predictions)]
        with path.open('a') as f:
            for r in new:f.write(json.dumps(r,sort_keys=True)+'\n')
            f.flush();os.fsync(f.fileno())
        records.extend(new)
        if len(records)%160==0 or offset==0 or len(records)==len(rows):
            print(f'{args.artifact}/{args.protocol}: {len(records)}/{len(rows)}; {time.monotonic()-start:.1f}s',flush=True)
    assert len(records)==2449
    result=metrics(records)|{'artifact':args.artifact,'protocol':args.protocol,'artifact_fingerprint':fingerprint,'predictions_sha256':sha(path.read_bytes()),'split':'development','peak_memory_gb':mx.get_peak_memory()/1e9,'seconds':time.monotonic()-start,'max_token_budget_completions':sum(r.get('output_tokens')==16 for r in records) if args.protocol=='raw' else None}
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
