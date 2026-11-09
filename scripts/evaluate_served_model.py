"""Evaluate the actual single-report HTTP route without sending labels or IDs."""
import argparse
import http.client
import json
import os
import statistics
import time
from baseline_common import ROOT, development_rows, metrics, sha


def request(port,path,body=None):
    connection=http.client.HTTPConnection('127.0.0.1',port,timeout=120)
    try:
        headers={'X-Review-App':'1','Content-Type':'application/json'}
        connection.request('POST' if body is not None else 'GET',path,body=json.dumps(body,ensure_ascii=False).encode() if body is not None else None,headers=headers)
        response=connection.getresponse();result=json.loads(response.read())
        if response.status!=200:raise RuntimeError(f'HTTP {response.status}: {result}')
        return result
    finally:connection.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8877);args=parser.parse_args()
    health=request(args.port,'/api/health')
    assert health.get('service')=='disaster-report-review' and health['ready'], 'Wrong service; do not send report text'
    merged=ROOT/'data/training/lora-v1/merged-manifest.json'
    assert health['model']['artifact_fingerprint']==sha(merged.read_bytes())
    assert health['model']['prompt_sha256']==sha((ROOT/'scripts/baseline_common.py').read_bytes())
    assert health['model']['predictor_sha256']==sha((ROOT/'scripts/evaluate_lora.py').read_bytes())
    for name,digest in health['model']['serving_code_sha256'].items():
        assert sha((ROOT/'review_app'/name).read_bytes())==digest
    out=ROOT/'data/evaluations/served-lora-v1-raw'
    if (out/'metrics.json').exists():raise SystemExit('Refusing served evidence overwrite')
    signature={'model':health['model'],'plan_sha256':sha((ROOT/'data/evaluations/served-plan-v1.json').read_bytes()),'evaluator_sha256':sha(__import__('pathlib').Path(__file__).read_bytes())}
    if out.exists():assert json.loads((out/'run-signature.json').read_text())==signature
    else:
        out.mkdir(parents=True);(out/'run-signature.json').write_text(json.dumps(signature,indent=2)+'\n')
    smoke_path=out/'smoke.json'
    if not smoke_path.exists():
        cases=json.loads((ROOT/'data/baseline-plan-v1.json').read_text())['smoke_cases_for_future_merged_artifact']
        results=[]
        for case in cases:
            result=request(args.port,'/api/classify',{'text':case['text']})
            assert result['model']==health['model']
            results.append({**case,'prediction':result['prediction'],'raw_completion':result['raw_completion'],'token_ids':result['token_ids'],'correct':case['expected']==result['prediction']})
        smoke_path.write_text(json.dumps({'passed':sum(r['correct'] for r in results),'total':5,'cases':results},indent=2)+'\n')
    assert json.loads(smoke_path.read_text())['passed']==5,'Served smoke gate failed'
    print('Served smoke: 5/5 passed',flush=True)
    rows=development_rows();path=out/'predictions.jsonl'
    records=[json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    assert [(r['event_id'],r['reference_label']) for r in records]==[(r['event_id'],r['reference_label']) for r in rows[:len(records)]]
    start=time.monotonic()
    for row in rows[len(records):]:
        tick=time.monotonic()
        result=request(args.port,'/api/classify',{'text':row['text']})
        assert result['model']==health['model']
        record={'event_id':row['event_id'],'reference_label':row['reference_label'],**{k:result[k] for k in ['prediction','raw_completion','token_ids','output_tokens']},'http_seconds':time.monotonic()-tick}
        with path.open('a') as f:
            f.write(json.dumps(record,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
        records.append(record)
        if len(records)%100==0 or len(records)==1 or len(records)==len(rows):print(f'Served single-report: {len(records)}/{len(rows)}; {time.monotonic()-start:.1f}s',flush=True)
    assert len(records)==2449
    offline=[json.loads(l) for l in (ROOT/'data/evaluations/merged-lora-v1-raw/predictions.jsonl').read_text().splitlines()]
    assert [(r['event_id'],r['reference_label']) for r in offline]==[(r['event_id'],r['reference_label']) for r in records]
    differences=[{'event_id':a['event_id'],'reference_label':a['reference_label'],'offline_prediction':a['prediction'],'served_prediction':b['prediction']} for a,b in zip(offline,records) if a['prediction']!=b['prediction']]
    times=sorted(r['http_seconds'] for r in records)
    report=metrics(records)|{'model':health['model'],'split':'development','request_batch_size':1,'predictions_sha256':sha(path.read_bytes()),'offline_batch8_prediction_disagreements':len(differences),'disagreement_events':differences,'offline_raw_completion_differences':sum(a['raw_completion']!=b['raw_completion'] for a,b in zip(offline,records)),'offline_token_sequence_differences':sum(a['token_ids']!=b['token_ids'] for a,b in zip(offline,records)),'http_latency_seconds':{'median':statistics.median(times),'p95':times[int(.95*(len(times)-1))],'total':sum(times)},'last_session_seconds':time.monotonic()-start,'final_accessed':False}
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
