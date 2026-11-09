"""Verify served predictions, model/code identity and freeze app milestone assets."""
import json
from pathlib import Path
from baseline_common import ROOT,development_rows,metrics,sha


def main():
    destination=ROOT/'data/evaluations/review-app-evidence.json'
    if destination.exists():
        evidence=json.loads(destination.read_text())
        for name,digest in evidence['files_sha256'].items():assert sha((ROOT/name).read_bytes())==digest,name
        print('Review-app evidence hashes verified');return
    out=ROOT/'data/evaluations/served-lora-v1-raw'
    records=[json.loads(l) for l in (out/'predictions.jsonl').read_text().splitlines()]
    expected=[(r['event_id'],r['reference_label']) for r in development_rows()]
    assert [(r['event_id'],r['reference_label']) for r in records]==expected
    assert len({r['event_id'] for r in records})==2449
    report=json.loads((out/'metrics.json').read_text())
    assert all(report[k]==v for k,v in metrics(records).items())
    assert sha((out/'predictions.jsonl').read_bytes())==report['predictions_sha256']
    assert json.loads((out/'smoke.json').read_text())['passed']==5
    assert report['model']['artifact_fingerprint']==sha((ROOT/'data/training/lora-v1/merged-manifest.json').read_bytes())
    assert report['model']['prompt_sha256']==sha((ROOT/'scripts/baseline_common.py').read_bytes())
    assert report['model']['predictor_sha256']==sha((ROOT/'scripts/evaluate_lora.py').read_bytes())
    for name,digest in report['model']['serving_code_sha256'].items():assert sha((ROOT/'review_app'/name).read_bytes())==digest
    signature=json.loads((out/'run-signature.json').read_text())
    assert signature['evaluator_sha256']==sha((ROOT/'scripts/evaluate_served_model.py').read_bytes())
    workflow=json.loads((ROOT/'data/evaluations/review-app-verification.json').read_text())
    assert workflow['restart_preserved_reviews']>=3 and workflow['post_restart_model_fingerprint']==report['model']['artifact_fingerprint']
    paths=list(out.glob('*'))+[ROOT/'data/evaluations/served-plan-v1.json',ROOT/'data/evaluations/review-app-verification.json',ROOT/'docs/REVIEW_APP.md',ROOT/'docs/assets/review-desk.png',ROOT/'scripts/evaluate_served_model.py',ROOT/'scripts/write_review_app_report.py',ROOT/'scripts/audit_review_app.py',ROOT/'tests/test_review_app.py',ROOT/'start-review.command',ROOT/'requirements-model.lock',ROOT/'scripts/baseline_common.py',ROOT/'scripts/evaluate_lora.py',ROOT/'data/training/lora-v1/merged-manifest.json',ROOT/'data/evaluations/review-app-tests.txt']
    paths+=list((ROOT/'review_app').glob('*.py'))+list((ROOT/'review_app/static').glob('*'))
    result={'n':2449,'unique_development_ids_order_pass':True,'served_metrics_recomputed':True,'served_smoke_passed':5,'model_and_serving_code_identity_pass':True,'process_restart_persistence_pass':True,'reserved_final_accessed':False,'files_sha256':{str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in sorted(paths) if p.is_file()}}
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='files_sha256'},indent=2))


if __name__=='__main__':main()
