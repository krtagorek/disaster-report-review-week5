"""Protect v1 and verify frontend-only scope without running model experiments."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
TAG='working-review-app-v1'
CHECKPOINT='0297d9d652a6493c9e31153e1ed2a39c4f453beb'


def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)


def verify():
    assert git('rev-parse',TAG+'^{}').decode().strip()==CHECKPOINT,'Checkpoint tag moved'
    assert git('rev-parse','main').decode().strip()==CHECKPOINT,'Main checkpoint branch changed'
    protected=['review_app/core.py','review_app/model.py','review_app/server.py','review_app/__main__.py','scripts/baseline_common.py','scripts/evaluate_lora.py','scripts/letter_choice.py','requirements-model.lock','data']
    assert not git('diff',TAG,'--name-only','--',*protected).strip(),'Inference, data or recorded results changed'
    baseline=json.loads(git('show',TAG+':data/evaluations/review-app-evidence.json'))
    for name,expected in baseline['files_sha256'].items():
        assert hashlib.sha256(git('show',TAG+':'+name)).hexdigest()==expected,'Checkpoint evidence mismatch: '+name
    subprocess.run([sys.executable,'scripts/build_demo_evidence.py','--check'],cwd=ROOT,check=True)
    return {'checkpoint_commit':CHECKPOINT,'checkpoint_tag':TAG,'tag_and_main_unchanged':True,'baseline_evidence_at_tag_valid':True,'inference_and_data_diff_empty':True,'recorded_snapshot_matches_sources':True,'reserved_final_evaluated':False,'training_runs_added':0}


if __name__=='__main__':print(json.dumps(verify(),indent=2))
