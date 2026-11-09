"""Persistence, provenance and real HTTP contract tests without GPU/model loading."""
import csv
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.request import Request, build_opener, ProxyHandler
urlopen=build_opener(ProxyHandler({})).open
from urllib.error import HTTPError
from review_app.core import Store, Problem, ROOT
from review_app.server import AppServer

TEXT='A tornado crossed a farm, damaging a barn.\nThe survey confirmed the path, "EF1".'


class Archive:
    def get(self,identifier):
        if identifier!='dev-1':raise Problem('Not in archive',404)
        return {'event_id':'dev-1','episode_id':'episode-1','text':TEXT,'reference_label':'Tornado',
                'attribution':'Source: NOAA test fixture','source_url':'https://example.org/source','state':'TEST'}
    def search(self,query=''):return []


class Model:
    metadata={'run_id':'test','artifact_fingerprint':'test-artifact','prompt_sha256':'test-prompt'}
    def classify(self,text):
        from review_app.core import valid_text
        valid_text(text)
        return {'prediction':'Tornado','raw_completion':'Tornado','model':self.metadata}


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/"work")
        self.path=Path(self.temp.name)/'reviews.sqlite3'
        self.store=Store(self.path)
        self.archive=Archive()
        self.suggestion=self.store.classification(TEXT,Model().classify(TEXT))

    def tearDown(self):self.temp.cleanup()

    def body(self,**changes):
        return {'text':TEXT,'classification_id':self.suggestion['id'],'review_state':'confirmed','selected_label':'Tornado',**changes}

    def test_save_reopen_correct_unresolved_and_history(self):
        r=self.store.create(self.body(archive_event_id='dev-1'),self.archive)
        reopened=Store(self.path).get(r['id'])
        self.assertEqual(reopened,r)
        corrected=self.store.update(r['id'],{'version':1,'review_state':'corrected','selected_label':'Thunderstorm Wind','note':'Survey reinterpretation'})
        self.assertEqual(corrected['source']['reference_label'],'Tornado')
        self.assertEqual(corrected['suggestion']['prediction'],'Tornado')
        unresolved=self.store.update(r['id'],{'version':2,'review_state':'unresolved','selected_label':None,'unresolved_reason':'ambiguous','note':'Two hazards'})
        self.assertIsNone(unresolved['selected_label'])
        with self.store.connect() as db:self.assertEqual(db.execute('SELECT count(*) FROM review_history').fetchone()[0],3)
        with self.assertRaises(Problem) as raised:self.store.update(r['id'],{'version':1,'review_state':'pending'})
        self.assertEqual(raised.exception.status,409)

    def test_provenance_cannot_be_forged_or_moved(self):
        for body in [self.body(text=TEXT+' changed'),self.body(text=TEXT+' changed',archive_event_id='dev-1'),self.body(archive_event_id='final-or-unknown')]:
            with self.assertRaises(Problem):self.store.create(body,self.archive)
        pasted=self.store.create(self.body(),self.archive)
        self.assertIsNone(pasted['source'])
        with self.assertRaises(Problem):self.store.update(pasted['id'],{'version':1,'text':'replacement','review_state':'pending'})
        with self.assertRaises(Problem):self.store.update(pasted['id'],{'version':1,'classification_id':self.suggestion['id'],'review_state':'pending'})

    def test_pending_can_be_classified_after_reopen(self):
        saved=self.store.create({'text':TEXT,'review_state':'pending'},self.archive)
        updated=self.store.update(saved['id'],{'version':1,'classification_id':self.suggestion['id'],'review_state':'confirmed','selected_label':'Tornado'})
        self.assertEqual(updated['suggestion']['id'],self.suggestion['id'])

    def test_decisions_reject_inconsistent_states(self):
        for changes in [{'selected_label':'Hail'},{'review_state':'unresolved','selected_label':None},
                        {'review_state':'corrected'},{'review_state':'pending','selected_label':'Tornado'}]:
            with self.assertRaises(Problem):self.store.create(self.body(**changes),self.archive)
        invalid=self.store.classification(TEXT,{'prediction':None,'raw_completion':'unsupported','model':Model.metadata})
        with self.assertRaises(Problem):self.store.create(self.body(classification_id=invalid['id']),self.archive)
        manual=self.store.create({'text':TEXT,'review_state':'unresolved','unresolved_reason':'out_of_scope'},self.archive)
        self.assertIsNone(manual['suggestion'])

    def test_exports_round_trip_and_filter(self):
        record=self.store.create(self.body(archive_event_id='dev-1'),self.archive)
        self.store.create({'text':'Uncertain text','review_state':'unresolved','unresolved_reason':'insufficient_text'},self.archive)
        csv_data,_=self.store.export('csv','confirmed')
        exported=list(csv.DictReader(io.StringIO(csv_data.decode())))
        self.assertEqual(len(exported),1)
        self.assertEqual(exported[0]['original_text'],TEXT)
        self.assertEqual(exported[0]['event_id'],'dev-1')
        self.assertEqual(exported[0]['model_fingerprint'],'test-artifact')
        data,_=self.store.export('json','confirmed')
        self.assertEqual(json.loads(data)['reviews'],[record])


class HttpTests(unittest.TestCase):
    def test_http_workflow_and_local_request_guards(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"work") as directory:
            server=AppServer(('127.0.0.1',0),Store(Path(directory)/'reviews.sqlite3'),Archive(),Model())
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            base=f'http://127.0.0.1:{server.server_address[1]}'
            def call(path,body=None,method=None,headers=None):
                request=Request(base+path,data=json.dumps(body).encode() if body is not None else None,
                    method=method,headers={'Content-Type':'application/json','X-Review-App':'1',**(headers or {})})
                with urlopen(request) as response:return json.load(response)
            try:
                s=call('/api/classify',{'text':TEXT})
                r=call('/api/reports',{'text':TEXT,'classification_id':s['id'],'review_state':'confirmed','selected_label':'Tornado'})
                self.assertEqual(call('/api/reports/'+r['id'])['text'],TEXT)
                for headers in [{'Host':'evil.example'},{'Origin':'https://evil.example'},{'X-Review-App':'0'}]:
                    with self.assertRaises(HTTPError) as raised:call('/api/classify',{'text':TEXT},headers=headers)
                    self.assertEqual(raised.exception.code,403)
                with self.assertRaises(HTTPError) as raised:call('/api/classify',{'text':'   '})
                self.assertEqual(raised.exception.code,400)
                with self.assertRaises(HTTPError):call('/api/archive/final-or-unknown')
            finally:server.shutdown();server.server_close();thread.join()


if __name__=='__main__':unittest.main()
