"""Review persistence and archive linkage, independent of GPU/model imports."""
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import sys
from datetime import datetime, timezone
from contextlib import contextmanager
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from baseline_common import LABELS, development_rows, sha

STATES = ('pending', 'confirmed', 'corrected', 'unresolved')
REASONS = ('ambiguous', 'out_of_scope', 'insufficient_text', 'invalid_model_output', 'other')


class Problem(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def now():
    return datetime.now(timezone.utc).isoformat()


def valid_text(text):
    if not isinstance(text, str) or not text.strip():
        raise Problem('Enter a report before continuing.')
    if len(text) > 20000:
        raise Problem('Reports can contain at most 20,000 characters.')
    return text


class Archive:
    """Only the frozen development IDs are available to the review application."""
    def __init__(self):
        manifest = json.loads((ROOT/'data/source-manifest.json').read_text())
        selected = {r['event_id']:r for r in development_rows()}
        self.records = {}
        with gzip.open(ROOT/manifest['local_file'], 'rt', encoding='utf-8-sig', newline='') as handle:
            for r in csv.DictReader(handle):
                if r['EVENT_ID'] in selected:
                    self.records[r['EVENT_ID']] = {
                        'event_id':r['EVENT_ID'], 'episode_id':r['EPISODE_ID'],
                        'text':r['EVENT_NARRATIVE'], 'reference_label':r['EVENT_TYPE'],
                        'state':r['STATE'], 'location':r['CZ_NAME'],
                        'date':r['BEGIN_DATE_TIME'], 'credited_source':r['SOURCE'],
                        'attribution':manifest['attribution'], 'source_url':manifest['source_url'],
                        'source_sha256':manifest['sha256'],
                    }

    def get(self, event_id):
        if not isinstance(event_id, str) or event_id not in self.records:
            raise Problem('That report is not in the development archive.', 404)
        return self.records[event_id]

    def search(self, query=''):
        query = query.casefold().strip()
        return [{'event_id':r['event_id'], 'title':r['text'][:110], 'location':r['location'],
                 'state':r['state'], 'date':r['date']} for r in self.records.values()
                if not query or query in (r['event_id']+' '+r['text']+' '+r['state']+' '+r['location']).casefold()][:40]


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS classifications (
                    id TEXT PRIMARY KEY, text_hash TEXT NOT NULL, payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS reports (
                    id TEXT PRIMARY KEY, version INTEGER NOT NULL,
                    updated_at TEXT NOT NULL, payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS review_history (
                    report_id TEXT NOT NULL, version INTEGER NOT NULL,
                    payload TEXT NOT NULL, PRIMARY KEY(report_id,version)
                );
            ''')

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def classification(self, text, result):
        identifier = str(uuid.uuid4())
        record = {'id':identifier, 'created_at':now(), **result}
        with self.connect() as db:
            db.execute('INSERT INTO classifications VALUES (?,?,?)',
                       (identifier, sha(text.encode()), json.dumps(record)))
        return record

    def get_classification(self, identifier, text):
        if identifier is None:
            return None
        if not isinstance(identifier,str):
            raise Problem('Invalid suggestion identifier.')
        with self.connect() as db:
            row = db.execute('SELECT * FROM classifications WHERE id=?',(identifier,)).fetchone()
        if not row or row['text_hash'] != sha(text.encode()):
            raise Problem('The suggestion belongs to a different report. Classify this text again.')
        return json.loads(row['payload'])

    def get(self, identifier):
        with self.connect() as db:
            row = db.execute('SELECT payload FROM reports WHERE id=?',(identifier,)).fetchone()
        if not row:
            raise Problem('Review not found.', 404)
        return json.loads(row['payload'])

    def list(self, state=None):
        if state is not None and state not in STATES:
            raise Problem('Unknown review state.')
        with self.connect() as db:
            records = [json.loads(r['payload']) for r in db.execute('SELECT payload FROM reports ORDER BY updated_at DESC,id')]
        return [r for r in records if state is None or r['review_state']==state]

    def decision(self, body, suggestion):
        state = body.get('review_state','pending')
        label = body.get('selected_label') or None
        note = body.get('note','')
        reason = body.get('unresolved_reason') or None
        if state not in STATES or label is not None and label not in LABELS:
            raise Problem('Choose a valid review state and category.')
        if not isinstance(note,str) or len(note)>4000:
            raise Problem('Review notes can contain at most 4,000 characters.')
        prediction = suggestion['prediction'] if suggestion else None
        if state == 'confirmed' and (prediction is None or label != prediction):
            raise Problem('Confirm requires the exact model suggestion. Use correction for a different category.')
        if state == 'corrected' and (label is None or label == prediction):
            raise Problem('Choose a different category for a correction.')
        if state == 'corrected' and suggestion is None and not note.strip():
            raise Problem('Add a note for a manual classification without a model suggestion.')
        if state in ('pending','unresolved') and label is not None:
            raise Problem('Pending and unresolved reviews must not have a final category.')
        if state == 'unresolved' and reason not in REASONS:
            raise Problem('Choose why this report remains unresolved.')
        if state == 'unresolved' and reason == 'other' and not note.strip():
            raise Problem('Add a note explaining the unresolved report.')
        return {'review_state':state, 'selected_label':label, 'note':note,
                'unresolved_reason':reason if state=='unresolved' else None}

    def create(self, body, archive):
        text = valid_text(body.get('text'))
        event_id = body.get('archive_event_id')
        source = archive.get(event_id) if event_id is not None else None
        if source and source['text'] != text:
            raise Problem('An archived report must retain its original text. Start a pasted report to edit it.')
        suggestion = self.get_classification(body.get('classification_id'),text)
        record = {'id':str(uuid.uuid4()), 'version':1, 'created_at':now(), 'updated_at':now(),
                  'text':text, 'source':source, 'suggestion':suggestion,
                  **self.decision(body,suggestion)}
        data = json.dumps(record,ensure_ascii=False)
        with self.connect() as db:
            db.execute('INSERT INTO reports VALUES (?,?,?,?)',(record['id'],1,record['updated_at'],data))
            db.execute('INSERT INTO review_history VALUES (?,?,?)',(record['id'],1,data))
        return record

    def update(self, identifier, body):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT payload FROM reports WHERE id=?',(identifier,)).fetchone()
            if not row:
                raise Problem('Review not found.',404)
            current = json.loads(row['payload'])
            if body.get('version') != current['version']:
                raise Problem('This review changed in another window. Reopen it before saving.',409)
            if any(k in body for k in ('text','source','archive_event_id','suggestion')):
                raise Problem('Original report and suggestion are immutable after saving.')
            suggestion=current['suggestion']
            if 'classification_id' in body:
                if suggestion is not None:
                    raise Problem('The saved model suggestion is immutable.')
                suggestion=self.get_classification(body['classification_id'],current['text'])
            updated = {**current, 'suggestion':suggestion, **self.decision(body,suggestion),
                       'version':current['version']+1,'updated_at':now()}
            data = json.dumps(updated,ensure_ascii=False)
            db.execute('UPDATE reports SET version=?,updated_at=?,payload=? WHERE id=?',
                       (updated['version'],updated['updated_at'],data,identifier))
            db.execute('INSERT INTO review_history VALUES (?,?,?)',(identifier,updated['version'],data))
        return updated

    def export(self, fmt, state=None):
        records = self.list(state)
        if fmt == 'json':
            return json.dumps({'schema_version':1,'exported_at':now(),'reviews':records},ensure_ascii=False,indent=2).encode(), 'application/json'
        if fmt != 'csv':
            raise Problem('Choose CSV or JSON for export.')
        columns = ['id','version','created_at','updated_at','original_text','event_id','episode_id',
                   'source_attribution','source_url','source_sha256','source_state','source_location','source_date',
                   'credited_source','reference_label','suggested_label','raw_completion','review_state',
                   'selected_label','unresolved_reason','note','model_run','model_fingerprint','prompt_sha256']
        output = io.StringIO(newline='')
        writer = csv.DictWriter(output,fieldnames=columns)
        writer.writeheader()
        for r in records:
            source=r['source'] or {};suggestion=r['suggestion'] or {};model=suggestion.get('model',{})
            writer.writerow({**{k:r[k] for k in ('id','version','created_at','updated_at','review_state','selected_label','unresolved_reason','note')},
                'original_text':r['text'],'event_id':source.get('event_id'),'episode_id':source.get('episode_id'),
                'source_attribution':source.get('attribution'),'source_url':source.get('source_url'),
                'source_sha256':source.get('source_sha256'),'source_state':source.get('state'),
                'source_location':source.get('location'),'source_date':source.get('date'),
                'credited_source':source.get('credited_source'),'reference_label':source.get('reference_label'),
                'suggested_label':suggestion.get('prediction'),'raw_completion':suggestion.get('raw_completion'),
                'model_run':model.get('run_id'),'model_fingerprint':model.get('artifact_fingerprint'),
                'prompt_sha256':model.get('prompt_sha256')})
        return output.getvalue().encode(), 'text/csv; charset=utf-8'
