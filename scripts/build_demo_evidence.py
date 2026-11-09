"""Embed a deterministic, provenance-labelled v1 evidence view; no inference."""
import argparse
import hashlib
import html
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BEGIN='<!-- BEGIN MODEL EVIDENCE SNAPSHOT -->'
END='<!-- END MODEL EVIDENCE SNAPSHOT -->'


def snapshot():
    sources={
        'base':'data/evaluations/untuned-development-v1/metrics.json',
        'keyword':'data/evaluations/keyword-development-v1/metrics.json',
        'merged':'data/evaluations/merged-lora-v1-raw/metrics.json',
        'served':'data/evaluations/served-lora-v1-raw/metrics.json',
        'base_letter':'data/evaluations/letter-development-v1/metrics.json',
        'merged_letter':'data/evaluations/merged-lora-v1-letter/metrics.json',
        'loss':'data/training/lora-v1/loss-review.json',
        'losses':'data/training/lora-v1/losses.jsonl',
        'training':'data/training/lora-v1/training-result.json',
        'merged_smoke':'data/training/lora-v1/merged-smoke.json',
        'served_smoke':'data/evaluations/served-lora-v1-raw/smoke.json',
        'tests':'data/evaluations/review-app-tests.txt',
    }
    raw={key:(ROOT/path).read_bytes() for key,path in sources.items()}
    records={key:json.loads(value) for key,value in raw.items() if key not in ('losses','tests')}
    assert all(records[key]['n']==2449 for key in ('base','keyword','merged','served'))
    assert records['merged_smoke']['passed']==records['served_smoke']['passed']==5
    assert 'Ran 12 tests' in raw['tests'].decode() and raw['tests'].decode().strip().endswith('OK')
    logs=[json.loads(line) for line in raw['losses'].splitlines()]
    training=[(p['iteration'],p['train_loss']) for p in logs if p['kind']=='train']
    monitor=[(p['iteration'],p['val_loss']) for p in records['loss']['monitor_points']]
    monitor.append((600,records['loss']['final600_monitor_loss']))
    return {'records':records,'training':training,'monitor':monitor,'sources':{sources[key]:hashlib.sha256(value).hexdigest() for key,value in raw.items()}}


def loss_chart(training,monitor):
    def x(step):return 48+step/600*428
    def y(value):return 184-value/.25*162
    lines=['<svg class="loss-svg" viewBox="0 0 500 224" role="img" aria-labelledby="loss-title loss-description"><title id="loss-title">Recorded LoRA loss curve</title><desc id="loss-description">Training uses 20-update means. The fixed development monitor contains 240 reports, 40 per class. Loss decreases overall with fluctuations. The final checkpoint was fixed at 600 updates.</desc>']
    for value in (0,.05,.1,.15,.2,.25):
        lines.append(f'<line class="chart-grid" x1="48" x2="476" y1="{y(value):.2f}" y2="{y(value):.2f}"/><text class="axis-label" x="39" y="{y(value)+4:.2f}" text-anchor="end">{value:.2f}</text>')
    for step in (0,200,400,600):lines.append(f'<text class="axis-label" x="{x(step):.2f}" y="202" text-anchor="middle">{step}</text>')
    for points,css in ((training,'training-line'),(monitor,'monitor-line')):
        path=' '.join(('M' if i==0 else 'L')+f'{x(step):.2f},{y(value):.2f}' for i,(step,value) in enumerate(points))
        lines.append(f'<path class="{css}" d="{path}"/>')
    for step,value in monitor:lines.append(f'<circle class="monitor-dot" cx="{x(step):.2f}" cy="{y(value):.2f}" r="3"/>')
    lines.append('<text class="axis-label" x="260" y="220" text-anchor="middle">Completed optimizer updates</text></svg>')
    return ''.join(lines)


def render():
    data=snapshot();r=data['records']
    methods=[('base','Original base model','base-bar'),('keyword','Keyword baseline','keyword-bar'),('merged','Merged LoRA · offline','merged-bar'),('served','Merged LoRA · app','served-bar')]
    cards=[];bars=['<svg class="accuracy-svg" viewBox="0 0 640 226" role="img" aria-labelledby="accuracy-title accuracy-description"><title id="accuracy-title">Accuracy on the same 2,449 development reports</title><desc id="accuracy-description">Original base 71.74 percent, keyword baseline 89.02 percent, merged offline 98.73 percent, served model 98.69 percent. Offline uses batches of eight. The app uses one report per request.</desc>']
    for i,(key,title,css) in enumerate(methods):
        report=r[key];accuracy=report['accuracy'];y=28+i*47
        cards.append(f'<article class="metric-card {css}"><span>{title}</span><strong>{accuracy:.2%}</strong><small>accuracy · macro-F1 {report["macro_f1"]:.2%}</small></article>')
        bars.append(f'<text class="bar-label" x="0" y="{y+15}">{title}</text><rect class="bar-track" x="193" y="{y}" width="355" height="23" rx="3"/><rect class="{css}" x="193" y="{y}" width="{355*accuracy:.3f}" height="23" rx="3"/><text class="bar-value" x="632" y="{y+16}" text-anchor="end">{accuracy:.2%}</text>')
    for tick in (0,50,100):bars.append(f'<text class="axis-label" x="{193+355*tick/100:.1f}" y="222" text-anchor="middle">{tick}%</text>')
    bars.append('</svg>')
    sources=''.join(f'<li><code>{html.escape(path)}</code><span>{digest}</span></li>' for path,digest in data['sources'].items())
    loss=r['loss'];served=r['served'];train=r['training']
    return f'''{BEGIN}
<dialog id="evidence-dialog" aria-labelledby="evidence-heading">
  <div class="dialog-head"><div><div class="eyebrow">RECORDED DEVELOPMENT RESULTS · V1</div><h2 id="evidence-heading">Evidence behind the suggestion</h2></div><button id="evidence-close" class="icon-button" aria-label="Close model evidence">×</button></div>
  <p class="evidence-intro">A real local LoRA run, compared on the same <strong>2,449 historical development reports</strong>. These results describe the saved v1 model, not the certainty of an individual suggestion.</p>
  <p id="evidence-identity" class="identity-status" data-artifact="{served['model']['artifact_fingerprint']}">Recorded artifact: merged-lora-v1 · checking workspace model…</p>
  <div class="metric-cards">{''.join(cards)}</div>
  <div class="evidence-charts"><section class="chart-card"><div class="eyebrow">SAME DATA · MODEL PROMPTS MATCHED</div><h3>Measured classification accuracy</h3>{''.join(bars)}<p class="chart-note">{100*(r['merged']['accuracy']-r['base']['accuracy']):+.2f} percentage points over matched base raw; {100*(r['merged']['accuracy']-r['keyword']['accuracy']):+.2f} over keyword. One prediction changes between offline batch 8 and single-report serving.</p></section>
  <section class="chart-card"><div class="eyebrow">ONE BOUNDED RUN · NO CHECKPOINT SEARCH</div><h3>Training and development-monitor loss</h3>{loss_chart(data['training'],data['monitor'])}<div class="chart-legend"><span class="training-key">Training · 20-update mean</span><span class="monitor-key">Fixed development monitor</span></div><p class="chart-note">Monitor loss {loss['initial_monitor_loss']:.4f} → {loss['final600_monitor_loss']:.4f}. It fluctuated; the lowest logged value was {loss['best_logged_monitor_loss']:.4f}. Monitor: 240 reports, 40 per class; label/EOS loss only.</p></section></div>
  <div class="proof-grid"><div><strong>5 / 5</strong><span>Merged smoke tests</span></div><div><strong>5 / 5</strong><span>Served smoke tests</span></div><div><strong>12 passed</strong><span>Checkpoint tests</span></div><div><strong>{served['invalid_count']}</strong><span>Invalid served outputs</span></div></div>
  <p class="run-note">Run lora-v1 · {train['completed_steps']} updates · {train['unique_training_rows']:,} distinct training reports (21% of a pass) · rank 8 · batch 4 · frozen base weights · native BF16 merge</p>
  <details class="evidence-limits"><summary>Protocol, limitations &amp; source records</summary><p>Offline base and merged raw completion use identical prompts, greedy decoding, a 16-token budget and exact-label parsing. App results were measured through HTTP, one report at a time. No quantization was introduced.</p><p>Handout letter-choice is a separate protocol: base {r['base_letter']['accuracy']:.2%}, merged {r['merged_letter']['accuracy']:.2%}. It is weaker and had 48 adapter/merge label changes. It is not the app inference path.</p><p>Development-only, reference-label, closed-set results. Source labels can be ambiguous; Wildfire has only 41 development reports. These are not calibrated confidence scores or universal input coverage. Reserved final was not evaluated. Full per-class reports and confusion matrices remain in the repository.</p><ul class="evidence-sources">{sources}</ul></details>
</dialog>
{END}'''


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    path=ROOT/'review_app/static/index.html';text=path.read_text()
    start=text.index(BEGIN);end=text.index(END,start)+len(END)
    expected=render()
    if args.check:
        if text[start:end]!=expected:raise SystemExit('Evidence snapshot differs from saved records')
        print('Evidence snapshot matches saved records');return
    path.write_text(text[:start]+expected+text[end:]);print('Embedded recorded evidence snapshot; no model calls')


if __name__=='__main__':main()
