'use strict';
// Pure queue helpers are also exercised by the Node regression tests.
function queueSummary(records){
  const counts={pending:0,confirmed:0,corrected:0,unresolved:0};
  for(const record of records)if(Object.prototype.hasOwnProperty.call(counts,record.review_state))counts[record.review_state]++;
  return {counts,total:records.length,categorized:counts.confirmed+counts.corrected};
}
function chooseNextPending(records,currentId=null){
  return records.filter(r=>r.review_state==='pending'&&r.id!==currentId)
    .sort((a,b)=>a.created_at.localeCompare(b.created_at)||a.id.localeCompare(b.id))[0]||null;
}

// Frozen v1 keyword rules: ties follow declared label order, no-hit fallback is Wind.
const keywordRules={"Flash Flood": "\\b(?:flash flood\\w*|flood\\w*|inundat\\w*)\\b", "Tornado": "\\b(?:tornado\\w*|twister\\w*|EF[0-5])\\b", "Thunderstorm Wind": "\\b(?:wind\\w*|gust\\w*|tree\\w*|power ?line\\w*)\\b", "Hail": "\\bhail\\w*\\b", "Winter Storm": "\\b(?:snow\\w*|blizzard\\w*|sleet|freezing rain|ice storm)\\b", "Wildfire": "\\b(?:wildfire\\w*|fire\\w*|burn\\w*|contain\\w*)\\b"};
const keywordLabels=["Flash Flood", "Tornado", "Thunderstorm Wind", "Hail", "Winter Storm", "Wildfire"];
function textCues(text){
  const hits=[];
  for(const label of keywordLabels){const pattern=keywordRules[label].replace(/^\\b/,'(?<![\\p{L}\\p{N}_])').replace(/\\b$/,'(?![\\p{L}\\p{N}_])').replace(/\\w/g,'[\\p{L}\\p{N}_]');const re=new RegExp(pattern,'giu');for(const match of text.replace(/[İı]/g,'i').matchAll(re))hits.push({label,start:match.index,end:match.index+match[0].length,text:text.slice(match.index,match.index+match[0].length)});}
  return hits.sort((a,b)=>a.start-b.start||b.end-a.end);
}
function keywordResult(text){
  const hits=textCues(text),scores=Object.fromEntries(keywordLabels.map(label=>[label,0]));
  for(const hit of hits)scores[hit.label]++;
  const prediction=hits.length?keywordLabels.reduce((best,label)=>scores[label]>scores[best]?label:best,keywordLabels[0]):'Thunderstorm Wind';
  return {prediction,hits,scores,fallback:!hits.length};
}
function methodsDisagree(record){return keywordLabels.includes(record.suggestion?.prediction)&&record.suggestion.prediction!==keywordResult(record.text).prediction;}
function reviewOutcomes(records){
  const counts=queueSummary(records).counts,changes={};
  for(const r of records){const from=r.suggestion?.prediction;if(r.review_state==='corrected'&&keywordLabels.includes(from)&&r.selected_label&&r.selected_label!==from){const key=from+' → '+r.selected_label;changes[key]=(changes[key]||0)+1;}}
  return {reports:records.length,accepted:counts.confirmed,corrected:counts.corrected,unresolved:counts.unresolved,pending:counts.pending,category_changes:Object.entries(changes).sort((a,b)=>b[1]-a[1]||a[0].localeCompare(b[0]))};
}

if(typeof module!=='undefined'&&module.exports)module.exports={queueSummary,chooseNextPending,textCues,keywordResult,methodsDisagree,reviewOutcomes};
if(typeof document!=='undefined'){
const $=id=>document.getElementById(id);
const state={record:null,source:null,suggestion:null,dirty:false,busy:false,labels:[],queue:[]};
const sessionStarted=new Date().toISOString(),sessionIds=new Set();
const names={pending:'Pending',confirmed:'Confirmed',corrected:'Corrected',unresolved:'Unresolved'};
async function api(path,options={}){const response=await fetch(path,{...options,headers:{'Content-Type':'application/json','X-Review-App':'1',...(options.headers||{})}});const result=await response.json();if(!response.ok)throw new Error(result.error||'Request failed.');return result;}
function message(text,error=false){$('message').textContent=text;$('message').classList.toggle('error',error);}
function decision(){return document.querySelector('input[name=decision]:checked').value;}
function markDirty(){state.dirty=true;$('record-status').textContent='Unsaved changes';$('record-status').className='badge pending';}
function safeLeave(){return !state.dirty||window.confirm('Discard unsaved review changes?');}
function busy(on){state.busy=on;for(const id of ['classify','save-review','new-report','archive-open'])$(id).disabled=on;for(const el of document.querySelectorAll('#queue-list button'))el.disabled=on;$('classify').disabled=on||!!state.record?.suggestion;for(const el of document.querySelectorAll('input[name=decision]'))el.disabled=on||(el.id==='confirm-choice'&&!state.suggestion?.prediction);for(const id of ['selected-label','unresolved-reason','review-note'])$(id).disabled=on;$('report-text').readOnly=on||!!state.record||!!state.source;renderProgress();}
function renderDecision(){const value=decision();$('category-field').hidden=value!=='corrected';$('reason-field').hidden=value!=='unresolved';$('confirm-choice').disabled=!state.suggestion?.prediction;}
function renderSuggestion(){const s=state.suggestion;$('suggested-label').textContent=s?(s.prediction||'Manual review needed'):'Not classified yet';$('suggestion-help').textContent=s?(s.prediction?(state.record?'Saved reviewer decision: '+names[state.record.review_state]+'. The model suggestion remains unchanged.':'A suggestion, awaiting your review. It does not confirm the source facts.'):'The model did not return a supported category. Correct it manually or leave unresolved.'):'The original report stays beside the suggestion. You make the final decision.';$('model-record').hidden=!s;$('model-details').textContent=s?JSON.stringify({run:s.model.run_id,artifact:s.model.artifact_fingerprint,created_at:s.created_at,raw_completion:s.raw_completion},null,2):'';renderDecision();renderComparison();}
function renderSource(){const card=$('source-card');card.replaceChildren();const strong=document.createElement('strong');strong.textContent=state.source?'NOAA archived report · Event '+state.source.event_id:'No source linked';card.append(strong);const p=document.createElement('p');p.textContent=state.source?[state.source.location,state.source.state,state.source.date].filter(Boolean).join(' · '):'Pasted reports retain their own text. They do not inherit facts from the archive.';card.append(p);if(state.source){const details=document.createElement('details');const summary=document.createElement('summary');summary.textContent='Source details & reference label';details.append(summary);const text=document.createElement('p');text.textContent='NOAA reference: '+state.source.reference_label+' · Episode '+state.source.episode_id+' · Credited source: '+state.source.credited_source;details.append(text);const attribution=document.createElement('p');attribution.textContent=state.source.attribution;details.append(attribution);const a=document.createElement('a');a.textContent='Pinned source snapshot ↗';a.href=state.source.source_url;a.target='_blank';a.rel='noopener';details.append(a);card.append(details);}$('input-kind').textContent=state.source?'Archived original':'Pasted text';}
function loadDraft(record=null,source=null){state.record=record;state.source=record?record.source:source;state.suggestion=record?record.suggestion:null;state.dirty=false;$('report-text').value=record?record.text:(source?.text||'');$('review-note').value=record?.note||'';$('selected-label').value=record?.selected_label||'';$('unresolved-reason').value=record?.unresolved_reason||'';document.querySelector(`input[name=decision][value="${record?.review_state||'pending'}"]`).checked=true;$('workspace-title').textContent=state.source?'Review event '+state.source.event_id:(record?'Saved report':'A report, ready for review.');$('record-status').textContent=record?names[record.review_state]:'New draft';$('record-status').className='badge '+(record?.review_state||'neutral');$('save-detail').textContent=record?'Saved '+new Date(record.updated_at).toLocaleString()+' · version '+record.version:'Saved reviews can be reopened from the queue.';$('text-hint').textContent=record?'Saved original text is preserved. Review decisions and notes can be updated.':source?'Archived text is preserved exactly as recorded.':'Paste an event narrative, or browse an archived NOAA report.';renderSource();renderSuggestion();updateCount();busy(false);message('');renderQueue();}
function updateCount(){$('text-count').textContent=$('report-text').value.length.toLocaleString()+' characters';}
function renderQueue(){const filter=$('filter').value;const list=$('queue-list');list.replaceChildren();const rows=visibleQueue();$('queue-count').textContent=rows.length;for(const r of rows){const button=document.createElement('button');button.className='queue-item'+(state.record?.id===r.id?' active':'');button.disabled=state.busy;const title=document.createElement('strong');title.textContent=r.text.slice(0,85)+(r.text.length>85?'…':'');const small=document.createElement('small');small.textContent=(r.source?'NOAA #'+r.source.event_id:'Pasted report')+' · '+new Date(r.updated_at).toLocaleDateString();const badge=document.createElement('span');badge.className='badge '+r.review_state;badge.textContent=names[r.review_state]+(r.selected_label?' · '+r.selected_label:'');button.append(title,small,badge);button.onclick=async()=>{if(!safeLeave())return;try{loadDraft(await api('/api/reports/'+r.id));}catch(e){message(e.message,true);}};list.append(button);}if(!rows.length){const p=document.createElement('p');p.className='empty';p.textContent=filter?'No saved reviews in this state.':'Your saved reviews will appear here. Start with a pasted report or the NOAA archive.';list.append(p);}for(const format of ['csv','json'])$('export-'+format).href='/api/export?format='+format+(filter?'&state='+filter:'');$('export-csv').hidden=$('disagreements-only').checked;renderProgress();renderSession();}
function renderProgress(){
  const summary=queueSummary(state.queue),filter=$('filter').value;
  for(const [name,count] of Object.entries(summary.counts))$('count-'+name).textContent=count;
  $('count-total').textContent=summary.total;
  $('category-progress').max=summary.total||1;
  $('category-progress').value=summary.categorized;
  $('progress-caption').textContent=summary.total?summary.categorized+' of '+summary.total+' have a confirmed or corrected category':'No saved reports yet';
  for(const button of document.querySelectorAll('[data-review-filter]')){
    button.setAttribute('aria-pressed',String(button.dataset.reviewFilter===filter));button.disabled=state.busy;
  }
  $('show-all-reviews').setAttribute('aria-pressed',String(filter===''));
  $('show-all-reviews').disabled=state.busy;
  const next=chooseNextPending(state.queue,state.record?.id);
  $('next-pending').disabled=state.busy||!next;
  $('next-pending-hint').textContent=next?'Oldest pending first · opens without saving':summary.counts.pending?'You are viewing the only pending report.':'No pending reports.';
}
for(const button of document.querySelectorAll('[data-review-filter]'))button.onclick=()=>{$('filter').value=button.dataset.reviewFilter;renderQueue();};
$('show-all-reviews').onclick=()=>{$('filter').value='';renderQueue();};
$('next-pending').onclick=async()=>{
  if(state.busy||!safeLeave())return;
  busy(true);
  try{
    await refreshQueue();
    const target=chooseNextPending(state.queue,state.record?.id);
    if(!target){message('No other pending reports.');return;}
    const record=await api('/api/reports/'+target.id);
    $('filter').value='pending';loadDraft(record);
    message('Next pending report loaded. Review and save when ready.');
  }catch(e){message(e.message,true);}finally{busy(false);}
};

function visibleQueue(){const filter=$('filter').value;return state.queue.filter(r=>(!filter||r.review_state===filter)&&(!$('disagreements-only').checked||methodsDisagree(r)));}
function downloadJSON(filename,payload){const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=filename;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$('disagreements-only').onchange=renderQueue;
$('export-json').addEventListener('click',event=>{if(!$('disagreements-only').checked)return;event.preventDefault();downloadJSON('disagreement-reviews.json',{schema_version:1,exported_at:new Date().toISOString(),filter:{state:$('filter').value||null,methods_disagree:true},reviews:visibleQueue()});});
function renderComparison(){
  const text=$('report-text').value,result=keywordResult(text),prediction=state.suggestion?.prediction;
  $('keyword-label').textContent=text.trim()?result.prediction:'Awaiting report';
  $('comparison-status').textContent=!text.trim()?'Paste or open a report to compare methods.':!keywordLabels.includes(prediction)?'Awaiting a supported model suggestion.':prediction===result.prediction?'Methods agree — agreement is not certainty.':'Methods disagree — a useful starting point for human review.';
  $('keyword-explanation').textContent=result.fallback?'No keyword hits. The baseline defaults to Thunderstorm Wind; this is not evidence of wind.':'Counts matching words; ties use the frozen category order. Cues can include negated or historical mentions.';
  const box=$('cue-text');box.replaceChildren();let offset=0;
  for(const hit of result.hits){if(hit.start<offset)continue;box.append(document.createTextNode(text.slice(offset,hit.start)));const mark=document.createElement('mark');mark.textContent=text.slice(hit.start,hit.end);mark.title=hit.label+' keyword cue';box.append(mark);offset=hit.end;}
  box.append(document.createTextNode(text.slice(offset)));if(!text)box.textContent='Text cues will appear here.';
}
function sessionRecords(){return state.queue.filter(r=>sessionIds.has(r.id));}
function renderSession(){const result=reviewOutcomes(sessionRecords());$('session-counts').textContent=result.reports+' reports saved this session · '+result.accepted+' accepted · '+result.corrected+' corrected · '+result.unresolved+' unresolved · '+result.pending+' pending';const list=$('session-changes');list.replaceChildren();for(const [change,count] of result.category_changes){const item=document.createElement('li');item.textContent=change+' · '+count;list.append(item);}if(!result.category_changes.length){const item=document.createElement('li');item.textContent='No category changes from a supported model suggestion yet.';list.append(item);}$('session-export').disabled=!result.reports;}
$('session-export').onclick=()=>{const reviews=sessionRecords();downloadJSON('review-session.json',{schema_version:1,session_started_at:sessionStarted,exported_at:new Date().toISOString(),scope:'Latest saved decision per report saved in this browser tab since page load. Reviewer outcomes are not measured model accuracy.',summary:reviewOutcomes(reviews),reviews});};

$('evidence-open').onclick=()=>$('evidence-dialog').showModal();
$('evidence-close').onclick=()=>$('evidence-dialog').close();
async function refreshQueue(){state.queue=await api('/api/reports');renderQueue();}
$('new-report').onclick=()=>{if(safeLeave())loadDraft();};
$('report-text').oninput=()=>{state.suggestion=null;document.querySelector('input[value=pending][name=decision]').checked=true;renderSuggestion();updateCount();markDirty();};
for(const el of document.querySelectorAll('input[name=decision]'))el.onchange=()=>{renderDecision();markDirty();};
for(const id of ['selected-label','unresolved-reason','review-note'])$(id).oninput=markDirty;
$('classify').onclick=async()=>{message('');busy(true);$('classify').textContent='Reading report locally…';try{state.suggestion=await api('/api/classify',{method:'POST',body:JSON.stringify({text:$('report-text').value})});document.querySelector('input[value=pending][name=decision]').checked=true;renderSuggestion();markDirty();message(state.suggestion.prediction?'Suggestion ready. Confirm, correct, or leave unresolved.':'Manual review needed: the model returned an unsupported category.');}catch(e){message(e.message,true);}finally{$('classify').innerHTML='Suggest a category <span aria-hidden="true">→</span>';busy(false);}};
$('save-review').onclick=async()=>{message('');busy(true);try{const value=decision();const body={review_state:value,selected_label:value==='confirmed'?state.suggestion?.prediction:(value==='corrected'?$('selected-label').value:null),unresolved_reason:value==='unresolved'?$('unresolved-reason').value:null,note:$('review-note').value};let result;if(state.record){body.version=state.record.version;if(!state.record.suggestion&&state.suggestion)body.classification_id=state.suggestion.id;result=await api('/api/reports/'+state.record.id,{method:'PUT',body:JSON.stringify(body)});}else{Object.assign(body,{text:$('report-text').value,archive_event_id:state.source?.event_id??null,classification_id:state.suggestion?.id??null});result=await api('/api/reports',{method:'POST',body:JSON.stringify(body)});}sessionIds.add(result.id);loadDraft(result);await refreshQueue();message('Review saved. Original text, suggestion and your decision are preserved.');}catch(e){message(e.message,true);}finally{busy(false);}};
$('filter').onchange=renderQueue;
$('archive-open').onclick=async()=>{if(!safeLeave())return;$('archive-dialog').showModal();$('archive-search').value='';$('archive-search').focus();await searchArchive();};
$('archive-close').onclick=()=>$('archive-dialog').close();
let searchTimer,searchVersion=0;
async function searchArchive(){const version=++searchVersion;try{const rows=await api('/api/archive?q='+encodeURIComponent($('archive-search').value));if(version!==searchVersion)return;const list=$('archive-results');list.replaceChildren();for(const r of rows){const button=document.createElement('button');button.className='archive-option';const strong=document.createElement('strong');strong.textContent='NOAA #'+r.event_id+' · '+r.state+' · '+r.date;const span=document.createElement('span');span.textContent=r.title;button.append(strong,span);button.onclick=async()=>{try{const source=await api('/api/archive/'+r.event_id);$('archive-dialog').close();loadDraft(null,source);}catch(e){message(e.message,true);}};list.append(button);}if(!rows.length)list.textContent='No matching reports in the development archive.';}catch(e){message(e.message,true);}}
$('archive-search').oninput=()=>{clearTimeout(searchTimer);searchTimer=setTimeout(searchArchive,180);};
window.addEventListener('beforeunload',event=>{if(state.dirty){event.preventDefault();event.returnValue='';}});
(async()=>{try{const health=await api('/api/health');state.labels=health.labels;const matches=health.model.artifact_fingerprint===$('evidence-identity').dataset.artifact;$('evidence-identity').textContent=matches?'This workspace uses the evaluated merged-lora-v1 model.':'Recorded v1 results: this workspace model differs from the evaluated artifact.';$('evidence-identity').classList.toggle('model-mismatch',!matches);for(const label of health.labels){const option=document.createElement('option');option.value=label;option.textContent=label;$('selected-label').append(option);}await refreshQueue();loadDraft();}catch(e){message('Could not connect to the local app. '+e.message,true);busy(true);}})();

}
