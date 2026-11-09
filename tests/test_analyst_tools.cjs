const test=require('node:test');
const assert=require('node:assert/strict');
const {keywordResult,textCues,methodsDisagree,reviewOutcomes}=require('../review_app/static/app.js');
test('frozen keyword rules match Python baseline across case, ties, fallback and Unicode boundaries',()=>{
 const samples=['','An uneventful day','HAIL hailstones','snow then hail','Tornado with wind gusts','flash flooding inundated streets','fire contained','éfire','snow_ice','EF2 damage','powerlines down','<script>hail</script>'];
 const {execFileSync}=require('node:child_process');
 const expected=JSON.parse(execFileSync('.venv/bin/python',['-c','import sys,json;sys.path.insert(0,"scripts");from baseline_common import keyword;print(json.dumps([keyword(t) for t in json.loads(sys.argv[1])]))',JSON.stringify(samples)],{cwd:require('node:path').resolve(__dirname,'..'),encoding:'utf8'}));
 assert.deepEqual(samples.map(t=>keywordResult(t).prediction),expected);
});
test('cue spans preserve original case and exact text without HTML interpretation',()=>{
 const text='<img src=x onerror=alert(1)> HAIL and wind';for(const hit of textCues(text))assert.equal(text.slice(hit.start,hit.end),hit.text);
 assert.deepEqual(textCues(text).map(h=>h.text),['HAIL','wind']);
});
test('only two supported differing predictions count as disagreement',()=>{
 assert.equal(methodsDisagree({text:'hail',suggestion:{prediction:'Tornado'}}),true);
 for(const prediction of ['Hail',null,'INVALID'])assert.equal(methodsDisagree({text:'hail',suggestion:{prediction}}),false);
 assert.equal(methodsDisagree({text:'hail'}),false);
});
test('review outcomes distinguish unresolved and unknown origins from category changes',()=>{
 const result=reviewOutcomes([{review_state:'confirmed'},{review_state:'unresolved'},{review_state:'pending'},...['Hail','Hail','Tornado',null].map(prediction=>({review_state:'corrected',selected_label:'Tornado',suggestion:{prediction}}))]);
 assert.deepEqual(result,{reports:7,accepted:1,corrected:4,unresolved:1,pending:1,category_changes:[['Hail → Tornado',2]]});
});
