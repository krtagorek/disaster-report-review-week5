const test = require('node:test');
const assert = require('node:assert/strict');
const {queueSummary,chooseNextPending} = require('../review_app/static/app.js');
const rows = [
  {id:'recent',review_state:'pending',created_at:'2026-11-06T12:00:00Z'},
  {id:'done',review_state:'confirmed',created_at:'2026-11-06T08:00:00Z'},
  {id:'fixed',review_state:'corrected',created_at:'2026-11-06T09:00:00Z'},
  {id:'uncertain',review_state:'unresolved',created_at:'2026-11-06T10:00:00Z'},
  {id:'oldest',review_state:'pending',created_at:'2026-11-06T11:00:00Z'},
];
test('overview counts all saved states and never treats unresolved as categorized',()=>{
  assert.deepEqual(queueSummary(rows),{counts:{pending:2,confirmed:1,corrected:1,unresolved:1},total:5,categorized:2});
});
test('empty workspace has no implied progress or next task',()=>{
  assert.equal(queueSummary([]).categorized,0);assert.equal(queueSummary([]).total,0);
  assert.equal(chooseNextPending([]),null);
});
test('next pending uses oldest created report, ignoring queue display order',()=>{
  assert.equal(chooseNextPending(rows).id,'oldest');
  assert.equal(chooseNextPending([...rows].reverse()).id,'oldest');
});
test('next navigation skips current and completed/unresolved reports',()=>{
  assert.equal(chooseNextPending(rows,'oldest').id,'recent');
  assert.equal(chooseNextPending(rows.filter(r=>r.id!=='recent'),'oldest'),null);
});
test('summary and next navigation never modify original records or order',()=>{
  const before=JSON.stringify(rows);queueSummary(rows);chooseNextPending(rows);
  assert.equal(JSON.stringify(rows),before);
});
