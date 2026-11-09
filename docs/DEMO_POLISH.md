# Bounded demo polish

Branch: `feature/demo-polish-v1`, based on `working-review-app-v1`
(`0297d9d`). Exactly three improvements are included.

## 1. Live review progress

The overview counts pending, confirmed, corrected and unresolved **saved** reports.
Click a state card to apply the same queue filter used by CSV/JSON export; click
All saved to clear it. Unsaved drafts do not change counts. The progress meter
counts only confirmed/corrected categories. Unresolved remains a separate state
with no final category, so it is never presented as a completed classification.

## 2. Next pending report

The button opens the oldest-created saved pending report other than the current
one. It refreshes the queue before selecting, switches the visible filter to
Pending, and fetches the saved report through the existing API. It never saves,
confirms or classifies automatically. After an explicit Save, click Next pending
to continue. It is disabled if no other pending report exists. The existing
unsaved-change confirmation remains in the navigation path.

## 3. Recorded model evidence

Model evidence opens a modal with the locked v1 development results: base raw,
keyword, merged offline and actual served accuracy/macro-F1, a baseline comparison
chart, the real recorded training/development-monitor loss curve, and recorded
merged/served smoke-test and checkpoint-test results.

`scripts/build_demo_evidence.py` deterministically embeds the view in the existing
HTML from saved JSON/JSONL/test records. `--check` detects stale generated content.
Source hashes are available in the provenance disclosure. The panel labels its
2,449-report development scope, the 240-report balanced loss monitor, the weaker
letter protocol, the single offline/served prediction difference, the partial
training pass and dataset limits. It does not present results as per-report
confidence or claim reserved-final evaluation. The model fingerprint is checked
against the live workspace health metadata before showing an identity match.

## Scope and checkpoint protection

All runtime changes are confined to the three existing static UI files. No new
backend endpoint, schema migration, model, prompt, decoding, training run or
benchmark was introduced. Original v1 evidence is preserved; its old frontend
hash audit still describes the tagged baseline rather than this feature branch.
New polish verification is kept separately under `docs/demo-polish/`.

The original tag and main branch remain at `0297d9d`. Model/adapter directories,
all dataset/split files, reserved final, original evaluation evidence and Curator's
Studio remain unchanged. Tests use only `runtime/demo-polish-qa.sqlite3` on a
separate loopback port, not the user's normal review database.

## Checks

```sh
.venv/bin/python scripts/build_demo_evidence.py --check
node --test tests/test_demo_polish.cjs
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/verify_demo_polish.py
```

Five new JavaScript tests cover correct counts, unresolved handling, empty state,
oldest-pending order/current-item exclusion and non-mutation. The twelve existing
Python tests also pass. The five predeclared smoke cases passed on the actual
unchanged merged model through the separate QA server. No full development
inference is repeated because serving/inference code and weights are unchanged.

Browser QA verified empty and populated counts, oldest-pending selection,
state-card filters and their export links, explicit save→next navigation,
a pending report without a suggestion, and the no-pending disabled state.
Canceling the native unsaved-change dialog with Escape preserved the note and
current report. Saving unresolved increased only the unresolved count, leaving
the categorized progress unchanged. The evidence panel and its two real charts
were visually inspected, including a narrow-screen layout check. The temporary
viewport override was reset. All fixture writes used the separate QA database.

Screenshots and the verification record are in `docs/demo-polish/`.

The old submission candidate remains available at `working-review-app-v1`.
This feature branch is not merged into main or published. Final submission
packaging and Loom remain separate work; no additional features are included.
