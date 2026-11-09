# Working app checkpoint

`working-review-app-v1` preserves the completed local review application before
prospective functionality changes. It is a submission candidate, not a claim that
GitHub publication, final licensing/reproducibility packaging or Loom are finished.

The checkpoint includes code, source/split manifests, pinned package versions,
training/loss/merge/smoke evidence, per-class evaluations, served-model evaluation,
UI screenshot, tests and instructions. App and LoRA evidence hash audits passed
before committing. Served development accuracy is 98.69%, macro-F1 98.76%.
Reserved final has not been evaluated; Curator's Studio remains separate.

## Local artifacts

Git intentionally excludes raw source data, model weights, environments, caches
and runtime databases. Preserve these existing local model directories:

- `runtime/models/qwen3-1.7b-base/`
- `runtime/lora-v1/`
- `runtime/merged-lora-v1/`

Any future model experiments must use new versioned paths, not overwrite v1.
The tracked merged manifest records the exact original artifact file hashes.
A consistent snapshot of the normal review database is stored locally at
`runtime/checkpoints/working-review-app-v1/review-app.sqlite3`, with a local
manifest beside it. It is ignored by Git and is not part of a published package.
Subsequent user reviews continue to use `runtime/review-app.sqlite3`.

## Return to this version

To inspect the checkpoint without discarding later commits, start a new branch
from the tag, with current changes committed or safely stashed first:

```sh
git switch -c restore/working-review-app-v1 working-review-app-v1
.venv/bin/python -m review_app
```

The Git checkout restores code and tracked evidence, not ignored runtime files.
Restore the database snapshot only if requested, with the app stopped and the
current database backed up first. Never use `git clean -fdx` to roll back: it
would remove ignored data and model artifacts needed by this application.

No remote push or publication is performed by creating this local checkpoint.
