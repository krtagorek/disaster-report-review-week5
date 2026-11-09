# Setup and reproduction

Verified environment: Apple silicon M5, 24 GB, macOS 26.6 and CPython 3.11.16.
Install Python 3.11 and `uv` before running these commands from the package root.

```sh
UV_CACHE_DIR="$PWD/work/uv-cache" uv venv --python 3.11 .venv
UV_CACHE_DIR="$PWD/work/uv-cache" uv pip install --python .venv/bin/python -r requirements-model.lock
mkdir -p data/raw work
curl --fail --location 'https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/StormEvents_details-ftp_v1.0_d2024_c20260728.csv.gz' --output data/raw/stormevents-2024.csv.gz
printf '%s  %s\n' '2070b83eccab041b36360ab73645b9a249c3eefc5b92b5b3fc0cbba4d9fcc09c' 'data/raw/stormevents-2024.csv.gz' | shasum -a 256 -c
```

Stop if the pinned snapshot is unavailable or its hash differs. The app uses
only the development IDs to populate its archive. The model receives narrative
text only.

## Run the recorded app

The archived code and evidence can be inspected immediately. Running the
classifier also requires the exact local merged directory at
`runtime/merged-lora-v1/`. If you already have that directory, the app verifies
its files against the recorded merge manifest on startup.

```sh
.venv/bin/python -m review_app --port 8877 --database runtime/review-app.sqlite3
```

Open http://127.0.0.1:8877. Use **Suggest a category**, then save a human
decision. Try **Methods disagree only**, **Text cues in the original report**
and **Review-session outcomes**. The session counts the latest decisions for
reports saved in the current tab since page load.

## Reproduce the training and merge separately

In a fresh disposable copy of this package, complete the environment and NOAA
setup above. Keep the evidence copy available for reference. In the disposable
copy, move the recorded run directory aside so the scripts can generate their
own outputs.

```sh
.venv/bin/python scripts/download_model.py
mv data/evaluations/local-model-verification.json work/recorded-local-model-verification.json
.venv/bin/python scripts/verify_local_model.py
mv data/training/lora-v1 work/recorded-lora-v1
mkdir -p data/training/lora-v1
cp work/recorded-lora-v1/plan.json data/training/lora-v1/
.venv/bin/python scripts/train_lora.py
.venv/bin/python scripts/merge_lora.py
.venv/bin/python -m review_app --port 8877 --database runtime/review-app.sqlite3
```

The reproduced merge generates a new manifest. Its weights and metrics can vary
across environments. Treat the packaged evaluation as evidence for the original
artifact, identified by its fingerprint. The app reports whether that
fingerprint matches. Run metadata and metrics are saved in the package.
See [the run report](LORA_RUN.md) and [served evaluation](REVIEW_APP.md).

For checks that do not load weights, create `work/` and run
`.venv/bin/python -m unittest discover -s tests -v`. With Node installed, also
run `node --test tests/test_demo_polish.cjs tests/test_analyst_tools.cjs`.
The checkpoint verifier requires the original local Git repository and its tags.
The publication snapshot intentionally omits the final manifest and duplicate
edge file, so it cannot run the complete split audit against the original
three-split evidence.
