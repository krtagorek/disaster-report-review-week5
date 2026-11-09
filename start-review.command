#!/bin/zsh
# Finder-friendly launcher; always use this repository and its isolated environment.
cd -- "${0:A:h}"
if [[ ! -x .venv/bin/python ]]; then
  print 'The project environment is missing. Follow docs/LOCAL_MODEL_SETUP.md first.'
  exit 1
fi
exec .venv/bin/python -m review_app "$@"
