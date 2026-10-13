#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ ! -x .venv-demo/bin/python ]]; then
  echo 'Lancez bash install.sh avant de lancer cette application.' >&2
  exit 1
fi
exec .venv-demo/bin/python -m streamlit run app.py --server.port "${PORT:-8501}" --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false
