#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
"$PYTHON" -m src.data
"$PYTHON" -m pytest -q
"$PYTHON" -m src.audit
"$PYTHON" -u -m src.train supervised
"$PYTHON" -u -m src.train rl
"$PYTHON" -u -m src.final
"$PYTHON" -m src.audit
"$PYTHON" -m src.report
