#!/usr/bin/env bash
# DRONA demo entrypoint. Everything runs offline on CPU.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=src
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2   # 2 physical cores
case "${1:-api}" in
  api)       exec .venv/bin/python -m drona.service ;;          # http://127.0.0.1:8000/docs
  # 0.0.0.0, not 127.0.0.1 -- a loopback bind is unreachable from any other
  # machine, so the dashboard appears not to run. headless +
  # gatherUsageStats=false are load-bearing too: without them Streamlit 1.65
  # runs a first-run email prompt and, with no TTY on stdin, aborts with
  # exit 255 instead of serving.
  dashboard) exec .venv/bin/streamlit run src/drona/dashboard.py \
               --server.port 8501 --server.address 0.0.0.0 \
               --server.headless true --browser.gatherUsageStats false ;;
  test)      exec .venv/bin/python -m pytest -q tests ;;
  *)         exec .venv/bin/python -c "from drona.engine import get_engine; \
      print(get_engine().triage('my card was declined twice and money was debited, urgent').category)" ;;
esac
