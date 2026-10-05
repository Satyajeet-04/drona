#!/usr/bin/env bash
# DRONA installer. Portable: resolves its own location, finds uv or falls back
# to stdlib venv, and never writes an absolute path from the build machine.
#
#   ./install-ml.sh          # lite -- everything needed to run the system
#   ./install-ml.sh full     # + torch/transformers, only for re-running the
#                            #   DistilBERT comparison (~47 min on 2 cores)
#
# Why "lite" is enough: the served model is artifacts/models/cascade.joblib,
# loaded with joblib. src/drona/agent.py imports torch and transformers *inside*
# functions, and nothing at inference calls them (the agent is advisory-only, so
# /health reports llm_enabled=false). A fresh clone therefore does not need a
# 200 MB CPU torch wheel to run the API, the dashboard or the tests.
set -euo pipefail
cd "$(dirname "$0")"
export UV_LINK_MODE=copy

MODE="${1:-lite}"
case "$MODE" in
  lite|full) ;;
  *) echo "usage: ./install-ml.sh [lite|full]  (got '$MODE')" >&2; exit 2 ;;
esac

# --- locate a package installer -------------------------------------------
UV="$(command -v uv || true)"
if [ -z "$UV" ] && [ -x "$HOME/.local/bin/uv" ]; then UV="$HOME/.local/bin/uv"; fi

# --- locate a base interpreter -------------------------------------------
PY=""
for cand in python3.12 python3.11 python3; do
  if command -v "$cand" >/dev/null 2>&1; then PY="$(command -v "$cand")"; break; fi
done
if [ -z "$PY" ]; then
  echo "ERROR: no python3 found on PATH. Install Python 3.11 or 3.12 and retry." >&2
  exit 1
fi
echo "=== interpreter: $PY ($("$PY" -V 2>&1)) ==="

if [ -n "$UV" ]; then
  echo "=== package installer: $UV ==="
  "$UV" venv --clear --python "$PY" .venv
  # shellcheck disable=SC2317  # referenced indirectly by pip_install
  pip_install() { "$UV" pip install --python .venv/bin/python "$@"; }
else
  echo "=== package installer: stdlib venv + pip (uv not found) ==="
  echo "    tip: 'pip install uv' or 'curl -LsSf https://astral.sh/uv/install.sh | sh'"
  "$PY" -m venv --clear .venv
  pip_install() { .venv/bin/python -m pip install "$@"; }
  pip_install --upgrade pip
fi

LITE_DEPS=(
  numpy scipy scikit-learn joblib
  fastapi "uvicorn[standard]" pydantic jinja2 python-multipart
  streamlit pytest
)
FULL_DEPS=(
  torch transformers tokenizers safetensors huggingface-hub
  datasets pandas onnx onnxruntime optimum
  cleanlab matplotlib lxml
)

if [ "$MODE" = lite ]; then
  echo "=== installing lite dependencies (system runs on these alone) ==="
  pip_install "${LITE_DEPS[@]}"
  VERIFY=("numpy" "scipy" "sklearn" "joblib" "fastapi" "uvicorn"
          "streamlit" "pytest" "pydantic")
else
  echo "=== installing full dependencies (includes CPU torch) ==="
  # torch comes from the PyTorch CPU index; the rest from PyPI.
  pip_install --index-url https://download.pytorch.org/whl/cpu "torch==2.8.0"
  pip_install "${LITE_DEPS[@]}" "${FULL_DEPS[@]}"
  VERIFY=("numpy" "scipy" "sklearn" "joblib" "fastapi" "uvicorn" "streamlit"
          "pytest" "pydantic" "torch" "transformers" "onnxruntime" "cleanlab")
fi

echo "=== verify ==="
# Import, don't read distribution metadata: the import name and the
# distribution name differ for at least one of these (sklearn is distributed as
# scikit-learn), and importing is the thing that actually has to work.
.venv/bin/python - "${VERIFY[@]}" <<'PY'
import importlib, sys
bad = []
for name in sys.argv[1:]:
    try:
        m = importlib.import_module(name)
        print(f"  {name:16s} {getattr(m, '__version__', 'imported')}")
    except Exception as e:
        print(f"  {name:16s} MISSING ({type(e).__name__}: {e})")
        bad.append(name)
if bad:
    print("\nFAILED to import:", ", ".join(bad), file=sys.stderr)
    sys.exit(1)
PY

echo "=== smoke test: load the shipped model ==="
PYTHONPATH=src .venv/bin/python - <<'PY'
import numpy, sklearn, joblib, scipy
m = joblib.load("artifacts/models/cascade.joblib")
print("  cascade.joblib loaded:", type(m).__name__)
PY

cat <<'NEXT'
INSTALL_DONE_OK

Next:
  ./run.sh test        32 tests
  ./run.sh             API      -> http://localhost:8000/docs
  ./run.sh dashboard   Dashboard -> http://localhost:8501
NEXT
