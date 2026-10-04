#!/usr/bin/env bash
set -euo pipefail
export UV_LINK_MODE=copy
UV=/home/satyajeet-rp/.local/bin/uv
cd /home/satyajeet-rp/drona
echo "=== creating venv ==="
$UV venv --python 3.12 .venv
VP=/home/satyajeet-rp/drona/.venv/bin/python
echo "=== torch CPU ==="
$UV pip install --python $VP --index-url https://download.pytorch.org/whl/cpu \
    torch==2.8.0
echo "=== rest (cpu torch already satisfied) ==="
$UV pip install --python $VP \
    "transformers>=4.56" tokenizers safetensors huggingface-hub \
    datasets scikit-learn pandas scipy numpy \
    fastapi "uvicorn[standard]" pydantic jinja2 python-multipart \
    onnx onnxruntime optimum \
    cleanlab matplotlib joblib lxml \
    streamlit pytest
echo "=== verify ==="
$VP - <<'PY'
import importlib.metadata as md
# Every entry here is imported by one of the three run.sh branches. A missing
# one does not fail the install -- it fails later, at demo time, which is worse.
for p in ["torch","transformers","onnxruntime","onnx","scikit-learn","datasets",
          "fastapi","uvicorn","streamlit","pytest","cleanlab","optimum",
          "pandas","numpy","joblib"]:
    try: print(f"{p:16s} {md.version(p)}")
    except Exception: print(f"{p:16s} MISSING")
PY
echo "INSTALL_DONE_OK"
