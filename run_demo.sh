#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "=========================================================="
echo " Starting Road Sign Recognition Demo (Option B) "
echo "=========================================================="

.venv/bin/streamlit run app.py --server.port 8501 --server.headless false
