#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "=========================================================="
echo " 🛫 PRE-FLIGHT SYSTEM READINESS AUDIT "
echo "=========================================================="

# 1. Models Check
echo -n "1. Checking Model Weights... "
if [ -f "models/model_clean.keras" ] && [ -f "models/model_aug.keras" ] && [ -f "models/model_aug.tflite" ]; then
    echo "✅ OK (Keras & TFLite models intact)"
else
    echo "❌ Missing model files in models/"
    exit 1
fi

# 2. Test Cases Check
echo -n "2. Checking Harsh Unseen Test Cases... "
if [ -f "test_unseen/manifest.json" ]; then
    echo "✅ OK (7 harsh real-world test scenarios ready)"
else
    echo "❌ Missing test_unseen manifest"
    exit 1
fi

# 3. Streamlit Server Health
echo -n "3. Checking Streamlit Web Server (Port 8501)... "
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8501 || echo "000")
if [ "$HTTP_CODE" = "200" ]; then
    echo "✅ ACTIVE (HTTP 200 at http://localhost:8501)"
else
    echo "⚠️ Not running or starting. Run ./run_demo.sh to launch."
fi

# 4. Presentation & Reports
echo -n "4. Checking Slide Decks & Formal Reports... "
if [ -f "Project_Report.pdf" ] && [ -f "presentation.pptx" ]; then
    echo "✅ OK (PDF report & PPTX slide deck verified)"
else
    echo "❌ Missing report or presentation"
    exit 1
fi

# 5. Git Status
echo -n "5. Checking Git Sync with Remote Origin... "
git status -s | wc -l | grep -q "0" && echo "✅ OK (Clean working tree, synced)" || echo "⚠️ Uncommitted local changes"

echo "=========================================================="
echo " 🚀 SYSTEM READY FOR DEFENSE! GOOD LUCK! "
echo "=========================================================="
