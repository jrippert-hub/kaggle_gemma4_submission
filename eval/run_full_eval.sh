#!/usr/bin/env bash
# run_full_eval.sh — End-to-end Caspar evaluation pipeline
# Usage: cd eval && bash run_full_eval.sh [--n 50]
set -euo pipefail

N=${2:-50}
if [[ "${1:-}" == "--n" ]]; then N=$2; fi

CASPAR_DIR="$(dirname "$0")/.."
EVAL_DIR="$(dirname "$0")"
CASPAR_PID=""

cleanup() {
    if [[ -n "$CASPAR_PID" ]]; then
        echo "── Stopping Caspar (pid $CASPAR_PID) ──"
        kill "$CASPAR_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ── Step 1: Generate datasets ──────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════"
echo "  Step 1 — Load datasets (n=$N each)"
echo "══════════════════════════════════════════"
cd "$EVAL_DIR"
python load_datasets.py --n "$N"

# ── Step 2: Start Caspar in background ────────────────────────────────────
echo ""
echo "══════════════════════════════════════════"
echo "  Step 2 — Starting Caspar"
echo "══════════════════════════════════════════"
cd "$CASPAR_DIR"
TEST_MODE=true OLLAMA_CHAT_MODEL="${OLLAMA_CHAT_MODEL:-gemma4:e4b}" \
    uvicorn main:app --host 127.0.0.1 --port 8000 > /tmp/caspar.log 2>&1 &
CASPAR_PID=$!
echo "  Caspar pid: $CASPAR_PID"

# Wait for healthy
echo "  Waiting for /health…"
for i in $(seq 1 30); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "  ✓ Caspar is up"
        break
    fi
    if [[ $i -eq 30 ]]; then
        echo "  ERROR: Caspar did not start in time. Check /tmp/caspar.log"
        exit 1
    fi
    sleep 2
done

# ── Step 3: Run evals ──────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════"
echo "  Step 3 — Running evaluations"
echo "══════════════════════════════════════════"
cd "$EVAL_DIR"

DATASETS=(
    "datasets/multi_turn/escalating"
    "datasets/multi_turn/benign"
    "datasets/long_context/escalating"
    "datasets/long_context/benign"
)

for DS in "${DATASETS[@]}"; do
    if [[ -d "$DS" ]] && compgen -G "$DS/*.json" > /dev/null 2>&1; then
        echo ""
        echo "── Evaluating: $DS ──"
        python run_eval.py --dataset "$DS" --baseline
    else
        echo "  Skipping $DS (no files)"
    fi
done

# ── Step 4: Compute metrics ────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════"
echo "  Step 4 — Computing metrics"
echo "══════════════════════════════════════════"
python metrics.py

echo ""
echo "══════════════════════════════════════════"
echo "  ✓ Eval complete — see results/summary.json"
echo "     and results/plots/ for risk arc charts"
echo "══════════════════════════════════════════"
