#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run.sh — Launch the RAG Chatbot with AMD ROCm GPU support
# GPU   : AMD RX 6500M (RDNA2, gfx1034)
# LLM   : phi4-mini via Ollama
# Embed : nomic-embed-text via Ollama
# ─────────────────────────────────────────────────────────────────────────────
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── ROCm / HSA overrides for RX 6500M (gfx1034 = RDNA2) ─────────────────────
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export ROCR_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0
export OLLAMA_NUM_GPU=1
export OLLAMA_GPU_OVERHEAD=256MiB
export OLLAMA_FLASH_ATTENTION=1        # faster attention on RDNA2

# Optional: suppress HSA debug noise
export HSA_ENABLE_SDMA=0

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🧠 RAG Chatbot — Local · AMD GPU · phi4-mini"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  GPU  : AMD RX 6500M (gfx1034, RDNA2)"
echo "  LLM  : phi4-mini"
echo "  Embed: nomic-embed-text"
echo "  HSA  : $HSA_OVERRIDE_GFX_VERSION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Check Ollama is running ───────────────────────────────────────────────────
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo ""
    echo "⚠️  Ollama is not running. Starting it..."
    ollama serve &
    OLLAMA_PID=$!
    sleep 3
    echo "✅  Ollama started (PID $OLLAMA_PID)"
fi

# ── Pull required models if missing ──────────────────────────────────────────
echo ""
echo "🔍 Checking required models..."

MODELS=$(ollama list 2>/dev/null | awk 'NR>1 {print $1}')

if ! echo "$MODELS" | grep -q "phi4-mini"; then
    echo "📥 Pulling phi4-mini (≈2.5 GB)..."
    ollama pull phi4-mini
fi

if ! echo "$MODELS" | grep -q "nomic-embed-text"; then
    echo "📥 Pulling nomic-embed-text (≈274 MB)..."
    ollama pull nomic-embed-text
fi

echo "✅ Models ready."
echo ""

# ── Activate virtualenv ───────────────────────────────────────────────────────
if [ -d "$SCRIPT_DIR/myvenv" ]; then
    source "$SCRIPT_DIR/myvenv/bin/activate"
    echo "🐍 Virtualenv activated: myvenv"
else
    echo "⚠️  No myvenv found — using system Python"
fi

# ── Launch Streamlit ──────────────────────────────────────────────────────────
echo ""
echo "🚀 Starting Streamlit at http://localhost:8501"
echo "   Press Ctrl+C to stop."
echo ""

streamlit run app.py \
    --server.port 8501 \
    --server.headless true \
    --browser.gatherUsageStats false \
    --theme.base dark \
    --theme.primaryColor "#6366f1" \
    --theme.backgroundColor "#0d0d1a" \
    --theme.secondaryBackgroundColor "#111128" \
    --theme.textColor "#e2e8f0"
