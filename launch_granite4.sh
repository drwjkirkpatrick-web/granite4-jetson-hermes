#!/usr/bin/env bash
# launch_granite4.sh — Granite 4.0-3B (Granite 4.0 Micro, 3.4B hybrid Mamba2)
# on Jetson Orin Nano 8GB. Max context / max tokens per profile, verified 2026-09-30.
#
# Usage:
#   bash launch_granite4.sh [profile]    # default: 64k
#   bash launch_granite4.sh 32k          # f16 KV, fastest
#   bash launch_granite4.sh 49k          # q8_0 KV, deep-load safe (gateway resident)
#   bash launch_granite4.sh 64k          # q8_0 KV, balanced   (default)
#   bash launch_granite4.sh 128k         # q4_0 KV, full native context
#
# Verified numbers (llama.cpp build 84e76d8, GDM off, GGML_CUDA_ENABLE_UNIFIED_MEMORY=1):
#   32k  f16 : ~20.0 tok/s gen, RSS ~5.07 GB
#   64k  q8_0: ~19.5 tok/s gen, RSS ~5.33 GB   <- default
#   128k q4_0: full native context; pp ~359 t/s; gen ~7.2 tok/s at 29K-token depth
#
# KEY FINDING: ngl 99 (full offload) works on build 84e76d8+ with unified memory
# and is ~3x faster than the old hybrid guidance (ngl 15: 6.4 tok/s -> 19.5 tok/s).
#
# Sampling: server defaults are chat-tuned (temp 0.6, top_p 0.95).
# For tool calling / function calling set temp=0.0, top_p=1.0, top_k=0 per request
# (IBM's recommendation — BFCL v2 champion settings).

set -euo pipefail

PROFILE="${1:-64k}"
MODEL="$HOME/models/new-zoo/granite4-3b.gguf"
SERVER="$HOME/llama.cpp/build/bin/llama-server"
PORT="${PORT:-8090}"
LOG="${LOG:-/tmp/granite4-3b.log}"

case "$PROFILE" in
  32k)
    CTX=32768;   CTKV=f16 ;;
  49k)
    CTX=49152;   CTKV=q8_0 ;;
  64k)
    CTX=65536;   CTKV=q8_0 ;;
  128k)
    CTX=131072;  CTKV=q4_0 ;;
  *)
    echo "Unknown profile: $PROFILE (use 32k | 49k | 64k | 128k)"; exit 1 ;;
esac

# Pre-flight: free memory (GDM off recommended; frees ~500MB)
MEM_AVAIL=$(awk '/MemAvailable/ {printf "%.1f", $2/1048576}' /proc/meminfo)
MEM_OK=$(awk '/MemAvailable/ {print ($2/1048576 >= 4.5) ? "yes" : "no"}' /proc/meminfo)
if [ "$MEM_OK" = "no" ]; then
  echo "WARNING: only ${MEM_AVAIL}GB available. Stop GDM first:"
  echo "  sudo systemctl stop gdm.service"
  echo "  pkill -9 -f 'llama-serv[e]r'"
  exit 1
fi

# Kill any stale server (bracket pattern so we don't kill our own shell)
pkill -9 -f 'llama-serv[e]r' 2>/dev/null || true
sleep 2

echo "Launching Granite 4.0-3B [$PROFILE] ctx=$CTX kv=$CTKV port=$PORT ..."
echo "Log: $LOG"

GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 \
nohup "$SERVER" \
  -m "$MODEL" \
  --alias granite4-3b \
  --host 127.0.0.1 --port "$PORT" \
  -ngl 99 \
  -c "$CTX" -ctk "$CTKV" -ctv "$CTKV" \
  -b 512 -ub 512 -fa on \
  --jinja --fit off -np 1 -t 6 \
  --temp 0.6 --top-p 0.95 --top-k 0 \
  > "$LOG" 2>&1 &

SERVER_PID=$!
echo "PID: $SERVER_PID"
echo "Waiting for model load (~8-10s) ..."

for i in $(seq 1 45); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "SERVER DIED — check $LOG"; tail -5 "$LOG"; exit 1
  fi
  # /props (not /health): returns 200 only after full VRAM load
  if curl -s -m 2 "http://127.0.0.1:$PORT/props" > /dev/null 2>&1; then
    echo ""
    echo "READY: http://127.0.0.1:$PORT  (OpenAI-compatible /v1)"
    echo "Hermes: hermes -m granite4-3b --provider custom"
    echo "       (or apply config/hermes-granite4.yaml)"
    exit 0
  fi
  sleep 2
done
echo "TIMEOUT waiting for server — check $LOG"
exit 1