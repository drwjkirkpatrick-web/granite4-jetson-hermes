# Project State: Granite 4.0-3B Jetson + Hermes

> **Last updated:** 2026-09-30
> **Current phase:** build
> **Overall health:** green

---

## 1. Goal (1–2 sentences)
The most-perfect settings for Granite 4.0-3B (BFCL v2 champion, 96.2%) on Jetson
Orin Nano 8GB: max context, max tokens, verified empirically — ready for a local
link to the Hermes agent.

## 2. Current Status
### Done
- [x] Model file secured: real 2.0GB copy at ~/models/new-zoo/granite4-3b.gguf
      (was a symlink chain into Ollama's blob store; sha256 verified)
- [x] Architecture decoded from GGUF: Granite 4.0 Micro, 3.4B, hybrid Mamba2
      (LLM_ARCH_GRANITE_HYBRID), 40 blocks, native ctx 131072, vocab 100352
- [x] ngl ladder test: ngl 99 (full offload) works on build 84e76d8+ → ~3x faster
      than old ngl-15 hybrid guidance (6.4 → 19.5 tok/s @ 64K q8_0)
- [x] Context ladder: f16 32K / q8_0 64K / q4_0 128K all work; q8_0 82K+ dies;
      q8_0 96K/128K dies; exact buffer sizes captured from logs
- [x] 128K deep test: 29K-token prompt @ 359 t/s pp, 7.2 t/s gen at depth
- [x] Under-load OOM reproduced + root-caused via journal (kernel OOM-killed
      llama-server: 64k profile, full KV, Hermes gateway resident)
- [x] 49k profile added (q8_0, 4.72GB RSS, 1.16GB free) - deep-load safe
- [x] launch_granite4.sh — 4-profile launcher with pre-flight checks (49k added)
- [x] config/hermes-granite4.yaml — Hermes custom-provider link
- [x] scripts/benchmark_granite4.py — verification suite (speed + deep + tool call)
- [x] results/benchmark_20260930.json — full empirical record
- [x] README.md

### In Progress
- [ ] End-to-end Hermes verification (launch + benchmark script + hermes one-shot)

### Not Started
- [ ] Git commit + GitHub repo + push
- [ ] Optional: Hermes profile for granite-only sessions

## 3. Architecture & Key Decisions
| Decision | Rationale | Date |
|---|---|---|
| ngl 99 (full offload) for hybrid SSM | ~3x faster than ngl 15 on build 84e76d8+; KV lands on GPU instead of CPU; verified across 8+ launches | 2026-09-30 |
| 3 profiles (32k f16 / 64k q8_0 / 128k q4_0) | Each KV quant has a hard context ceiling; one size cannot fit all | 2026-09-30 |
| 64k as default profile | 19.5 tok/s, 2x context over 32k for ~2% speed | 2026-09-30 |
| 49k profile for gateway-resident use | OOM-kill under full KV + gateway load; 49k leaves 1.16GB headroom | 2026-09-30 |
| Real file copy, not symlink, into new-zoo | Old symlink chained into Ollama blob store — breaks silently if Ollama prunes | 2026-09-30 |
| Sampling: server 0.6/0.95 chat defaults; greedy per request for tools | IBM recommendation + BFCL champion config | 2026-09-30 |
| Hermes link via custom provider to :8090 | llama-server --jinja is OpenAI-compatible; no bridge needed | 2026-09-30 |

## 4. Blockers & Risks
- **Risk:** 128K profile has only ~0.24GB headroom — launch it first, nothing else
  running, and budget max_tokens ≤ 16384.
- **Risk:** 64k profile can be kernel-OOM-killed if a deep-gen fills the KV while
  the Hermes gateway is resident (reproduced 2026-09-30) — use 49k in that mode.
- **Risk:** GDM eats ~500MB — pre-flight check refuses to launch below 4.5GB avail.
- **Risk:** llama.cpp flag drift (--fit auto-checker) — launcher pins --fit off.

## 5. Next Step (only ONE)
> **Next:** End-to-end verification: launch 64k profile via the launcher script,
> run scripts/benchmark_granite4.py, then `hermes chat -q` one-shot against the
> local server; then git commit + push.

## 6. Environment & Tooling Notes
- Project root: `~/projects/granite4-jetson-hermes/`
- Model: `~/models/new-zoo/granite4-3b.gguf` (2.0GB Q4_K_M)
- Server: `~/llama.cpp/build/bin/llama-server` (build 84e76d8)
- Launch: `bash launch_granite4.sh [32k|64k|128k]`
- Verify: `python3 scripts/benchmark_granite4.py`
- Logs: `/tmp/granite4-3b.log` (launcher), `/tmp/granite-*.log` (ladder tests)

## 7. Recent Session Log
- 2026-09-30: Full empirical tuning session. GGUF metadata decoded (hybrid
  Mamba2, native 128K ctx). ngl 99 vs 15 ladder (user-suggested — 3x win).
  Context/KV ladder to find ceilings. 128K q4_0 confirmed working with 29K-token
  deep test. Launcher + Hermes config + benchmark script + README written.

## 8. References
- BFCL v2 results: `~/projects/bfcl-jetson-test-v2/` (granite4-3b 50/52 champion)
- jetson-llm-launch skill (launch patterns, /props polling, OOM rules)
- IBM/Unsloth sampling guidance (greedy for tools)