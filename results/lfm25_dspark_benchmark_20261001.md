# LFM2.5-VL-3B + DSpark Jetson Benchmarks — 2026-10-01

llama.cpp 0.5.0-dev (build 965, ec7630a), Jetson Orin Nano 8GB, GDM off,
GGML_CUDA_ENABLE_UNIFIED_MEMORY=1, setsid launch.

## Verdict

**NEW SPEED CHAMPION for this box.** 34-37 tok/s generation - the fastest
model ever measured here, beating Granite 4.0-3B (19.4) and Ornith-9B (9.5).
Vision-capable (mmproj on disk), 32K native context, native tool calls.

## Measured

| Config | Short-gen | Deep-gen pp (7.7K prompt) |
|---|---|---|
| **+ DSpark drafter** | **34-37 tok/s** | 787-811 t/s |
| Baseline (no drafter) | 24.5-25.1 tok/s | 811 t/s |

DSpark speculative decode contribution: **+40% generation speed** (pp
identical, as expected). Acceptance 38-50% on prose, 100% on the tool call
(n_max=3). The 567MB drafter is cheap.

Tool call (greedy, temp 0.0): PASS - get_weather(city="Portland", unit="c"),
16 tokens, 0.8 s, no thinking burn (Qwen-style <think> soft switch; LFM
template preserves prior-turn reasoning only if requested).

## Launch command (verified first-run)

    GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 \
    llama-server -m ~/models/new-zoo/LFM2.5-VL-3B-Q4_K_M.gguf \
      -md ~/models/new-zoo/LFM2.5-VL-3B-DSpark-F16.gguf \
      --spec-type draft-dspark \
      --alias lfm25-3b --host 127.0.0.1 --port 8090 -ngl 99 -c 32768 \
      -ctk q8_0 -ctv q8_0 -b 512 -ub 512 -fa on --jinja --fit off -np 1 -t 6

## Model details (decoded from GGUF)

- arch lfm2: 30 blocks, hybrid - only 10 of 30 blocks have attention
  (head_count_kv = 8 on those, 0 elsewhere); conv + GDN (gated deltalNet)
  layers elsewhere. KV/token f16 ~= 20 KB (10 attn layers x 8 kvh x 128).
- 32K context natively (context_length = 32768; no YaRN extension needed).
- emb 2048, head_count 32.
- Thinking: template uses message-passing for <think> blocks only -
  NO default CoT burn. preserve_thinking=false by default.

## Files

- LFM2.5-VL-3B-Q4_K_M.gguf (1.56 GB)
- LFM2.5-VL-3B-DSpark-F16.gguf (0.53 GB) - draft model
- mmproj-LFM2.5-VL-3B-Q8_0.gguf (0.54 GB) - vision projector (untested)
