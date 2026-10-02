# Nanbeige4.2-3B + SmolLM3-3B Jetson Benchmarks — 2026-10-01

llama.cpp 0.5.0-dev (build 965, ec7630a), Jetson Orin Nano 8GB, GDM off,
GGML_CUDA_ENABLE_UNIFIED_MEMORY=1, setsid launch. Both loaded and ran
first-try with standard settings.

## Nanbeige4.2-3B (looped transformer)

Arch nanbeige: 22 blocks (looped x2 = 44 effective), emb 3072, 8 KV heads
x128 key_len, dense attention, native ctx 262144. No thinking switch in
template. Q4_K_M 2.40 GB.

Launch (verified): -c 16384 -ctk q8_0 -ctv q8_0, MemAvail 1.76 GB after load.

| Metric | Value |
|---|---|
| Short-gen | **11.0-11.2 tok/s** |
| Deep-gen pp @ 7.1K | 189 t/s |
| Tool call (greedy) | PASS - get_weather city=Portland, 130 tok, 13.4 s |

Verdict: functional but ~3x slower than LFM2.5-VL-3B. The loop x2 compute
tax is real on this hardware. The 262K native context is unreachable on 8 GB
(dense 88 KB/token: 64K would need 5.6 GB KV alone). Not competitive here.

## SmolLM3-3B

(see below — results appended after test)


## SmolLM3-3B

Arch smollm3: 36 blocks, emb 2048, 4 KV heads x 128, dense attention,
native ctx 65536. Template HAS enable_thinking (default on) - pass
chat_template_kwargs {"enable_thinking": false} to avoid CoT burn.
Q4_K_M 1.8 GB.

Launch (verified): -c 32768 -ctk q8_0 -ctv q8_0.

| Metric | Value |
|---|---|
| Short-gen (thinking off) | 21.4-22.1 tok/s |
| Deep-gen pp @ 7.4K | 637 t/s |
| Tool call (greedy, thinking off) | FAIL (parse) - correct JSON inside code fences in content, no native tool_calls object |

Verdict: solid chat speed (2nd fastest of the new batch), but tool calls
are emitted as fenced JSON in plain content (SmolLM3 uses a nonstandard
"# Tools" header format that llama.cpp jinja does not parse into
tool_calls). Fine as a pure chat model; NOT ready as a Hermes agent link
without a content-wrangling workaround.

## Cross-model comparison (this batch, 8 GB, GDM off)

| Model | Short-gen | pp t/s | Tool calls | File |
|---|---|---|---|---|
| LFM2.5-VL-3B + DSpark | **34-37** | 787-811 | native PASS | 1.56+0.53 GB |
| LFM2.5-VL-3B baseline | 24.5-25.1 | 811 | native PASS | 1.56 GB |
| SmolLM3-3B | 21.4-22.1 | 637 | parse FAIL | 1.8 GB |
| Granite 4.0-3B (prior) | 19.4 | 391 | native PASS | 2.0 GB |
| Nanbeige4.2-3B | 11.0-11.2 | 189 | native PASS | 2.4 GB |
| Granite 4.2-8B IQ3_M | 9.6 | 188 | kwarg needed | 4.0 GB |
