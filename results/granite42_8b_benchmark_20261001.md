# Granite 4.2-8B Jetson Benchmarks — 2026-10-01

llama.cpp 0.5.0-dev (build 965, commit ec7630a), Jetson Orin Nano 8GB, GDM off,
GGML_CUDA_ENABLE_UNIFIED_MEMORY=1, launched via setsid.

## Verdict

**Not a replacement for Granite 4.0-3B on this hardware.** The 4.2-8B is a
*dense* 40-layer attention model (arch=granite, 8 KV-heads, 128 head_dim, 160
KB/token f16 KV) - unlike the 3B hybrid whose Mamba2 layers offload most KV to
cheap SSM state. Dense 8B math:

- f16 KV @ 42K = 6.9 GB - impossible (5.2 GB available)
- q4_0 KV @ 42K = 3.4 GB + 5.16 GB Q4_K_M weights - impossible
- q4_0 KV @ 16K = 0.64 GB + 4.0 GB IQ3_M weights - viable, tight (1.03 GB headroom measured)

42K context is unreachable at any quant for this model on 8 GB. IQ3_M is the
only quant that leaves usable headroom.

## Measured (IQ3_M 4.0 GB, 16K q4_0 KV, ngl 99)

| Metric | Value |
|---|---|
| Load time | ~10 s |
| Short-gen | **9.6-9.7 tok/s** (thinking on or off - no effect) |
| Deep-gen pp @ 7.4K prompt | 188 t/s |
| Tool call (thinking on) | FAILED - burned 200 tokens in reasoning_content, empty content, no tool call |
| Tool call (chat_template_kwargs enable_thinking:false) | **WORKS** - get_weather city=Portland, 30 tok in 3.3 s |

## The thinking switch (decoded from chat template)

Granite 4.2 templates default enable_thinking = True. Every API call that
does not want chain-of-thought burned into the output must pass:

    {"chat_template_kwargs": {"enable_thinking": false}}

Optional: reasoning_effort "low" sets low-effort mode (template var
low_effort). Template also supports truncate_history_thinking for trimming
CoT from older turns.

## Launch command (verified)

    GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 \
    llama-server -m ~/models/new-zoo/granite-4.2-8b-IQ3_M.gguf \
      --alias granite42-8b --host 127.0.0.1 --port 8090 -ngl 99 \
      -c 16384 -ctk q4_0 -ctv q4_0 -b 512 -ub 512 -fa on --jinja --fit off -np 1 -t 6

Via setsid, never the terminal background wrapper (memcg OOM trap).

## Comparison vs incumbents (8 GB, GDM off)

| Model | Quant | Ctx | Gen | Tool calls | File |
|---|---|---|---|---|---|
| **Granite 4.0-3B (hybrid)** | Q4_K_M | 49K q8_0 | **19.4 tok/s** | native, greedy | 2.0 GB |
| Granite 4.2-8B (dense) | IQ3_M | 16K q4_0 | 9.6 tok/s | w/ enable_thinking:false | 4.0 GB |
| Ornith-1.0-9B (dense) | IQ3_M | 24K | ~9.5 tok/s | - | 4.4 GB |

Granite 4.0-3B remains champion for this box: 2x speed, 3x context, half the
file, native tool calls. The 4.2-8B agentic RL training is real, but dense
attention makes it strictly slower per token here.

## Files

- granite-4.2-8b-Q4_K_M.gguf (5.16 GB) - kept for reference; no usable context
- granite-4.2-8b-IQ3_M.gguf (4.0 GB) - the only launchable quant
