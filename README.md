# Granite 4.0-3B — Jetson Orin Nano 8GB Perfect Settings

The most-perfect launch configuration for **Granite 4.0 Micro (3.4B hybrid Mamba2)** on
Jetson Orin Nano 8GB — max context, max tokens, verified empirically 2026-09-30 —
ready for a local link to the Hermes agent.

**The headline: this model's full native 128K context fits in 8GB, and full GPU
offload (ngl 99) is ~3x faster than the old hybrid guidance.**

- BFCL v2 champion: 50/52 (96.2%) — best function-calling model tested on this hardware
- 3 launch profiles: 32K / 64K / 128K context, each at its KV-quality ceiling
- One-command launch → OpenAI-compatible endpoint → Hermes `custom` provider

## Quick start

```bash
bash launch_granite4.sh 64k     # default: 64K ctx, q8_0 KV, ~19.5 tok/s
bash launch_granite4.sh 32k    # f16 KV, ~20 tok/s — fastest, full precision
bash launch_granite4.sh 49k    # q8_0 KV, deep-load safe (gateway resident)
bash launch_granite4.sh 128k   # q4_0 KV — FULL native context window

# Verify (short + deep-gen + tool-calling smoke test):
python3 scripts/benchmark_granite4.py
```

## Link to Hermes agent

The server is OpenAI-compatible (`--jinja`), so Hermes's `custom` provider links
directly (see `config/hermes-granite4.yaml`):

```bash
hermes config set model.default granite4-3b
hermes config set model.provider custom
hermes config set model.base_url http://127.0.0.1:8090/v1
hermes config set model.api_key none
hermes config set model.context_length 65536    # match launched profile
```

## The three verified profiles

| Profile | Context | KV cache | Gen speed | RSS | Free after | Use for |
|---|---|---|---|---|---|---|
| `32k` | 32,768 | f16 | 20.0 tok/s | 5.07 GB | 0.85 GB | fastest, best quality — tool calling |
| `49k` | 49,152 | q8_0 | 19.4 tok/s | 4.72 GB | 1.16 GB | deep-load safe — use with Hermes gateway resident |
| `64k` | 65,536 | q8_0 | 19.5 tok/s | 5.33 GB | 0.53 GB | balanced default (standalone use) |
| `128k` | 131,072 | q4_0 | 7.2 tok/s @ 29K depth | 5.66 GB | 0.24 GB | full native context — huge prompts |

All three: `-ngl 99 -b 512 -ub 512 -fa on --jinja --fit off -np 1 -t 6`, launched with
`GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` and GDM stopped.

**Max tokens guidance:** keep prompt+response inside the profile window. On the 128K
profile budget `max_tokens ≤ 16384` so a long generation never overflows. At 64K,
`max_tokens ≤ 24576` is the practical ceiling (prompt budget + headroom). At 32K
use `≤ 12288`.

## Key findings (verified, not assumed)

1. **ngl 99 beats ngl 15 by ~3x.** The old guidance said hybrid SSM models need
   partial offload (ngl 15). On llama.cpp build 84e76d8+ with unified memory, full
   offload works: 6.4 → 19.5 tok/s (64K q8_0). At ngl 15 the KV cache lands on
   CPU — at ngl 99 it lives on GPU (CUDA0), which is where the speed comes from.

2. **The KV ceiling per quant.** Measured (CUDA0 KV buffer at 64K q8_0 = 2720 MiB;
   128K q8_0 would be ~5.4GB → dead; confirmed by test):

   | KV type | Max working context |
   |---|---|
   | f16 | 32K |
   | q8_0 | 64K |
   | q4_0 | 128K (native max) |

   82K/96K q8_0 both fail at compute-buffer allocation even with clean memory.

3. **128K works, with a speed tax at depth.** A 29K-token prompt processed at
   359 tok/s, generation 7.2 tok/s. Use the 128K profile for huge-prompt ingestion,
   not for long generations at depth.

4. **Clean memory release.** Unlike dense Ornith Q4_K_M (GPU-fragmentation lock
   after first run), the hybrid at ngl 99 releases memory cleanly on pkill —
   re-launched 8+ times this session, zero fragment failures.

5. **Under-load OOM (reproduced + root-caused).** With the Hermes gateway
   resident, the 64k profile can be kernel-OOM-killed when a deep-gen fills the
   KV and another request lands on top (journal: `Out of memory: Killed
   process ... llama-server ... anon-rss:5206480kB ... task_memcg=hermes-gateway.service`,
   2026-09-30 21:58). 64k is fine standalone; **use the 49k profile when the
   gateway is running** — 4.72GB RSS leaves 1.16GB headroom even after generation.

## Sampling

The server launches with chat-tuned defaults (temp 0.6, top_p 0.95, top_k 0 —
IBM's chat guidance per Unsloth docs; Cloudflare lists 0.6/0.95 defaults too).

- **Tool calling / function calling:** send `temperature: 0.0, top_p: 1.0` per
  request — IBM's recommendation and the BFCL v2 champion config.
- **Chat:** server defaults are fine.

## Project layout

```
launch_granite4.sh                 one-command launcher (profile arg)
config/hermes-granite4.yaml        Hermes agent link config
scripts/benchmark_granite4.py      verification: speed + deep-gen + tool-call smoke test
results/benchmark_20260930.json   full empirical results (all configs, buffer sizes)
results/benchmark_verify.json      written by the benchmark script
Project_state.md                   living project state
```

## Requirements

- Jetson Orin Nano 8GB, GDM stopped (`sudo systemctl stop gdm.service`)
- llama.cpp build 84e76d8 or newer at `~/llama.cpp/build/bin/llama-server`
- Model: `~/models/new-zoo/granite4-3b.gguf` (2.0GB, Q4_K_M, sha256-verified copy
  of the Ollama blob — no symlink into Ollama's blob store)

## License

MIT © 2026 Walker Kirkpatrick