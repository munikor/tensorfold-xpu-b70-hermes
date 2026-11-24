# TensorFold XPU on Arc Pro B70 — Qwen3.8-27B Hermes-adapted, GPTQ-Int4

A single-card Intel Arc Pro B70 serving lane for a **Hermes-tool-trained fine-tune of Qwen3.8-27B**,
quantized to GPTQ int4 (g128) and served by [TensorFold](https://github.com/ashhart/TensorFold)'s
native XPU engine with a DFlash2 drafter, int8 KV cache, 200K context, and structured output /
tool calling via `libtfgrammar` (xgrammar v0.2.8).

No model weights live in this repository. This repo holds the build recipe, the serving image
definition, and the measured results.

## Results (1 card, 200K context, 2026-11-24)

Measured by a streaming client on another host over LAN; greedy sampling.

| Category | Decode (t/s) | TTFT |
|---|---|---|
| Code | 160.6 | 0.22 s |
| Reasoning / math | 145.5 | 0.09 s |
| JSON (structured) | 170.6 | 0.14 s |
| Prose | 113.1 | 0.09 s |
| Chat / explain | 108.8 | 0.10 s |
| Edit / rewrite | 110.6 | 0.09 s |
| Summarize | 80.5 | 0.17 s |

Aggregate throughput when the context is traded for slots:

| Config | 1 stream | 2 streams | 4 streams |
|---|---|---|---|
| 200K ctx · 1 slot | 108.8 (chat) | — | — |
| 65K ctx · 2 slots | 104.4 | 180.2 | — |
| 32K ctx · 4 slots | 97.4 | 179.1 | **262.9** |

Depth (prefill and decode with a long document in context):

| Prompt tokens | TTFT | Prefill t/s | Decode at depth |
|---|---|---|---|
| 5,986 | 3.20 s | 1,869 | — |
| 23,866 | 13.98 s | 1,708 | 53.3 t/s |
| 80,486 | 59.68 s | 1,349 | 27.8 t/s |

Long-context recall verified at 108,823 and 141,603 prompt tokens (buried fact returned exactly).
Quality: 10/10 checkable questions (greedy). Required tool call: correct typed arguments in 0.79 s;
`tool_choice: auto` selects the right function and emits no call for plain chat.

Raw numbers: [`bench/bench175.json`](bench/bench175.json) · harness: [`bench/bench175.py`](bench/bench175.py) ·
render the interactive report with `python3 bench/gen_report.py` (writes `bench/bench175.html`).

## Why it is fast on one card

VTune GPU-hotspot profiling of a real 400-token generation showed the card is **not** bandwidth- or
compute-bound: XVE active 7.3%, idle 74.7%, L3-bandwidth-bound 0.0%, occupancy 27% of peak. Decode is
launch/latency-bound, which is why concurrent streams scale aggregate throughput nearly linearly
(262.9 t/s at four streams) while per-stream speed stays flat.

## Model provenance

- **Base:** `Qwen/Qwen3.8-27B` (BF16, 64 layers, hybrid linear attention)
- **Adapter:** `harness-hub/Qwen3.8-27B-hermes-adapted` (PEFT LoRA, alpha 64 — GRPO on agent/tool
  trajectories for the Hermes harness)
- **Bake:** adapter merged into BF16 base, then GPTQ int4:
  - `bits=4, group_size=128, sym=True, desc_act=False`
  - `lm_head` left dense (int4 head breaks the XPU loader)
  - `embed_tokens` excluded from quantization
  - vision tower (`model.visual.*`) dropped — text-only serving
  - every remaining float tensor cast to F16, `qweight`/`qzeros`/`g_idx` stay I32
  - final size ≈ 17 GB, single safetensors set with rewritten index

Quantization was done with gptqmodel 7.3.2 + optimum on an aarch64 Grace box; calibration mixed
wikitext-2 with agent-shaped prompts.

Note: the DFlash2 drafter is trained against **base** Qwen3.8-27B, so draft acceptance on this
fine-tune sits around 24–37% rather than the base model's numbers — it still buys a large speedup.

## Engine

TensorFold `tensorfold-native` (Zig, Apache-2.0) built for the XPU lane:

```
zig build xpu-native -Doptimize=ReleaseFast -Dngen-include=<oneDNN nGEN headers>
```

Structured output needs `libtfgrammar.so`, built from an xgrammar v0.2.8 checkout:

```
git clone --branch v0.2.8 --recurse-submodules https://github.com/mlc-ai/xgrammar.git
zig build grammar -Dxgrammar=/path/to/xgrammar
```

The server looks for the library in `<server dir>/../lib/` (or beside the binary). Without it,
`tool_choice: required` and structured output return HTTP 400.

## Serving

```
docker run --device /dev/dri --group-add $(getent group render | cut -d: -f3) \
  --security-opt seccomp=unconfined --ipc=host --shm-size 32g \
  -p 8202:8000 \
  -v /path/to/hermes-gptq-int4:/models/model:ro \
  -v /path/to/DFlash2-27B:/drafter:ro \
  -e ZE_AFFINITY_MASK=0 -e XPU_KV_INT8=1 \
  tensorfold-xpu-b70:hermes-2026.11.24 \
  serve /models/model --host 0.0.0.0 --port 8000 \
  --context 200000 --prompt-cache-gib 0.25 --drafter /drafter --no-thinking
```

Context is traded for slots: 200K → 1 stream, 131K → 1, 65K → 2, 32K → 4 (`--parallel N`).

## Hardware notes

- 2× Intel Arc Pro B70, Ubuntu 26.04, `xe` kernel driver, Level Zero 1.35.
- On this box the driver does **not** export `power1_input`; board power must be derived from
  `energy1_input` deltas (≈190 W under decode, ≈6 W idle). A dashboard reading `power1` shows
  nothing because the file is absent — not because the card is idle.
- Card index ≠ hwmon index: the two B70s appear as `card1`/`card2` → `hwmon7`/`hwmon8`;
  `card0` has no hwmon and reads `gt_cur_freq_mhz=0`.

## Licences

TensorFold is Apache-2.0. xgrammar is Apache-2.0. The serving image's base is
`intel/llm-scaler-vllm` (Intel, Apache-2.0), used for its Level Zero / oneAPI runtime. The model
weights are **not** distributed here; the base model and adapter each carry their own licence —
check them at their source before redistributing a quantized export.

## Reproduce

1. Merge the adapter into the BF16 base (PEFT `merge_and_unload`), copy the VL processor files
   (`preprocessor_config.json`, `video_preprocessor_config.json`, `chat_template.jinja`) — `save_pretrained`
   drops them and GPTQModel then fails with "Can't load image processor".
2. Quantize with gptqmodel 7.3.2 using the config above; cast dense tensors to F16 afterwards.
3. Build TensorFold `xpu-native` and `libtfgrammar`, install the library next to the server.
4. `docker build -t tensorfold-xpu-b70:<tag> .` from this repo's `Dockerfile`.
5. Serve as above; bench with `bench/bench175.py` (see the JSON for the exact prompts).
