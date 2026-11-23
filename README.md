# mini-vLLM

Fork of [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) @ `bb823b3` with a mixed-batch scheduler, its own KV block manager, Llama TP and OLMoE expert parallelism, an OpenAI-compatible streaming server, CPU tests and benchmarks.

## Changes vs upstream

| | upstream | mini-vLLM |
|---|---|---|
| one step | all-prefill **or** all-decode | decode tokens **and** prefill chunks together |
| chunked prefill | only the head of the waiting queue | any number of waiting requests, each chunkable |
| prefill/decode state | `is_prefill` flag set by the scheduler | derived from `num_cached_tokens` vs `num_tokens` (`Sequence.is_prefill`) |
| KV block manager | free deque; a cached block is lost as soon as its slot is reused | `block_manager.py` rewritten: refcounts, prefix hash chain, true LRU eviction of cached-but-unreferenced blocks, hit/eviction stats |
| models | Qwen3 | + Llama-3.x (GQA, RoPE llama3 scaling), + OLMoE (MoE) |
| parallelism | TP for Qwen3 | TP for Llama; expert parallel for OLMoE with data-parallel attention (`utils/parallel.py`) |
| MoE | none | router, top-k, grouped expert GEMM (Triton kernel on GPU, torch fallback on CPU), EP all_to_all dispatch/combine |
| metrics | throughput | + TTFT/ITL p50/p95, MFU (prefill), MBU (decode) in `bench/metrics.py` |
| serving | offline `generate()` | `server.py`: `/v1/chat/completions`, `/v1/completions`, SSE streaming |
| decode attention | `flash_attn_with_kvcache` | + own Triton paged decode kernel (`decode_attention_backend="triton_paged"`): PagedAttention with GQA grouping and split-K, see below |

Scheduling (`nanovllm/engine/scheduler.py`, `enable_mixed_batch=True`):
1. Every running request gets 1 decode token (preempt from the tail if out of KV blocks).
2. Leftover `max_num_batched_tokens` budget is filled FCFS with prefill chunks.
3. A step with any prefill chunk runs the varlen FlashAttention path (a decode is a `q_len=1` chunk over the paged cache); a pure-decode step keeps the CUDA-graph path.

`enable_mixed_batch=False` restores the upstream policy for A/B comparison.

KV cache (`nanovllm/engine/block_manager.py`): every block is `used` (ref_count > 0), `evictable` (ref_count 0 but its content hash is registered, so the KV is still valid and a new request with the same prefix revives it) or `free`. Allocation takes free blocks first and only then evicts the least recently released evictable block. Sequences release blocks tail-first, so a prompt's tail is evicted before its head. `LLMEngine.cache_stats()` returns `hit_tokens`, `hit_rate`, `evictions` and block counts. Matching compares token ids as well as the 64-bit hash.

Expert parallelism (`nanovllm/layers/moe.py`, `nanovllm/models/olmoe.py`, `nanovllm/engine/ep_engine.py`):
1. Each rank runs attention on its own requests (own scheduler and KV cache); attention weights are replicated.
2. Router: softmax over all experts, top-8 (OLMoE does not renormalise the top-k weights).
3. Dispatch: each (token, expert) pair is sent to the rank owning the expert (`all_to_all_single` with data-dependent split sizes). A token with several experts on one rank is sent once per expert (no dedup).
4. Owning rank sorts rows by expert and runs two grouped GEMMs (gate+up, down).
5. Combine: rows return to the sending rank in send order, are un-permuted, multiplied by the routing weights and summed over k.
6. A rank with no work runs a 1-token dummy forward so the collectives stay matched. EP forces `enforce_eager` (split sizes are data dependent, so no CUDA graph).

Paged decode kernel (`nanovllm/layers/paged_attention.py`), used for pure-decode steps when `decode_attention_backend="triton_paged"`:
1. Reproduces the decode kernel of PagedAttention (vLLM, SOSP'23): each sequence's K/V is read page by page through its block table.
2. GQA grouping: one program per (sequence, KV head) handles all query heads of that group, so a KV page is loaded once per group, not once per query head.
3. Split-K (PagedAttention V2 / Flash-Decoding): the context is cut into `partition_size`-token pieces that run in parallel; a second kernel merges the partial (max, sum, output) with the usual softmax rescaling.
4. The grid depends only on batch size and block-table width, so the kernel works inside the captured decode CUDA graphs.

## Run

```bash
pip install -e . fastapi uvicorn
python server.py --model ~/huggingface/Qwen3-0.6B
python bench/ablation.py --model ~/huggingface/Qwen3-0.6B     # mixed/legacy x shared/unique prefix
python bench/budget_sweep.py --model ~/huggingface/Qwen3-0.6B # max_num_batched_tokens in {512,2048,8192}
bash scripts/run_2x4090.sh all                                  # everything planned for 2x RTX 4090
```

CPU tests:

```bash
# scheduler, block manager, metrics, server: no torch needed (torch tests are skipped)
pip install xxhash numpy pytest fastapi httpx uvicorn
pytest tests
# + TP / MoE / EP tests on CPU with gloo (2 processes) and HF reference models; ~2 minutes
pip install torch transformers
pytest tests
```

## MFU and MBU

Defined in `bench/metrics.py`.
- MFU (prefill) = achieved FLOPs/s / peak. FLOPs = 2 * active params * tokens + causal attention (4 * layers * heads * head_dim * sum of context lengths). Only prefill-only steps are counted.
- MBU (decode) = (weight bytes + KV bytes read per step) / step time / peak bandwidth. For MoE, weight bytes use the expected number of distinct experts hit by the batch. Only pure-decode steps are counted.
- RTX 4090 peaks: 165.2 TFLOPS dense bf16 tensor with fp32 accumulate (330.4 is with 2:4 sparsity; 330.3 is fp16 with fp16 accumulate), 1008 GB/s. Source: NVIDIA Ada GPU Architecture whitepaper V2.02, Appendix A. The 165.2 figure is the correct bf16 number; there is no bf16-with-bf16-accumulate mode.

## Status

| | implemented | CPU-tested | GPU |
|---|---|---|---|
| mixed scheduler, server | yes | yes (13 tests) | not run |
| KV block manager (refcount, LRU, prefix, stats) | yes | yes: refcount invariants, LRU order, eviction under pressure, prefix hit after free, randomized no-leak | not run |
| budget sweep, MFU/MBU | yes | formulas and command generation only | not run, no numbers |
| Llama TP | yes | TP=2 == TP=1 == HF `LlamaForCausalLM` (tiny random config, fp32, gloo) | not run; Llama-3.1-8B not loaded |
| OLMoE model, MoE layer, EP | yes | EP=2 == single rank == plain-loop reference; EP=2 == EP=1 == HF `OlmoeForCausalLM` (tiny config); dispatch/combine round trip; load-balance stats | not run; OLMoE-1B-7B weights not loaded |
| Triton paged decode kernel | yes | **not executed** (no GPU/Triton on CPU); a step-by-step torch emulation of both kernels matches the reference (page boundaries, GQA, split sizes, empty CUDA-graph rows) | not run (`scripts/run_2x4090.sh f`) |
| Triton grouped GEMM | yes | **not executed** (no GPU/Triton on CPU); tile bookkeeping tested through a torch emulation of the kernel's indexing | not run |
| EPEngine (DP attention + EP lockstep) | yes | not tested (needs CUDA + NCCL) | not run |

Known limits: CPU attention fallback covers prefill without prefix cache only; the paged-KV path (`flash_attn`) is GPU-only. Throughput, latency, MFU and MBU numbers: **not measured**. Results go to `bench/*.jsonl` once `scripts/run_2x4090.sh` has been run.

---

*Upstream README below.*

<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)