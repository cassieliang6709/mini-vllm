# mini-vLLM

Fork of [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) @ `bb823b3` with a mixed-batch scheduler, an OpenAI-compatible streaming server, CPU tests and an ablation benchmark.

## Changes vs upstream

| | upstream | mini-vLLM |
|---|---|---|
| one step | all-prefill **or** all-decode | decode tokens **and** prefill chunks together |
| chunked prefill | only the head of the waiting queue | any number of waiting requests, each chunkable |
| prefill/decode state | `is_prefill` flag set by the scheduler | derived from `num_cached_tokens` vs `num_tokens` (`Sequence.is_prefill`) |
| serving | offline `generate()` | `server.py`: `/v1/chat/completions`, `/v1/completions`, SSE streaming |

Scheduling (`nanovllm/engine/scheduler.py`, `enable_mixed_batch=True`):
1. Every running request gets 1 decode token (preempt from the tail if out of KV blocks).
2. Leftover `max_num_batched_tokens` budget is filled FCFS with prefill chunks.
3. A step with any prefill chunk runs the varlen FlashAttention path (a decode is a `q_len=1` chunk over the paged cache); a pure-decode step keeps the CUDA-graph path.

`enable_mixed_batch=False` restores the upstream policy for A/B comparison.

## Run

```bash
pip install -e . fastapi uvicorn
python server.py --model ~/huggingface/Qwen3-0.6B
python bench/ablation.py --model ~/huggingface/Qwen3-0.6B     # mixed/legacy x shared/unique prefix
```

CPU tests (no GPU, no torch):

```bash
pip install xxhash numpy pytest fastapi httpx uvicorn
pytest tests
```

## Status

- CPU: scheduler / KV-block invariants, chunking, prefix reuse, preemption, randomized workloads, server streaming — 13 tests pass.
- GPU end-to-end correctness and benchmark numbers: **not run yet**. Results go to `bench/results.jsonl`.

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