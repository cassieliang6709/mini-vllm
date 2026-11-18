#!/usr/bin/env bash
# Commands to run on the 2x RTX 4090 box. NOTHING here has been run yet; no numbers exist.
#
#   bash scripts/run_2x4090.sh all        # everything, in order
#   bash scripts/run_2x4090.sh a          # one stage: a b c d e
#
# Model dirs (download first, e.g. huggingface-cli download <repo> --local-dir <dir>):
#   Qwen/Qwen3-0.6B, meta-llama/Llama-3.1-8B (gated), allenai/OLMoE-1B-7B-0924
set -euo pipefail
cd "$(dirname "$0")/.."

QWEN=${QWEN:-$HOME/huggingface/Qwen3-0.6B}
LLAMA=${LLAMA:-$HOME/huggingface/Llama-3.1-8B}
OLMOE=${OLMOE:-$HOME/huggingface/OLMoE-1B-7B-0924}
mkdir -p bench/out
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

nvidia-smi --query-gpu=name,memory.total --format=csv   # confirm 2x RTX 4090 24 GB

stage_a() {  # (a) GPU correctness vs HF: greedy-token agreement. Llama-8B bf16 = 16 GB: fits one 4090 for HF; also run TP=2.
  python bench/check_vs_hf.py --model "$QWEN"
  python bench/check_vs_hf.py --model "$LLAMA" --tp 2
  # single-GPU Llama-8B would leave ~no KV room with 24 GB; lower the length if you want it:
  # python bench/check_vs_hf.py --model "$LLAMA" --tp 1
}

stage_b() {  # (b) max_num_batched_tokens sweep {512,2048,8192}: throughput, TTFT, ITL p50/p95, MFU/MBU -> bench/budget_sweep.jsonl
  python bench/budget_sweep.py --model "$QWEN" --budgets 512,2048,8192 --only mixed,shared
  python bench/budget_sweep.py --model "$QWEN" --budgets 512,2048,8192 --only legacy,shared   # upstream scheduler for contrast
}

stage_c() {  # (c) TP=2 Llama-3.1-8B (same workload as the 1-GPU Qwen runs; budget 2048). Output: bench/tp2_llama.jsonl
  python bench/ablation.py --model "$LLAMA" --tp 2 --only mixed,shared --max-num-batched-tokens 2048 --out bench/tp2_llama.jsonl
  python bench/ablation.py --model "$LLAMA" --tp 2 --only mixed,unique --max-num-batched-tokens 2048 --out bench/tp2_llama.jsonl
}

stage_d() {  # (d) OLMoE-1B-7B, EP=2 (+ EP=1 baseline, same code path). Output: bench/ep_results.jsonl
  python bench/ep_bench.py --model "$OLMOE" --ep 1
  python bench/ep_bench.py --model "$OLMOE" --ep 2
}

stage_e() {  # (e) official vLLM on the same configs. Check flags with `vllm bench throughput --help` for your vLLM version.
  # fixed-length synthetic workload close to ours: in 512..1536, out 64..512 -> use mean values
  IN=1024; OUT=256; N=200
  vllm bench throughput --model "$QWEN"  --input-len $IN --output-len $OUT --num-prompts $N --dtype bfloat16 \
      --max-num-batched-tokens 2048 --enable-prefix-caching --output-json bench/out/vllm_qwen.json
  vllm bench throughput --model "$LLAMA" --input-len $IN --output-len $OUT --num-prompts $N --dtype bfloat16 \
      --tensor-parallel-size 2 --max-num-batched-tokens 2048 --output-json bench/out/vllm_llama_tp2.json
  # OLMoE: vLLM's EP is enabled on top of its TP/DP layout. DP-attention + EP (matches our design):
  vllm bench throughput --model "$OLMOE" --input-len $IN --output-len $OUT --num-prompts $N --dtype bfloat16 \
      --data-parallel-size 2 --enable-expert-parallel --output-json bench/out/vllm_olmoe_ep2.json \
    || vllm bench throughput --model "$OLMOE" --input-len $IN --output-len $OUT --num-prompts $N --dtype bfloat16 \
      --tensor-parallel-size 2 --enable-expert-parallel --output-json bench/out/vllm_olmoe_tp2_ep.json
  # Latency (TTFT/ITL) under load: use `vllm serve ...` + `vllm bench serve --request-rate 8` to compare with ablation.py's Poisson arrivals.
}

case "${1:-all}" in
  a) stage_a ;; b) stage_b ;; c) stage_c ;; d) stage_d ;; e) stage_e ;;
  all) stage_a; stage_b; stage_c; stage_d; stage_e ;;
  *) echo "usage: $0 [a|b|c|d|e|all]"; exit 2 ;;
esac
