"""GPU correctness: greedy tokens from nanovllm vs HF transformers on the same prompts.

    python bench/check_vs_hf.py --model ~/huggingface/Qwen3-0.6B
    python bench/check_vs_hf.py --model ~/huggingface/Llama-3.1-8B --tp 2

Both run in bf16, so exact agreement over a long continuation is not guaranteed (a near-tie logit can
flip). We report, per prompt, the number of leading tokens that agree out of --new-tokens, and exit
non-zero if any prompt disagrees before --min-agree tokens.
nanovllm "greedy" = temperature 1e-4 (the sampler divides by temperature; tiny T makes argmax dominate).
"""
import argparse
import gc
import json
import os
import sys

PROMPTS = ["The capital of France is", "def fibonacci(n):", "In 1969, humans first", "Explain tensor parallelism in one sentence:"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--new-tokens", type=int, default=32)
    ap.add_argument("--min-agree", type=int, default=8)
    ap.add_argument("--decode-backend", default="flash_attn", choices=["flash_attn", "triton_paged"])
    ap.add_argument("--cuda-graph", action="store_true", help="capture decode CUDA graphs (default: eager)")
    args = ap.parse_args()
    path = os.path.expanduser(args.model)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    hf = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="auto" if args.tp > 1 else "cuda").eval()
    ref = []
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids.to(hf.device)
        out = hf.generate(ids, max_new_tokens=args.new_tokens, do_sample=False, min_new_tokens=args.new_tokens)
        ref.append(out[0, ids.size(1):].tolist())
    del hf
    gc.collect()
    torch.cuda.empty_cache()

    from nanovllm import LLM, SamplingParams
    llm = LLM(path, enforce_eager=not args.cuda_graph, tensor_parallel_size=args.tp, max_model_len=2048,
              decode_attention_backend=args.decode_backend)
    outs = llm.generate([tok.encode(p) for p in PROMPTS],
                        SamplingParams(temperature=1e-4, max_tokens=args.new_tokens, ignore_eos=True), use_tqdm=False)
    bad = 0
    for p, r, o in zip(PROMPTS, ref, outs):
        n = next((i for i, (a, b) in enumerate(zip(r, o["token_ids"])) if a != b), args.new_tokens)
        print(json.dumps(dict(prompt=p, agree_prefix=n, of=args.new_tokens)))
        bad += n < args.min_agree
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
