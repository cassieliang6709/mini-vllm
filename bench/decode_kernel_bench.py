"""Decode attention microbenchmark: flash_attn_with_kvcache vs our Triton paged kernel.

    python bench/decode_kernel_bench.py                       # -> bench/out/decode_kernel.jsonl

Per (shape, bs, context, backend): median latency, and effective bandwidth = K+V bytes read / time,
compared with the RTX 4090 peak (1008 GB/s). Every sequence in a batch has the same context length.
Correctness is checked against the PyTorch reference before timing.
"""
import argparse
import json
import os

import torch
import triton.testing

from nanovllm.layers.paged_attention import paged_decode_attention, paged_decode_reference

SHAPES = {"qwen3-0.6b": (16, 8, 128), "llama-3.1-8b": (32, 8, 128)}    # (Hq, Hkv, D)
PEAK_GBPS = 1008.0


def make(bs, ctx, hq, hkv, d, page=256):
    blocks = (ctx + page - 1) // page
    bt = torch.randperm(bs * blocks, device="cuda", dtype=torch.int32).view(bs, blocks)
    q = torch.randn(bs, hq, d, device="cuda", dtype=torch.bfloat16)
    kc = torch.randn(bs * blocks, page, hkv, d, device="cuda", dtype=torch.bfloat16)
    vc = torch.randn_like(kc)
    lens = torch.full((bs,), ctx, device="cuda", dtype=torch.int32)
    return q, kc, vc, bt, lens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", default="1,8,32,128")
    ap.add_argument("--ctx", default="512,2048,8192")
    ap.add_argument("--partitions", default="0,512,1024", help="Triton split size in tokens; 0 = no split")
    ap.add_argument("--out", default="bench/out/decode_kernel.jsonl")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    try:
        from flash_attn import flash_attn_with_kvcache
    except ImportError:    # still measure our kernel on its own (bandwidth vs peak)
        flash_attn_with_kvcache = None
        print("flash_attn not installed: skipping the flash_attn baseline")

    with open(args.out, "a") as f:
        for name, (hq, hkv, d) in SHAPES.items():
            scale = d ** -0.5
            for bs in map(int, args.bs.split(",")):
                for ctx in map(int, args.ctx.split(",")):
                    q, kc, vc, bt, lens = make(bs, ctx, hq, hkv, d)
                    ref = paged_decode_reference(q, kc, vc, bt, lens, scale).float()
                    kv_bytes = 2 * bs * ctx * hkv * d * 2
                    runs = {}
                    if flash_attn_with_kvcache is not None:
                        runs["flash_attn"] = lambda: flash_attn_with_kvcache(
                            q.unsqueeze(1), kc, vc, cache_seqlens=lens, block_table=bt, softmax_scale=scale, causal=True)
                    for p in map(int, args.partitions.split(",")):
                        runs[f"triton_paged_p{p}"] = (lambda p=p: paged_decode_attention(
                            q, kc, vc, bt, lens, scale, partition_size=p))
                    for backend, fn in runs.items():
                        err = (fn().reshape(ref.shape).float() - ref).abs().max().item()
                        ms = triton.testing.do_bench(fn, warmup=25, rep=100)
                        gbps = kv_bytes / ms / 1e6
                        row = dict(shape=name, bs=bs, ctx=ctx, backend=backend, ms=round(ms, 4),
                                   gbps=round(gbps, 1), mbu=round(gbps / PEAK_GBPS, 3), max_abs_err=err)
                        print(json.dumps(row))
                        f.write(json.dumps(row) + "\n")
                    del q, kc, vc, bt, lens, ref
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
