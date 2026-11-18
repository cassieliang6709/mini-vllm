"""OLMoE expert-parallel benchmark: EP=N ranks, data-parallel attention, one process per GPU.

    python bench/ep_bench.py --model ~/huggingface/OLMoE-1B-7B-0924 --ep 2
    python bench/ep_bench.py --model ... --ep 1          # single-GPU baseline, same code path

Synthetic requests (random token ids, fixed seed) are split round-robin over ranks, all submitted
at t=0 (offline throughput). Rows append to bench/ep_results.jsonl. Needs a GPU + NCCL: not run yet.
Records: output tok/s, TTFT/ITL p50/p95, MFU (prefill-only steps), MBU (decode-only steps, per-GPU view),
and MoE load balance (tokens per expert accumulated over the run; max/mean averaged over layers).
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))] * 1000 if xs else float("nan")


def worker(rank, args):
    import torch
    import torch.distributed as dist
    from nanovllm.engine.ep_engine import EPEngine
    from nanovllm.layers.moe import ExpertParallelMoE
    from nanovllm.sampling_params import SamplingParams

    eng = EPEngine(os.path.expanduser(args.model), rank, args.ep, max_num_batched_tokens=args.max_num_batched_tokens,
                   max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs)
    hf = eng.config.hf_config
    sm = M.StepMetrics(M.ModelShape.from_hf(hf, hf.dtype.itemsize), torch.cuda.get_device_name(), 1, 1, args.ep)

    eng.generate([[1, 2, 3]], SamplingParams(max_tokens=4))    # warmup (all ranks take part)
    moes = [m for m in eng.runner.model.modules() if isinstance(m, ExpertParallelMoE)]
    for m in moes:
        m.collect_stats, m.cum_counts = True, None

    rng = random.Random(args.seed)
    reqs = [([rng.randint(0, 10000) for _ in range(rng.randint(args.min_in, args.max_in))],
             rng.randint(args.min_out, args.max_out)) for _ in range(args.num_requests)]
    seqs = [eng.add_request(p, SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=o))
            for i, (p, o) in enumerate(reqs) if i % args.ep == rank]
    dist.barrier()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for s in seqs:
        s.arrival_time = t0
    while True:
        t = time.perf_counter()
        more = eng.step()
        if not more:
            break
        sm.record(eng.last_step_info, time.perf_counter() - t)
    elapsed = time.perf_counter() - t0

    ratios = []
    for m in moes:    # per-layer max/mean of tokens per expert over the whole run, this rank's tokens
        if m.cum_counts is not None:
            c = m.cum_counts.float()
            ratios.append(float(c.max() / c.mean().clamp_min(1e-9)))
    mine = dict(rank=rank, out_tokens=sum(s.num_completion_tokens for s in seqs),
                ttft=[s.token_times[0] - s.arrival_time for s in seqs],
                itl=[b - a for s in seqs for a, b in zip(s.token_times, s.token_times[1:])],
                moe_imbalance_mean=sum(ratios) / len(ratios) if ratios else None,
                cache=eng.cache_stats(), summary=sm.summary(),
                peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30)
    gathered = [None] * args.ep
    dist.all_gather_object(gathered, mine)
    if rank == 0:
        ttft = [x for g in gathered for x in g["ttft"]]
        itl = [x for g in gathered for x in g["itl"]]
        mfus = [g["summary"].get("mfu_prefill") for g in gathered if g["summary"].get("mfu_prefill") is not None]
        mbus = [g["summary"].get("mbu_decode") for g in gathered if g["summary"].get("mbu_decode") is not None]
        res = dict(model=os.path.basename(args.model.rstrip("/")), ep=args.ep, requests=args.num_requests,
                   max_num_batched_tokens=args.max_num_batched_tokens, elapsed_s=round(elapsed, 2),
                   output_tok_per_s=round(sum(g["out_tokens"] for g in gathered) / elapsed, 1),
                   ttft_p50_ms=round(pct(ttft, 50), 1), ttft_p95_ms=round(pct(ttft, 95), 1),
                   itl_p50_ms=round(pct(itl, 50), 2), itl_p95_ms=round(pct(itl, 95), 2),
                   mfu_prefill=round(sum(mfus) / len(mfus), 4) if mfus else None,
                   mbu_decode=round(sum(mbus) / len(mbus), 4) if mbus else None,
                   moe_imbalance_mean=[g["moe_imbalance_mean"] for g in gathered],
                   cache_hit_rate=[round(g["cache"]["hit_rate"], 4) for g in gathered],
                   peak_mem_gb=[round(g["peak_mem_gb"], 2) for g in gathered],
                   gpu=torch.cuda.get_device_name())
        print("RESULT " + json.dumps(res), flush=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(res) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ep", type=int, default=2)
    ap.add_argument("--num-requests", type=int, default=128)
    ap.add_argument("--min-in", type=int, default=64)
    ap.add_argument("--max-in", type=int, default=1024)
    ap.add_argument("--min-out", type=int, default=64)
    ap.add_argument("--max-out", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-num-batched-tokens", type=int, default=4096)
    ap.add_argument("--max-num-seqs", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ep_results.jsonl"))
    args = ap.parse_args()
    import torch.multiprocessing as mp
    mp.spawn(worker, args=(args,), nprocs=args.ep)


if __name__ == "__main__":
    main()
