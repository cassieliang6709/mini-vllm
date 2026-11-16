"""Ablation: scheduler (mixed vs legacy) x workload (shared prefix vs unique prompts).

    python bench/ablation.py --model ~/huggingface/Qwen3-0.6B            # all 4 configs
    python bench/ablation.py --model ... --only mixed,shared             # one config

Each config runs in its own subprocess (fresh CUDA context and KV cache).
Requests arrive over time (Poisson, --rate req/s), so long prefills land
while other requests are decoding -- the case mixed batching targets.

Metrics: output throughput (tok/s), TTFT p50/p99, ITL p50/p99, peak GPU memory.
Peak memory is dominated by the pre-allocated KV cache (gpu_memory_utilization),
so it mostly confirms both schedulers use the same budget.
Results append to bench/results.jsonl.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time

CONFIGS = [(s, w) for s in ("mixed", "legacy") for w in ("shared", "unique")]


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))] * 1000 if xs else float("nan")


def make_workload(kind, n, rng, prefix_len, min_in, max_in, min_out, max_out):
    shared = [rng.randint(0, 10000) for _ in range(prefix_len)]
    reqs = []
    for _ in range(n):
        tail = [rng.randint(0, 10000) for _ in range(rng.randint(min_in, max_in))]
        prompt = (shared + tail) if kind == "shared" else [rng.randint(0, 10000) for _ in range(prefix_len)] + tail
        reqs.append((prompt, rng.randint(min_out, max_out)))
    return reqs


def run_one(args, sched, work):
    import torch
    from nanovllm import LLM, SamplingParams

    llm = LLM(os.path.expanduser(args.model), enforce_eager=args.enforce_eager, max_model_len=args.max_model_len,
              max_num_batched_tokens=args.max_num_batched_tokens, enable_mixed_batch=(sched == "mixed"))
    rng = random.Random(args.seed)
    reqs = make_workload(work, args.num_requests, rng, args.prefix_len, args.min_in, args.max_in,
                         args.min_out, args.max_out)
    gaps = [rng.expovariate(args.rate) for _ in reqs]

    llm.generate([[1, 2, 3]], SamplingParams(max_tokens=4), use_tqdm=False)    # warmup
    torch.cuda.reset_peak_memory_stats()

    seqs, i = [], 0
    t0 = time.perf_counter()
    next_arrival = t0
    while i < len(reqs) or not llm.is_finished():
        now = time.perf_counter()
        while i < len(reqs) and now >= next_arrival:
            prompt, out = reqs[i]
            seq = llm.add_request(prompt, SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=out))
            seq.arrival_time = next_arrival
            seqs.append(seq)
            next_arrival += gaps[i]
            i += 1
        if llm.is_finished():
            time.sleep(max(0.0, next_arrival - time.perf_counter()))
            continue
        llm.step()
    elapsed = time.perf_counter() - t0

    ttft = [s.token_times[0] - s.arrival_time for s in seqs]
    itl = [b - a for s in seqs for a, b in zip(s.token_times, s.token_times[1:])]
    out_tokens = sum(s.num_completion_tokens for s in seqs)
    res = dict(scheduler=sched, workload=work, requests=len(seqs), rate=args.rate,
               elapsed_s=round(elapsed, 2), output_tok_per_s=round(out_tokens / elapsed, 1),
               ttft_p50_ms=round(pct(ttft, 50), 1), ttft_p99_ms=round(pct(ttft, 99), 1),
               itl_p50_ms=round(pct(itl, 50), 2), itl_p99_ms=round(pct(itl, 99), 2),
               peak_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
               gpu=torch.cuda.get_device_name(), model=os.path.basename(args.model.rstrip("/")))
    print("RESULT " + json.dumps(res), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--only", default="", help="e.g. mixed,shared")
    ap.add_argument("--num-requests", type=int, default=200)
    ap.add_argument("--rate", type=float, default=8.0, help="arrivals per second")
    ap.add_argument("--prefix-len", type=int, default=512)
    ap.add_argument("--min-in", type=int, default=64)
    ap.add_argument("--max-in", type=int, default=1536)
    ap.add_argument("--min-out", type=int, default=64)
    ap.add_argument("--max-out", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-num-batched-tokens", type=int, default=2048)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--_child", default="")
    args = ap.parse_args()

    if args._child:
        run_one(args, *args._child.split(","))
        return

    configs = [tuple(args.only.split(","))] if args.only else CONFIGS
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.jsonl")
    rows = []
    for sched, work in configs:
        print(f"== {sched} / {work}", flush=True)
        cmd = [sys.executable, os.path.abspath(__file__), *sys.argv[1:], "--_child", f"{sched},{work}"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT ")), None)
        if line is None:
            print(proc.stdout[-2000:], proc.stderr[-4000:])
            continue
        row = json.loads(line[7:])
        rows.append(row)
        with open(out_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    keys = ["scheduler", "workload", "output_tok_per_s", "ttft_p50_ms", "ttft_p99_ms", "itl_p50_ms", "itl_p99_ms", "peak_mem_gb"]
    print("\n| " + " | ".join(keys) + " |\n|" + "---|" * len(keys))
    for r in rows:
        print("| " + " | ".join(str(r[k]) for k in keys) + " |")


if __name__ == "__main__":
    main()
