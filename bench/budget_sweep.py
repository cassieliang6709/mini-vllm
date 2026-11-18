"""Sweep max_num_batched_tokens and record throughput, TTFT, ITL p50/p95 (and MFU/MBU) to JSONL.

    python bench/budget_sweep.py --model ~/huggingface/Qwen3-0.6B
    python bench/budget_sweep.py --model ... --budgets 512,2048,8192 --only mixed,shared --tp 2
    python bench/budget_sweep.py --model ... --dry-run          # print the commands, run nothing

Each budget runs bench/ablation.py in a fresh subprocess (fresh KV cache / CUDA context).
Rows go to bench/budget_sweep.jsonl (override with --out). Extra args after `--` go to ablation.py.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def build_commands(model, budgets, only, tp, out, extra):
    cmds = []
    for b in budgets:
        cmd = [sys.executable, os.path.join(HERE, "ablation.py"), "--model", model,
               "--max-num-batched-tokens", str(b), "--only", only, "--tp", str(tp), "--out", out, *extra]
        cmds.append(cmd)
    return cmds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--budgets", default="512,2048,8192")
    ap.add_argument("--only", default="mixed,shared", help="scheduler,workload as in ablation.py")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(HERE, "budget_sweep.jsonl"))
    ap.add_argument("--dry-run", action="store_true")
    args, extra = ap.parse_known_args()
    extra = [e for e in extra if e != "--"]
    budgets = [int(x) for x in args.budgets.split(",")]
    for cmd in build_commands(args.model, budgets, args.only, args.tp, args.out, extra):
        print(" ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
