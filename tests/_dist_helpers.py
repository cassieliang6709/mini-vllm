"""Shared helpers for multi-process gloo tests. Import this module before any `nanovllm` import."""
import pathlib
import socket
import sys
import types

if "nanovllm" not in sys.modules:    # same stub as conftest.py; spawned children don't run conftest
    pkg = types.ModuleType("nanovllm")
    pkg.__path__ = [str(pathlib.Path(__file__).resolve().parents[1] / "nanovllm")]
    sys.modules["nanovllm"] = pkg


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _entry(rank, world, port, fn, args):
    import torch
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    try:
        fn(rank, world, *args)
    finally:
        dist.destroy_process_group()


def run_ranks(fn, world, *args):
    """Run fn(rank, world, *args) on `world` gloo processes; raises if any rank fails."""
    import torch.multiprocessing as mp
    mp.spawn(_entry, args=(world, free_port(), fn, args), nprocs=world, join=True)
