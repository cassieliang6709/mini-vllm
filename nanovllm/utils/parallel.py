"""Which ranks shard what.

Two modes, both use the whole default process group:

  tensor parallel (default): dense layers (attention, MLP, embedding, lm_head) are sharded
      across all ranks. tp_size = world size, ep_size = 1.
  expert parallel: dense layers are replicated, each rank runs attention on its own
      requests (data parallel); MoE experts are split across ranks. tp_size = 1, ep_size = world.

Layers ask this module instead of calling dist.get_world_size() themselves.
"""
import torch.distributed as dist

_mode = "tp"


def set_parallel_mode(mode: str):
    assert mode in ("tp", "ep")
    global _mode
    _mode = mode


def get_parallel_mode() -> str:
    return _mode


def _world():
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def _rank():
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def get_tp_size() -> int:
    return _world() if _mode == "tp" else 1


def get_tp_rank() -> int:
    return _rank() if _mode == "tp" else 0


def get_ep_size() -> int:
    return _world() if _mode == "ep" else 1


def get_ep_rank() -> int:
    return _rank() if _mode == "ep" else 0
