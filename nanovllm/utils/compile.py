import os
import torch


def maybe_compile(fn):
    """torch.compile on GPU; plain eager on CPU (CPU tests, no C++ toolchain needed)."""
    if torch.cuda.is_available() and not os.environ.get("NANOVLLM_NO_COMPILE"):
        return torch.compile(fn)
    return fn
