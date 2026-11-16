# Let CPU tests import nanovllm.engine.* without running nanovllm/__init__.py,
# which pulls in torch / flash-attn through LLM.
import pathlib, sys, types

pkg = types.ModuleType("nanovllm")
pkg.__path__ = [str(pathlib.Path(__file__).resolve().parents[1] / "nanovllm")]
sys.modules.setdefault("nanovllm", pkg)
