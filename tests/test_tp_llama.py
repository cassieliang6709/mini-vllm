"""Tensor-parallel Llama on CPU (gloo, 2 processes), tiny random config.

  TP=2 logits == TP=1 logits == HF LlamaForCausalLM logits (fp32, allclose).
Needs torch + transformers; skipped otherwise.
"""
import _dist_helpers  # noqa: F401  (stub must come first)
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

SEQ_LENS = [5, 7]


def tiny_config():
    from transformers import LlamaConfig
    return LlamaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=8, num_key_value_heads=4, max_position_embeddings=512,
        rms_norm_eps=1e-5, tie_word_embeddings=False, hidden_act="silu",
        rope_parameters={"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                         "high_freq_factor": 4.0, "original_max_position_embeddings": 128,
                         "rope_theta": 10000.0},
    )


def make_inputs():
    g = torch.Generator().manual_seed(1)
    return [torch.randint(0, 128, (n,), generator=g) for n in SEQ_LENS]


def _run_model(rank, world, state_path, out_path):
    from nanovllm.models.llama import LlamaForCausalLM
    from nanovllm.layers.sampler import Sampler  # noqa: F401  (import check only)
    from nanovllm.utils.context import set_context, reset_context
    from nanovllm.utils.loader import load_weights

    cfg = tiny_config()
    model = LlamaForCausalLM(cfg)
    load_weights(model, torch.load(state_path).items())
    seqs = make_inputs()
    ids = torch.cat(seqs)
    pos = torch.cat([torch.arange(n) for n in SEQ_LENS])
    cu = torch.tensor([0, SEQ_LENS[0], sum(SEQ_LENS)], dtype=torch.int32)
    with torch.inference_mode():
        set_context(True, cu, cu, max(SEQ_LENS), max(SEQ_LENS))
        hidden = model(ids, pos)
        set_context(False)    # lm_head on every position (prefill ctx would keep only the last)
        logits = model.compute_logits(hidden)
        # also the engine's own path: last token of each sequence
        set_context(True, cu, cu, max(SEQ_LENS), max(SEQ_LENS))
        last = model.compute_logits(hidden)
    reset_context()
    if rank == 0:
        torch.save({"logits": logits, "last": last}, out_path)


@pytest.fixture(scope="module")
def hf_and_state(tmp_path_factory):
    from transformers import LlamaForCausalLM as HF
    torch.manual_seed(0)
    hf = HF(tiny_config()).eval().float()
    with torch.no_grad():    # make norms non-trivial
        for n, p in hf.named_parameters():
            if "norm" in n:
                p.add_(torch.randn_like(p) * 0.1)
    path = tmp_path_factory.mktemp("w") / "state.pt"
    torch.save(hf.state_dict(), path)
    return hf, str(path)


def hf_logits(hf):
    outs = []
    with torch.no_grad():
        for s in make_inputs():
            outs.append(hf(s[None]).logits[0])
    return torch.cat(outs)


@pytest.mark.parametrize("world", [1, 2])
def test_matches_hf(world, hf_and_state, tmp_path):
    hf, state = hf_and_state
    out = str(tmp_path / "out.pt")
    _dist_helpers.run_ranks(_run_model, world, state, out)
    got = torch.load(out)
    ref = hf_logits(hf)
    assert torch.allclose(got["logits"], ref, atol=2e-4, rtol=1e-4), (got["logits"] - ref).abs().max()
    last_idx = torch.tensor([SEQ_LENS[0] - 1, sum(SEQ_LENS) - 1])
    assert torch.allclose(got["last"], ref[last_idx], atol=2e-4, rtol=1e-4)


def test_tp2_equals_tp1(hf_and_state, tmp_path):
    _, state = hf_and_state
    o1, o2 = str(tmp_path / "o1.pt"), str(tmp_path / "o2.pt")
    _dist_helpers.run_ranks(_run_model, 1, state, o1)
    _dist_helpers.run_ranks(_run_model, 2, state, o2)
    a, b = torch.load(o1)["logits"], torch.load(o2)["logits"]
    assert torch.allclose(a, b, atol=1e-5, rtol=1e-5), (a - b).abs().max()
    assert not torch.allclose(a[0], a[1])    # sanity: logits are not degenerate


def test_llama3_rope_scaling_matches_hf():
    from nanovllm.layers.rotary_embedding import llama3_scale_inv_freq
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    cfg = tiny_config()
    ref, _ = ROPE_INIT_FUNCTIONS["llama3"](cfg, "cpu")
    head = cfg.hidden_size // cfg.num_attention_heads
    inv = 1.0 / (10000.0 ** (torch.arange(0, head, 2, dtype=torch.float) / head))
    got = llama3_scale_inv_freq(inv, 8.0, 1.0, 4.0, 128)
    assert torch.allclose(got, ref.float())
    assert not torch.allclose(got, inv)    # scaling actually changes something (incl. medium band)
