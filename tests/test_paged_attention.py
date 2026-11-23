"""Paged decode attention (nanovllm/layers/paged_attention.py).

CPU: the PyTorch reference against dense attention, and a step-by-step torch emulation of the Triton
kernels (same tiles, partitions, page lookup, online softmax and merge) against the reference.
GPU (skipped without CUDA + Triton): the real kernels against the reference and flash_attn, and under
CUDA-graph replay.
"""
import math

import pytest

torch = pytest.importorskip("torch")

from nanovllm.layers.paged_attention import HAS_TRITON, paged_decode_reference

GPU = torch.cuda.is_available() and HAS_TRITON


def make_case(bs, hq, hkv, d, page, lens, num_blocks=None, dtype=torch.float32, device="cpu", seed=0):
    """Random paged cache; each sequence gets distinct, shuffled pages (padding -1 like model_runner)."""
    g = torch.Generator().manual_seed(seed)
    need = [(n + page - 1) // page for n in lens]
    max_blocks = max(max(need), 1)
    num_blocks = num_blocks or sum(need) + 3
    perm = torch.randperm(num_blocks, generator=g).tolist()
    bt = torch.full((bs, max_blocks), -1, dtype=torch.int32)
    i = 0
    for b, k in enumerate(need):
        bt[b, :k] = torch.tensor(perm[i:i + k], dtype=torch.int32)
        i += k
    q = torch.randn(bs, hq, d, generator=g).to(dtype)
    k_cache = torch.randn(num_blocks, page, hkv, d, generator=g).to(dtype)
    v_cache = torch.randn(num_blocks, page, hkv, d, generator=g).to(dtype)
    lens_t = torch.tensor(lens, dtype=torch.int32)
    return [t.to(device) for t in (q, k_cache, v_cache, bt, lens_t)]


def dense_attention(q, k_cache, v_cache, bt, lens, scale):
    """Independent check of the reference: build each sequence's contiguous K/V token by token."""
    bs, hq, d = q.shape
    page, hkv = k_cache.shape[1], k_cache.shape[2]
    out = torch.zeros_like(q)
    for b in range(bs):
        n = int(lens[b])
        if n == 0:
            continue
        idx = [(int(bt[b, t // page]), t % page) for t in range(n)]
        k = torch.stack([k_cache[p, s] for p, s in idx])    # [n, hkv, d]
        v = torch.stack([v_cache[p, s] for p, s in idx])
        for h in range(hq):
            kh = h // (hq // hkv)
            w = torch.softmax(k[:, kh] @ q[b, h] * scale, 0)
            out[b, h] = w @ v[:, kh]
    return out


def emulate_kernels(q, k_cache, v_cache, bt, lens, scale, partition_size, block_n):
    """Mirror of _paged_decode_partition_kernel + _paged_decode_reduce_kernel, one program at a time."""
    bs, hq, d = q.shape
    page, hkv = k_cache.shape[1], k_cache.shape[2]
    group = hq // hkv
    max_tokens = bt.size(1) * page
    if partition_size <= 0:
        partition_size = max_tokens
    splits = math.ceil(max_tokens / partition_size)
    m = torch.empty(bs, hq, splits)
    l = torch.empty(bs, hq, splits)
    acc = torch.empty(bs, hq, splits, d)
    for b in range(bs):
        ctx = int(lens[b])
        for h in range(hkv):
            qg = q[b, h * group:(h + 1) * group].float()    # [G, d]
            for s in range(splits):
                start, end = s * partition_size, min(s * partition_size + partition_size, ctx)
                m_i = torch.full((group,), float("-inf"))
                l_i = torch.zeros(group)
                a = torch.zeros(group, d)
                for n0 in range(start, end, block_n):
                    p_id = int(bt[b, n0 // page])
                    offs = torch.arange(n0, n0 + block_n)
                    mask = offs < end
                    slot = offs % page
                    k = torch.where(mask[:, None], k_cache[p_id, slot, h].float(), 0.0)
                    v = torch.where(mask[:, None], v_cache[p_id, slot, h].float(), 0.0)
                    sc = torch.where(mask[None, :], qg @ k.T * scale, float("-inf"))
                    m_new = torch.maximum(m_i, sc.max(1).values)
                    alpha = torch.exp(m_i - m_new)
                    p = torch.exp(sc - m_new[:, None])
                    l_i = l_i * alpha + p.sum(1)
                    a = a * alpha[:, None] + p @ v
                    m_i = m_new
                m[b, h * group:(h + 1) * group, s] = m_i
                l[b, h * group:(h + 1) * group, s] = l_i
                acc[b, h * group:(h + 1) * group, s] = a
    m_max = m.max(-1).values
    m_max = torch.where(m_max == float("-inf"), 0.0, m_max)
    w = torch.exp(m - m_max[..., None])
    num = (acc * w[..., None]).sum(2)
    den = (l * w).sum(2)
    return torch.where(den[..., None] > 0, num / den.clamp_min(1e-30)[..., None], 0.0).to(q.dtype)


# page boundaries (31/32/33 with page 32), a single token, an empty CUDA-graph row (0)
LENS = [1, 31, 32, 33, 100, 0, 64]


@pytest.mark.parametrize("hq,hkv", [(4, 4), (8, 2), (8, 1)])
def test_reference_matches_dense(hq, hkv):
    q, kc, vc, bt, lens = make_case(len(LENS), hq, hkv, 16, 32, LENS)
    scale = 16 ** -0.5
    torch.testing.assert_close(paged_decode_reference(q, kc, vc, bt, lens, scale),
                               dense_attention(q, kc, vc, bt, lens, scale), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("partition_size", [0, 16, 32, 64])
@pytest.mark.parametrize("hq,hkv", [(4, 4), (8, 2)])
def test_kernel_emulation_matches_reference(partition_size, hq, hkv):
    q, kc, vc, bt, lens = make_case(len(LENS), hq, hkv, 16, 32, LENS, seed=1)
    scale = 16 ** -0.5
    out = emulate_kernels(q, kc, vc, bt, lens, scale, partition_size, block_n=16)
    torch.testing.assert_close(out, paged_decode_reference(q, kc, vc, bt, lens, scale), atol=1e-5, rtol=1e-5)


def test_empty_row_is_zero_not_nan():
    q, kc, vc, bt, lens = make_case(2, 4, 2, 16, 32, [0, 5])
    out = emulate_kernels(q, kc, vc, bt, lens, 0.25, partition_size=16, block_n=16)
    assert torch.isfinite(out).all() and out[0].abs().max() == 0


# ---------------------------------------------------------------- GPU

gpu = pytest.mark.skipif(not GPU, reason="needs CUDA + Triton")
GPU_LENS = [1, 255, 256, 257, 1000, 0, 4096, 3000]


@gpu
@pytest.mark.parametrize("partition_size", [0, 256, 512])
@pytest.mark.parametrize("hq,hkv,d", [(16, 8, 128), (32, 8, 128), (14, 2, 64), (8, 8, 64)])
def test_triton_matches_reference(partition_size, hq, hkv, d):
    from nanovllm.layers.paged_attention import paged_decode_attention
    q, kc, vc, bt, lens = make_case(len(GPU_LENS), hq, hkv, d, 256, GPU_LENS, dtype=torch.bfloat16, device="cuda")
    scale = d ** -0.5
    out = paged_decode_attention(q, kc, vc, bt, lens, scale, partition_size=partition_size)
    torch.testing.assert_close(out.float(), paged_decode_reference(q, kc, vc, bt, lens, scale).float(),
                               atol=2e-2, rtol=2e-2)


@gpu
def test_triton_matches_flash_attn():
    flash_attn = pytest.importorskip("flash_attn")
    from nanovllm.layers.paged_attention import paged_decode_attention
    q, kc, vc, bt, lens = make_case(len(GPU_LENS), 32, 8, 128, 256, GPU_LENS, dtype=torch.bfloat16, device="cuda")
    keep = lens > 0    # flash_attn's output for a 0-length row is not defined the same way
    scale = 128 ** -0.5
    ours = paged_decode_attention(q, kc, vc, bt, lens, scale)
    ref = flash_attn.flash_attn_with_kvcache(q.unsqueeze(1), kc, vc, cache_seqlens=lens, block_table=bt,
                                             softmax_scale=scale, causal=True).squeeze(1)
    torch.testing.assert_close(ours[keep].float(), ref[keep].float(), atol=2e-2, rtol=2e-2)


@gpu
def test_triton_under_cuda_graph():
    """Capture with one set of lengths, replay with others (what model_runner does)."""
    from nanovllm.layers.paged_attention import paged_decode_attention
    q, kc, vc, bt, lens = make_case(4, 16, 4, 128, 256, [4096, 4096, 4096, 4096], dtype=torch.bfloat16, device="cuda")
    scale = 128 ** -0.5
    paged_decode_attention(q, kc, vc, bt, lens, scale)    # compile outside capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = paged_decode_attention(q, kc, vc, bt, lens, scale)
    lens.copy_(torch.tensor([1, 300, 0, 2049], dtype=torch.int32, device="cuda"))
    q.copy_(torch.randn_like(q))
    graph.replay()
    torch.testing.assert_close(out.float(), paged_decode_reference(q, kc, vc, bt, lens, scale).float(),
                               atol=2e-2, rtol=2e-2)
