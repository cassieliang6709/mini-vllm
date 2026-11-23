"""Paged decode attention: one query token per sequence over a paged KV cache.

Reproduces the decode kernel of PagedAttention (vLLM, SOSP'23) in Triton, with two changes:
- GQA grouping: one program handles all query heads that share a KV head, so each KV page is read
  once per group instead of once per query head.
- Split-K over the context (PagedAttention V2 / Flash-Decoding): the context is cut into partitions
  that run in parallel; a second kernel merges the partial softmax results.

Layouts (same as flash_attn_with_kvcache in model_runner):
    q            [bs, Hq, D]
    k/v_cache    [num_blocks, block_size, Hkv, D]
    block_tables [bs, max_blocks] int32 (padding -1, never read because of context_lens)
    context_lens [bs] int32 (0 = padded CUDA-graph row; its output is 0)

CUDA graphs: the grid depends only on bs and block_tables.size(1), which are fixed per captured graph.
"""
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def paged_decode_reference(q, k_cache, v_cache, block_tables, context_lens, scale):
    """Plain PyTorch: gather each sequence's tokens from its pages, then softmax attention."""
    bs, Hq, D = q.shape
    block_size, Hkv = k_cache.shape[1], k_cache.shape[2]
    group = Hq // Hkv
    out = torch.zeros_like(q)
    for b in range(bs):
        n = int(context_lens[b])
        if n == 0:
            continue
        pages = block_tables[b, : (n + block_size - 1) // block_size].long()
        k = k_cache[pages].reshape(-1, Hkv, D)[:n].float()    # [n, Hkv, D]
        v = v_cache[pages].reshape(-1, Hkv, D)[:n].float()
        k = k.repeat_interleave(group, 1)    # [n, Hq, D]
        v = v.repeat_interleave(group, 1)
        s = torch.einsum("hd,nhd->hn", q[b].float(), k) * scale
        out[b] = torch.einsum("hn,nhd->hd", s.softmax(-1), v).to(q.dtype)
    return out


if HAS_TRITON:

    @triton.jit
    def _paged_decode_partition_kernel(
        q_ptr, k_ptr, v_ptr, bt_ptr, lens_ptr, m_ptr, l_ptr, acc_ptr,
        stride_qb, stride_qh,
        stride_kb, stride_ks, stride_kh,
        stride_vb, stride_vs, stride_vh,
        stride_btb,
        scale, partition_size, num_splits,
        HQ: tl.constexpr, GROUP: tl.constexpr, BLOCK_G: tl.constexpr,
        D: tl.constexpr, BLOCK_D: tl.constexpr,
        PAGE: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        b = tl.program_id(0)
        h = tl.program_id(1)    # kv head
        s = tl.program_id(2)    # partition
        ctx = tl.load(lens_ptr + b)
        start = s * partition_size
        end = tl.minimum(start + partition_size, ctx)

        offs_g = tl.arange(0, BLOCK_G)
        offs_d = tl.arange(0, BLOCK_D)
        g_mask = offs_g < GROUP
        d_mask = offs_d < D
        qh = h * GROUP + offs_g    # query heads of this group
        q = tl.load(q_ptr + b * stride_qb + qh[:, None] * stride_qh + offs_d[None, :],
                    mask=g_mask[:, None] & d_mask[None, :], other=0.0)

        m_i = tl.full([BLOCK_G], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_G], tl.float32)
        acc = tl.zeros([BLOCK_G, BLOCK_D], tl.float32)
        for n0 in range(start, end, BLOCK_N):
            # PAGE % BLOCK_N == 0 and partitions start on BLOCK_N boundaries: a tile never crosses a page
            page = tl.load(bt_ptr + b * stride_btb + n0 // PAGE)
            offs_n = n0 + tl.arange(0, BLOCK_N)
            n_mask = offs_n < end
            slot = offs_n % PAGE
            kv_mask = n_mask[:, None] & d_mask[None, :]
            k = tl.load(k_ptr + page * stride_kb + slot[:, None] * stride_ks + h * stride_kh + offs_d[None, :],
                        mask=kv_mask, other=0.0)
            scores = tl.dot(q, tl.trans(k)) * scale    # [BLOCK_G, BLOCK_N], fp32
            scores = tl.where(n_mask[None, :], scores, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(scores, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            v = tl.load(v_ptr + page * stride_vb + slot[:, None] * stride_vs + h * stride_vh + offs_d[None, :],
                        mask=kv_mask, other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new

        # partials, not normalised: m/l [bs, Hq, splits], acc [bs, Hq, splits, D]
        row = (b * HQ + qh) * num_splits + s
        tl.store(m_ptr + row, m_i, mask=g_mask)
        tl.store(l_ptr + row, l_i, mask=g_mask)
        tl.store(acc_ptr + row[:, None] * D + offs_d[None, :], acc, mask=g_mask[:, None] & d_mask[None, :])

    @triton.jit
    def _paged_decode_reduce_kernel(
        m_ptr, l_ptr, acc_ptr, out_ptr, stride_ob, stride_oh, num_splits,
        HQ: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_S: tl.constexpr,
    ):
        b = tl.program_id(0)
        h = tl.program_id(1)
        offs_s = tl.arange(0, BLOCK_S)
        offs_d = tl.arange(0, BLOCK_D)
        s_mask = offs_s < num_splits
        d_mask = offs_d < D
        row = (b * HQ + h) * num_splits + offs_s
        m = tl.load(m_ptr + row, mask=s_mask, other=float("-inf"))
        l = tl.load(l_ptr + row, mask=s_mask, other=0.0)
        m_max = tl.max(m, 0)
        m_max = tl.where(m_max == float("-inf"), 0.0, m_max)    # context_len 0: all partitions empty
        w = tl.exp(m - m_max)    # empty partition: exp(-inf) = 0
        acc = tl.load(acc_ptr + row[:, None] * D + offs_d[None, :],
                      mask=s_mask[:, None] & d_mask[None, :], other=0.0)
        num = tl.sum(acc * w[:, None], 0)
        den = tl.sum(l * w, 0)
        o = tl.where(den > 0, num / den, 0.0)
        tl.store(out_ptr + b * stride_ob + h * stride_oh + offs_d, o.to(out_ptr.dtype.element_ty), mask=d_mask)


def _next_pow2(x: int) -> int:
    return 1 << (max(int(x), 1) - 1).bit_length()


def paged_decode_attention(q, k_cache, v_cache, block_tables, context_lens, scale,
                           partition_size: int = 512, block_n: int = 64):
    """partition_size: tokens per split; 0 = no split (one partition covers the whole context)."""
    assert HAS_TRITON, "paged_decode_attention needs Triton (GPU)"
    bs, Hq, D = q.shape
    page, Hkv = k_cache.shape[1], k_cache.shape[2]
    assert Hq % Hkv == 0 and page % block_n == 0
    assert q.stride(-1) == 1 and k_cache.stride(-1) == 1 and v_cache.stride(-1) == 1
    group = Hq // Hkv
    max_tokens = block_tables.size(1) * page
    if partition_size <= 0:
        partition_size = max_tokens
    assert partition_size % block_n == 0
    num_splits = triton.cdiv(max_tokens, partition_size)

    m = torch.empty(bs, Hq, num_splits, dtype=torch.float32, device=q.device)
    l = torch.empty_like(m)
    acc = torch.empty(bs, Hq, num_splits, D, dtype=torch.float32, device=q.device)
    out = torch.empty_like(q)
    BLOCK_D = max(16, _next_pow2(D))
    _paged_decode_partition_kernel[(bs, Hkv, num_splits)](
        q, k_cache, v_cache, block_tables, context_lens, m, l, acc,
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        block_tables.stride(0),
        scale, partition_size, num_splits,
        HQ=Hq, GROUP=group, BLOCK_G=max(16, _next_pow2(group)),    # tl.dot needs >= 16 rows
        D=D, BLOCK_D=BLOCK_D, PAGE=page, BLOCK_N=block_n,
    )
    _paged_decode_reduce_kernel[(bs, Hq)](
        m, l, acc, out, out.stride(0), out.stride(1), num_splits,
        HQ=Hq, D=D, BLOCK_D=BLOCK_D, BLOCK_S=_next_pow2(num_splits),
    )
    return out
