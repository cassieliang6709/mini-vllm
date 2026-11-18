import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "bench"))
import metrics as M


def llama8b():
    return M.ModelShape(layers=32, hidden=4096, n_heads=32, n_kv_heads=8, head_dim=128, vocab=128256, dense_ffn=14336)


def test_llama8b_param_count():
    # Llama-3.1-8B has 8.03B parameters (untied embedding + lm_head)
    assert abs(llama8b().total_params() / 1e9 - 8.03) < 0.01


def test_prefill_flops_matmul_dominated():
    m = llama8b()
    f = M.prefill_flops(m, [(0, 1)])
    assert abs(f - 2 * m.active_params() - 4 * 32 * 32 * 128) < 1


def test_decode_bytes_weights_plus_kv():
    m = llama8b()
    b = M.decode_bytes(m, [100, 200])
    kv = 300 * 2 * 32 * 8 * 128 * 2
    assert b == (m.layers * m.dense_params_per_layer + m.lm_head_params) * 2 + kv


def test_moe_expected_experts():
    assert M.expected_distinct_experts(64, 8, 1) == 8
    assert 63 < M.expected_distinct_experts(64, 8, 256) <= 64


def test_unknown_gpu_gives_none():
    assert M.peak_for("Some GPU") is None
    assert M.peak_for("NVIDIA GeForce RTX 4090") == (165.2e12, 1008e9)
