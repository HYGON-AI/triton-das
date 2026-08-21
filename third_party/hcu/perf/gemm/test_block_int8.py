# Copyright 2023-2024 SGLang Team
# SPDX-License-Identifier: Apache-2.0
# Modified by Hygon Information Technology Co., Ltd., 2026.
# Adapted from https://github.com/sgl-project/sglang/pull/3730

import itertools
import unittest

import torch
import triton
import triton.language as tl
import pytest

#from vllm.model_executor.layers.activation import SiluAndMul
#from vllm.model_executor.layers.fused_moe import fused_moe
from int8_utils import per_token_group_quant_int8, w8a8_block_int8_matmul

#torch.cuda.set_device(1)

# For test
def native_per_token_group_quant_int8(x, group_size, eps=1e-10, dtype=torch.int8):
    """per-token-group quantization on an input tensor `x` using native torch.

    It converts the tensor values into float8 values and returns the
    quantized tensor along with the scaling factor used for quantization.
    Note that only `torch.float8_e4m3fn` is supported for now.
    """
    assert (
        x.shape[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    iinfo = torch.iinfo(dtype)
    int8_min = iinfo.min
    int8_max = iinfo.max

    x_ = x.reshape(x.numel() // group_size, group_size)
    amax = x_.abs().max(dim=-1, keepdim=True)[0].clamp(min=eps).to(torch.float32)
    x_s = amax / int8_max
    x_q = (x_ / x_s).clamp(min=int8_min, max=int8_max).to(dtype)
    x_q = x_q.reshape(x.shape)
    x_s = x_s.reshape(x.shape[:-1] + (x.shape[-1] // group_size,))

    return x_q, x_s


# For test
def native_w8a8_block_int8_matmul(A, B, As, Bs, block_size, output_dtype=torch.float16):
    """matrix multiplication with block-wise quantization using native torch.

    It takes two input tensors `A` and `B` with scales `As` and `Bs`.
    The output is returned in the specified `output_dtype`.
    """

    A = A.to(torch.float32)
    B = B.to(torch.float32)
    assert A.shape[-1] == B.shape[-1]
    assert B.ndim == 2 and B.is_contiguous() and Bs.ndim == 2
    assert len(block_size) == 2
    block_n, block_k = block_size[0], block_size[1]
    assert (A.shape[-1] + block_k - 1) // block_k == As.shape[-1]
    assert A.shape[:-1] == As.shape[:-1]

    M = A.numel() // A.shape[-1]
    N, K = B.shape
    origin_C_shape = A.shape[:-1] + (N,)
    A = A.reshape(M, A.shape[-1])
    As = As.reshape(M, As.shape[-1])
    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k
    assert n_tiles == Bs.shape[0]
    assert k_tiles == Bs.shape[1]

    C_shape = (M, N)
    C = torch.zeros(C_shape, dtype=torch.float32, device=A.device)

    A_tiles = [A[:, i * block_k : min((i + 1) * block_k, K)] for i in range(k_tiles)]
    B_tiles = [
        [
            B[
                j * block_n : min((j + 1) * block_n, N),
                i * block_k : min((i + 1) * block_k, K),
            ]
            for i in range(k_tiles)
        ]
        for j in range(n_tiles)
    ]
    C_tiles = [C[:, j * block_n : min((j + 1) * block_n, N)] for j in range(n_tiles)]
    As_tiles = [As[:, i : i + 1] for i in range(k_tiles)]

    for i in range(k_tiles):
        for j in range(n_tiles):
            a = A_tiles[i]
            b = B_tiles[j][i]
            c = C_tiles[j]
            s = As_tiles[i] * Bs[j][i]
            c[:, :] += torch.matmul(a, b.t()) * s

    C = C.reshape(origin_C_shape).to(output_dtype)
    return C


# For test
def torch_w8a8_block_int8_moe(a, w1, w2, w1_s, w2_s, score, topk, block_shape):
    """fused moe with block-wise quantization using native torch."""

    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)

    _, block_k = block_shape[0], block_shape[1]
    a_q, a_s = native_per_token_group_quant_int8(a, block_k)
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            inter_out = native_w8a8_block_int8_matmul(
                a_q[mask], w1[i], a_s[mask], w1_s[i], block_shape, output_dtype=a.dtype
            )
            act_out = SiluAndMul().forward_native(inter_out)
            act_out_q, act_out_s = native_per_token_group_quant_int8(act_out, block_k)
            act_out = act_out.to(torch.float32)
            out[mask] = native_w8a8_block_int8_matmul(
                act_out_q, w2[i], act_out_s, w2_s[i], block_shape, output_dtype=a.dtype
            )
    return (
        out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)
    ).sum(dim=1)

'''
class TestW8A8BlockINT8FusedMoE(unittest.TestCase):
    DTYPES = [torch.half, torch.bfloat16]
    M = [1, 33, 64, 222]
    N = [128, 1024]
    K = [256, 4096]
    E = [8, 24]
    TOP_KS = [2, 6]
    # BLOCK_SIZE = [[64, 64], [64, 128], [128, 64], [128, 128]]
    BLOCK_SIZE = [[128, 128]]
    SEEDS = [0]

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _w8a8_block_int8_fused_moe(self, M, N, K, E, topk, block_size, dtype, seed):
        torch.manual_seed(seed)
        # NOTE(HandH1998): to avoid overflow when out_dtype = torch.half
        factor_for_scale = 1e-2
        int8_info = torch.iinfo(torch.int8)
        int8_max, int8_min = int8_info.max, int8_info.min

        a = torch.randn((M, K), dtype=dtype) / 10

        w1_fp32 = (torch.rand((E, 2 * N, K), dtype=torch.float32) - 0.5) * 2 * int8_max
        w1 = w1_fp32.clamp(min=int8_min, max=int8_max).to(torch.int8)

        w2_fp32 = (torch.rand((E, K, N), dtype=torch.float32) - 0.5) * 2 * int8_max
        w2 = w2_fp32.clamp(min=int8_min, max=int8_max).to(torch.int8)

        block_n, block_k = block_size[0], block_size[1]
        n_tiles_w1 = (2 * N + block_n - 1) // block_n
        n_tiles_w2 = (K + block_n - 1) // block_n
        k_tiles_w1 = (K + block_k - 1) // block_k
        k_tiles_w2 = (N + block_k - 1) // block_k

        w1_s = (
            torch.rand((E, n_tiles_w1, k_tiles_w1), dtype=torch.float32)
            * factor_for_scale
        )
        w2_s = (
            torch.rand((E, n_tiles_w2, k_tiles_w2), dtype=torch.float32)
            * factor_for_scale
        )

        score = torch.randn((M, E), dtype=dtype)

        with torch.inference_mode():
            out = fused_moe(
                a,
                w1,
                w2,
                score,
                topk,
                renormalize=False,
                use_int8_w8a8=True,
                global_num_experts=E,
                w1_scale=w1_s,
                w2_scale=w2_s,
                block_shape=block_size,
            )
            ref_out = torch_w8a8_block_int8_moe(
                a, w1, w2, w1_s, w2_s, score, topk, block_size
            )

        self.assertTrue(
            torch.mean(torch.abs(out.to(torch.float32) - ref_out.to(torch.float32)))
            / torch.mean(torch.abs(ref_out.to(torch.float32)))
            < 0.02
        )

    def test_w8a8_block_int8_fused_moe(self):
        for params in itertools.product(
            self.M,
            self.N,
            self.K,
            self.E,
            self.TOP_KS,
            self.BLOCK_SIZE,
            self.DTYPES,
            self.SEEDS,
        ):
            with self.subTest(
                M=params[0],
                N=params[1],
                K=params[2],
                E=params[3],
                topk=params[4],
                block_size=params[5],
                dtype=params[6],
                seed=params[7],
            ):
                self._w8a8_block_int8_fused_moe(*params)
'''


# Test cases for per_token_group_quant_int8
QUANT_TEST_CASES = [
    # rows, cols, group_size  # frequency
    (128, 7168, 128),  # 107326
    (128, 128, 128),  # 25812
    (1024, 128, 128),  # 25697
    (128, 1152, 128),  # 1399
    (4096, 7168, 128),  # 241
    (16384, 7168, 128),  # 241
    (512, 512, 128),  # 61
    (1024, 1024, 128),  # 61
    (1536, 1536, 128),  # 61
    (4096, 512, 128),  # 61
    (4096, 1024, 128),  # 61
    (4096, 1536, 128),  # 61
    (16384, 512, 128),  # 61
    (16384, 1024, 128),  # 61
    (16384, 1536, 128),  # 61
    (4096, 128, 128),  # 58
    (16384, 128, 128),  # 58
    (32768, 128, 128),  # 58
    (131072, 128, 128),  # 58
    (4096, 1152, 128),  # 3
    (16384, 1152, 128),  # 3
    (128, 7168, 1),  # 1
]

@pytest.mark.parametrize("rows,cols,group_size", QUANT_TEST_CASES)
def test_per_token_group_quant_int8(rows, cols, group_size):
    """Test per_token_group_quant_int8 implementation."""
    device = "cuda"
    x = torch.randn((rows, cols), dtype=torch.float16, device=device)
    
    # Run Triton implementation
    x_q_triton, x_s_triton = per_token_group_quant_int8(x, group_size)
    
    # Run native PyTorch implementation
    x_q_torch, x_s_torch = native_per_token_group_quant_int8(x, group_size)
    
    # Compare results
    torch.testing.assert_close(x_q_triton, x_q_torch, rtol=0, atol=1)
    torch.testing.assert_close(x_s_triton, x_s_torch, rtol=1e-3, atol=1e-3)

# Test cases for w8a8_block_int8_matmul
MATMUL_TEST_CASES = [
    # M, K, N, block_size  # frequency
    (128, 7168, 1536, [128, 128]),  # 30994
    (128, 7168, 576, [128, 128]),  # 30969
    (128, 128, 7168, [128, 128]),  # 29449
    (128, 7168, 256, [128, 128]),  # 29368
    (128, 1152, 7168, [128, 128]),  # 1526
    (128, 7168, 2304, [128, 128]),  # 1524
    (512, 512, 2048, [128, 128]),  # 61
    (1024, 1024, 7168, [128, 128]),  # 61
    (1536, 1536, 1536, [128, 128]),  # 61
    (4096, 512, 2048, [128, 128]),  # 61
    (4096, 1024, 7168, [128, 128]),  # 61
    (4096, 1536, 1536, [128, 128]),  # 61
    (4096, 7168, 576, [128, 128]),  # 61
    (4096, 7168, 1536, [128, 128]),  # 61
    (16384, 512, 2048, [128, 128]),  # 61
    (16384, 1024, 7168, [128, 128]),  # 61
    (16384, 1536, 1536, [128, 128]),  # 61
    (16384, 7168, 576, [128, 128]),  # 61
    (16384, 7168, 1536, [128, 128]),  # 61
    (4096, 128, 7168, [128, 128]),  # 58
    (4096, 7168, 256, [128, 128]),  # 58
    (16384, 128, 7168, [128, 128]),  # 58
    (16384, 7168, 256, [128, 128]),  # 58
    (4096, 1152, 7168, [128, 128]),  # 3
    (4096, 7168, 2304, [128, 128]),  # 3
    (16384, 1152, 7168, [128, 128]),  # 3
    (16384, 7168, 2304, [128, 128]),  # 3
]

KNB = [
    (128, 7168, [128, 128]),
    (512, 2048, [128, 128]),
    (512, 2048, [128, 128]),
    (1024, 7168, [128, 128]),
    (1024, 7168, [128, 128]),
    (1152, 7168, [128, 128]),
    (1536, 1536, [128, 128]),
    (1536, 1536, [128, 128]),
    (7168, 256, [128, 128]),
    (7168, 576, [128, 128]),
    (7168, 1536, [128, 128]),
    (7168, 2304, [128, 128])
]

# Function to expand test cases with power-of-2 M values
def expand_test_cases_with_m():
    original_cases = MATMUL_TEST_CASES.copy()
    expanded_cases = []

    # Generate M values: 1,2,4,8,16,...,16384
    m_values = [2**i for i in range(15)]  # 2^0 to 2^14 (1 to 16384)

    for k, n, block_size in KNB:
        # Add new test cases with different M values
        for m in m_values:
            expanded_cases.append((m, k, n, block_size))

    original_cases.extend(expanded_cases)
    return original_cases

# Expand test cases
MATMUL_TEST_CASES = expand_test_cases_with_m()
print("size of MATMUL_TEST_CASES: ", len(MATMUL_TEST_CASES))

@pytest.mark.parametrize("M,K,N,block_size", MATMUL_TEST_CASES)
def test_w8a8_block_int8_matmul(M, K, N, block_size):
    """Test w8a8_block_int8_matmul implementation."""
    device = "cuda"
    
    # Generate test data
    A = torch.randn((M, K), dtype=torch.float16, device=device)
    B = torch.randint(-128, 127, (N, K), dtype=torch.int8, device=device)
    
    # Quantize input A
    A_q, A_s = per_token_group_quant_int8(A, block_size[1])
    
    # Generate random scales for B
    block_n, block_k = block_size[0], block_size[1]
    n_tiles = triton.cdiv(N, block_n)
    k_tiles = triton.cdiv(K, block_k)
    B_s = torch.rand((n_tiles, k_tiles), dtype=torch.float32, device=device)
    
    # Run Triton implementation
    C_triton = w8a8_block_int8_matmul(A_q, B, A_s, B_s, block_size)
    
    # Run native PyTorch implementation
    C_torch = native_w8a8_block_int8_matmul(A_q, B, A_s, B_s, block_size)
    
    # Compare results
    torch.testing.assert_close(C_triton, C_torch, rtol=1e-2, atol=1e-2)

# Performance benchmarking configurations
quant_bench_configs = [
    triton.testing.Benchmark(
        x_names=['NUM_ROWS', 'NUM_COLS', 'GROUP_SIZE'],
        x_vals=QUANT_TEST_CASES,
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('red', '-'), ('blue', '--')],
        ylabel='Time (ms)',
        xlabel='Matrix Dimensions (Rows x Cols)',
        plot_name='Per-Token-Group INT8 Quantization Performance',
        args={'dtype': torch.float16, 'device': 'cuda'},
    )
]

@triton.testing.perf_report(quant_bench_configs)
def bench_per_token_group_quant_int8(NUM_ROWS, NUM_COLS, GROUP_SIZE, provider,
                                   dtype=torch.float16, device="cuda"):
    """Benchmark per_token_group_quant_int8 performance."""
    warmup = 25
    rep = 10
    
    x = torch.randn((NUM_ROWS, NUM_COLS), dtype=dtype, device=device)
    
    if provider == "triton":
        fn = lambda: per_token_group_quant_int8(x, GROUP_SIZE)
    else:  # provider == "torch"
        fn = lambda: native_per_token_group_quant_int8(x, GROUP_SIZE)
    
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    return ms

matmul_bench_configs = [
    triton.testing.Benchmark(
        x_names=['M', 'K', 'N', 'BLOCK_SIZE'],
        x_vals=MATMUL_TEST_CASES,
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('red', '-'), ('blue', '--')],
        ylabel='Time (ms)',
        xlabel='Matrix Dimensions (M×K×N)',
        plot_name='W8A8 Block INT8 MatMul Performance',
        args={'device': 'cuda'}
    )
]

@triton.testing.perf_report(matmul_bench_configs)
def bench_w8a8_block_int8_matmul(M, K, N, BLOCK_SIZE, provider, device="cuda"):
    """Benchmark w8a8_block_int8_matmul performance."""
    warmup = 25
    rep = 10
    
    block_n, block_k = BLOCK_SIZE[0], BLOCK_SIZE[1]
    n_tiles = triton.cdiv(N, block_n)
    k_tiles = triton.cdiv(K, block_k)
    
    # Generate benchmark data
    A = torch.randn((M, K), dtype=torch.float16, device=device)
    B = torch.randint(-128, 127, (N, K), dtype=torch.int8, device=device)
    A_q, A_s = native_per_token_group_quant_int8(A, block_k)
    B_s = torch.rand((n_tiles, k_tiles), dtype=torch.float32, device=device)
    
    if provider == "triton":
        fn = lambda: w8a8_block_int8_matmul(A_q, B, A_s, B_s, BLOCK_SIZE)
    else:  # provider == "torch"
        fn = lambda: native_w8a8_block_int8_matmul(A_q, B, A_s, B_s, BLOCK_SIZE)
    
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    return ms

if __name__ == "__main__":
    #unittest.main(verbosity=2)
    # Run benchmarks
    bench_per_token_group_quant_int8.run(print_data=True)
    bench_w8a8_block_int8_matmul.run(print_data=True)
