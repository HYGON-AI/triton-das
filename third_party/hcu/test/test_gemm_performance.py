# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Fixed-schedule, runtime-K GEMM gates measured on an idle NMZ (gfx938)."""
import statistics

import pytest
import torch
import triton
import triton.language as tl


# Reference TFLOPS come from the same opt:8576 container and graph protocol.
# Rows are (M=N=K, B layout, MMAC layout); explicit legacy and defaults are
# distinct compiler paths. Do not replace these with BMZ or static-K numbers.
# The calibrated references are listed below the timing helper.


@triton.jit
def _gemm(a, b, c, M, N, K,
          stride_am: tl.constexpr, stride_ak: tl.constexpr,
          stride_bk: tl.constexpr, stride_bn: tl.constexpr,
          stride_cm: tl.constexpr, stride_cn: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
          GROUP_M: tl.constexpr):
    tl.assume(M > 0)
    tl.assume(N > 0)
    tl.assume(K > 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    ap = a + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    bp = b + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in range(0, K, BK):
        acc = tl.dot(tl.load(ap), tl.load(bp), acc)
        ap += BK * stride_ak
        bp += BK * stride_bk
    cp = c + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(cp, acc.to(c.dtype.element_ty))


def _graph_samples(run):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):
            run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(4):
                run()
    torch.cuda.synchronize()
    for _ in range(3):
        graph.replay()
    samples = []
    for _ in range(11):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / 4)
    return samples


# gfx938/BW1100, 72 CUs, opt:8576; Triton cdf6827 + LLVM 96fd797.
# Lowest median of three independent runs, rounded down to 0.01 TFLOPS.
# Evidence: triton_gemm_asm/experiments/20261030-nmz-release-regression/.
REFERENCES = {
    (4096, 'row-col', 'default'): 137.06,
    (4096, 'row-col', 'legacy'): 243.99,
    (4096, 'row-row', 'default'): 248.94,
    (4096, 'row-row', 'legacy'): 257.31,
    (5120, 'row-col', 'default'): 141.43,
    (5120, 'row-col', 'legacy'): 253.57,
    (5120, 'row-row', 'default'): 271.58,
    (5120, 'row-row', 'legacy'): 278.83,
}


@pytest.mark.parametrize('size,layout,mmac', REFERENCES)
def test_nmz_gemm_performance(monkeypatch, size, layout, mmac):
    if not torch.cuda.is_available():
        pytest.skip('requires an NMZ GPU')
    device = torch.cuda.get_device_properties(0)
    if device.gcnArchName.split(':')[0] != 'gfx938' or device.multi_processor_count != 72:
        pytest.skip('references measured on 72-CU gfx938 (BW1100)')
    monkeypatch.setenv('AMDGCN_USE_BUFFER_OPS', '1')
    monkeypatch.setenv('TRITON_HCU_ENABLE_DS_READ_M', '1')
    monkeypatch.setenv('TRITON_BUFFER_CACHE_SWIZZLE', '1')
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', False)
    monkeypatch.setattr(torch.backends.cudnn, 'allow_tf32', False)
    torch.manual_seed(44)
    a = torch.randn((size, size), device='cuda', dtype=torch.float16)
    storage = torch.randn((size, size), device='cuda', dtype=torch.float16)
    b = storage.T if layout == 'row-col' else storage
    c = torch.empty_like(a)
    options = dict(BM=256, BN=256, BK=32, GROUP_M=2,
                   num_warps=8, num_stages=4, waves_per_eu=0,
                   use_block_pingpong=True, use_async_copy=True,
                   buffer_cache_swizzle=True,
                   block_pingpong_async_global_prefetch=2,
                   block_pingpong_async_local_prefetch=1,
                   matrix_instr_nonkdim=16, optimize_epilogue=False,
                   max_memory_cluster_dwords=32,
                   amdgpu_enable_max_ilp_scheduling_strategy=False)
    if layout == 'row-row':
        options.update(BK=16, num_stages=5, waves_per_eu=3,
                       block_pingpong_async_global_prefetch=4,
                       optimize_epilogue=True)
    if mmac == 'legacy':
        options['mmac_layout_force'] = 1
    k = size

    def run():
        return _gemm[(triton.cdiv(size, 256)**2,)](
            a, b, c, size, size, k, *a.stride(), *b.stride(),
            *c.stride(), **options)

    hashes = set()

    def check_runtime_k():
        nonlocal k
        for k in [size - 64, size]:
            compiled = run()
            torch.cuda.synchronize()
            hashes.add(compiled.hash)
            for start in range(0, size, 256):
                ref = (a[start:start + 256, :k].float() @ b[:k].float()).to(c.dtype)
                torch.testing.assert_close(c[start:start + 256], ref, rtol=.02, atol=.02)
        assert len(hashes) == 1, 'K must remain runtime and reuse one binary'

    check_runtime_k()
    ms = statistics.median(_graph_samples(run))
    check_runtime_k()
    current = 2 * size**3 / ms / 1e9
    minimum = REFERENCES[size, layout, mmac] * .99
    print(f'NMZ GEMM {size=} {layout=} {mmac=}: {current:.3f} TFLOPS, floor {minimum:.3f}')
    assert current >= minimum, (
        f'{size=} {layout=} {mmac=}: {current:.3f} TFLOPS < '
        f'{minimum:.3f} TFLOPS (99% of frozen reference)')
