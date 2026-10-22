# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Runtime K and output-mask regressions for the G2/R1 N-split pipeline."""
import pytest
import torch
import triton
import triton.language as tl


@pytest.fixture(scope="module", autouse=True)
def full_precision_reference():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


@triton.jit
def _row_col_gemm(A, B, C, M, N, K, stride_am: tl.constexpr,
                  stride_bk: tl.constexpr, stride_cm: tl.constexpr,
                  BM: tl.constexpr = 256, BN: tl.constexpr = 256,
                  BK: tl.constexpr = 32, EVEN_K: tl.constexpr = False):
    tl.assume(K > 0)
    pid = tl.program_id(0)
    rm = (pid // tl.cdiv(N, BN)) * BM + tl.arange(0, BM)
    rn = (pid % tl.cdiv(N, BN)) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    ap = A + rm[:, None] * stride_am + rk[None, :]
    bp = B + rk[:, None] + rn[None, :] * stride_bk
    acc = tl.full((BM, BN), 0, tl.float32)
    for k in range(0, K, BK):
        if EVEN_K:
            # This variant uses padded input storage and complete BK32 tiles;
            # only output M/N are ragged. K remains a runtime argument.
            a = tl.load(ap)
            b = tl.load(bp)
        else:
            a = tl.load(ap, (rm[:, None] < M) & (k + rk[None, :] < K), 0)
            b = tl.load(bp, (rn[None, :] < N) & (k + rk[:, None] < K), 0)
        acc = tl.dot(a, b, acc)
        ap += BK
        bp += BK
    tl.store(C + rm[:, None] * stride_cm + rn[None, :], acc,
             (rm[:, None] < M) & (rn[None, :] < N))


@pytest.fixture(scope="module", params=[
    (torch.float16, False), (torch.float16, True),
    (torch.bfloat16, False), (torch.bfloat16, True),
])
def nsplit_kernel(request):
    if not torch.cuda.is_available():
        pytest.skip("requires an HCU device")
    dtype, aligned = request.param
    torch.manual_seed(88)
    a = torch.randn((768, 1024), device="cuda", dtype=dtype)
    b = torch.randn_like(a).T
    c = torch.empty((768, 768), device="cuda", dtype=dtype)
    # Reuse each binary across its runtime-K class and every output edge.
    # The masked variant also accepts K values without alignment hints.
    args = [a, b, c, 257, 257, 64 if aligned else 33, a.stride(0), b.stride(1), c.stride(0)]
    # gfx938 defaults to interleave MMAC; N-split uses the legacy layout.
    compiled = _row_col_gemm.warmup(
        *args, grid=(4,), num_warps=8, num_stages=4,
        use_async_copy=True, use_block_pingpong=True,
        block_pingpong_async_global_prefetch=2,
        block_pingpong_async_local_prefetch=1,
        matrix_instr_nonkdim=16, mmac_layout_force=1,
        optimize_epilogue=False, EVEN_K=aligned)
    if aligned:
        assert "hcu.block_pingpong.nsplit_applied" in compiled.asm["ttgir"]
    return compiled, args, aligned


@pytest.mark.parametrize("m,n", [(255, 257), (513, 513)])
@pytest.mark.parametrize("k", [1, 16, 31, 32, 33, 48, 64, 65, 96, 127, 128, 129])
def test_nsplit_runtime_k_and_masked_output(nsplit_kernel, m, n, k):
    compiled, original, aligned = nsplit_kernel
    if aligned and k % 32:
        pytest.skip("unmasked BK32 kernel requires a complete final K tile")
    args = list(original)
    args[3:6] = [m, n, k]
    a, b, c = args[:3]
    c.fill_(float("nan"))
    compiled[(triton.cdiv(m, 256) * triton.cdiv(n, 256), 1, 1)](*args, 256, 256, 32, aligned)
    ref = (a[:m, :k].float() @ b[:k, :n].float()).to(c.dtype)
    torch.testing.assert_close(c[:m, :n], ref, rtol=.02,
                               atol=.125 if c.dtype == torch.bfloat16 else .02)
    assert torch.isnan(c[m:, :]).all().item()
    assert torch.isnan(c[:m, n:]).all().item()


def test_fp32_output_uses_generic_accumulator_path(nsplit_kernel):
    _, original, complete_tiles = nsplit_kernel
    args = list(original)
    a, b = args[:2]
    c = torch.empty((768, 768), device="cuda", dtype=torch.float32)
    args[2] = c
    compiled = _row_col_gemm.warmup(
        *args, grid=(4,), num_warps=8, num_stages=4,
        use_async_copy=True, use_block_pingpong=True,
        block_pingpong_async_global_prefetch=2,
        block_pingpong_async_local_prefetch=1,
        matrix_instr_nonkdim=16, mmac_layout_force=1, optimize_epilogue=False,
        EVEN_K=complete_tiles)
    assert "hcu.block_pingpong.nsplit_applied" not in compiled.asm["ttgir"]
    for k in ([32, 128] if complete_tiles else [33, 129]):
        args[5] = k
        c.fill_(float("nan"))
        compiled[(4, 1, 1)](*args, 256, 256, 32, complete_tiles)
        ref = a[:257, :k].float() @ b[:k, :257].float()
        torch.testing.assert_close(c[:257, :257], ref, rtol=.02, atol=.02)
        assert torch.isnan(c[257:, :]).all().item()
        assert torch.isnan(c[:257, 257:]).all().item()


@triton.jit
def _row_col_gemm_with_reduction(A, B, C, R, K,
                                 PITCH: tl.constexpr):
    # An additional consumer must survive if the N-split tail cannot own the
    # whole accumulator-use graph. K is still a runtime scalar.
    tl.assume(K > 0)
    m = tl.arange(0, 256)
    n = tl.arange(0, 256)
    k = tl.arange(0, 32)
    ap = A + m[:, None] * PITCH + k[None, :]
    bp = B + k[:, None] + n[None, :] * PITCH
    acc = tl.full((256, 256), 0, tl.float32)
    for _ in range(0, K, 32):
        acc = tl.dot(tl.load(ap), tl.load(bp), acc)
        ap += 32
        bp += 32
    tl.store(C + m[:, None] * 256 + n[None, :], acc)
    tl.store(R + n, tl.sum(acc, axis=0))


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_extra_accumulator_consumer_uses_generic_path(dtype):
    if not torch.cuda.is_available():
        pytest.skip('requires an HCU device')
    torch.manual_seed(93)
    a = torch.randn((256, 1024), device='cuda', dtype=dtype)
    b = torch.randn_like(a).T
    c = torch.empty((256, 256), device='cuda', dtype=dtype)
    r = torch.empty(256, device='cuda', dtype=torch.float32)
    compiled = _row_col_gemm_with_reduction.warmup(
        a, b, c, r, 64, 1024, grid=(1,), num_warps=8, num_stages=4,
        use_async_copy=True, use_block_pingpong=True,
        block_pingpong_async_global_prefetch=2,
        block_pingpong_async_local_prefetch=1,
        matrix_instr_nonkdim=16, mmac_layout_force=1, optimize_epilogue=False)
    assert 'hcu.block_pingpong.nsplit_applied' not in compiled.asm['ttgir']
    for k in (32, 64, 128):
        compiled[(1, 1, 1)](a, b, c, r, k, 1024)
        ref = a[:, :k].float() @ b[:k, :].float()
        torch.testing.assert_close(c, ref.to(dtype), rtol=.02,
                                   atol=.125 if dtype == torch.bfloat16 else .02)
        torch.testing.assert_close(r, ref.sum(dim=0), rtol=.02, atol=.125)


@triton.jit
def _row_col_gemm_with_auxiliary_load(A, B, P, C, R, K,
                                     PITCH: tl.constexpr):
    tl.assume(K > 0)
    m = tl.arange(0, 256)
    n = tl.arange(0, 256)
    rk = tl.arange(0, 32)
    ap = A + m[:, None] * PITCH + rk[None, :]
    bp = B + rk[:, None] + n[None, :] * PITCH
    acc = tl.full((256, 256), 0, tl.float32)
    extra = tl.full((), 0, tl.float32)
    for k in range(0, K, 32):
        acc = tl.dot(tl.load(ap), tl.load(bp), acc)
        extra += tl.sum(tl.load(P + k + rk), axis=0)
        ap += 32
        bp += 32
    tl.store(C + m[:, None] * 256 + n[None, :], acc)
    tl.store(R, extra)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_auxiliary_global_load_uses_generic_waits(dtype):
    if not torch.cuda.is_available():
        pytest.skip('requires an HCU device')
    torch.manual_seed(95)
    a = torch.randn((256, 1024), device='cuda', dtype=dtype)
    b = torch.randn_like(a).T
    p = torch.randn(1024, device='cuda', dtype=torch.float32)
    c = torch.empty((256, 256), device='cuda', dtype=dtype)
    r = torch.empty((), device='cuda', dtype=torch.float32)
    # Two stages keep the generic fallback within the 64KiB LDS budget.
    compiled = _row_col_gemm_with_auxiliary_load.warmup(
        a, b, p, c, r, 64, 1024, grid=(1,), num_warps=8, num_stages=2,
        use_async_copy=True, use_block_pingpong=True,
        block_pingpong_async_global_prefetch=2,
        block_pingpong_async_local_prefetch=1,
        matrix_instr_nonkdim=16, mmac_layout_force=1, optimize_epilogue=False)
    assert 'hcu.block_pingpong.nsplit' not in compiled.asm['ttgir']
    for k in (32, 64, 128):
        compiled[(1, 1, 1)](a, b, p, c, r, k, 1024)
        ref = a[:, :k].float() @ b[:k, :].float()
        torch.testing.assert_close(c, ref.to(dtype), rtol=.02,
                                   atol=.125 if dtype == torch.bfloat16 else .02)
        torch.testing.assert_close(r, p[:k].sum(), rtol=1e-4, atol=1e-4)
