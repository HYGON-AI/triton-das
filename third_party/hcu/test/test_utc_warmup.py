"""NMZ/BMZ UTC warmup smoke test."""

import os
import re

import pytest
import torch
import triton
import triton.language as tl


os.environ.setdefault("AMDGCN_USE_BUFFER_OPS", "1")


def _is_nmz_or_bmz():
    try:
        target = triton.runtime.driver.active.get_current_target()
    except Exception:
        return False
    return (
        target is not None
        and target.backend == "hip"
        and target.arch.split(":")[0] in {"gfx936", "gfx938"}
    )


pytestmark = pytest.mark.skipif(
    not _is_nmz_or_bmz(),
    reason="UTC warmup smoke test requires BMZ (gfx936) or NMZ (gfx938)",
)


@triton.jit
def _utc_warmup_gemm(
    a_ptr,
    b_ptr,
    c_ptr,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tl.assume(K > 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = (
        a_ptr
        + offs_m[:, None] * stride_am
        + offs_k[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr
        + offs_k[:, None] * stride_bk
        + offs_n[None, :] * stride_bn
    )
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = (
        c_ptr
        + offs_m[:, None] * stride_cm
        + offs_n[None, :] * stride_cn
    )
    tl.store(c_ptrs, accumulator.to(tl.float16))


def _compile_and_run(a, b, output, *, utc_warmup):
    K = a.shape[1]
    options = {
        "BLOCK_M": 256,
        "BLOCK_N": 256,
        "BLOCK_K": 32,
        "num_warps": 8,
        "num_stages": 2,
        "buffer_cache_swizzle": True,
        "use_block_pingpong": False,
        "utc_warmup": utc_warmup,
    }
    compiled = _utc_warmup_gemm.warmup(
        a,
        b,
        output,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        output.stride(0),
        output.stride(1),
        grid=(1,),
        **options,
    )
    _utc_warmup_gemm[(1,)](
        a,
        b,
        output,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        output.stride(0),
        output.stride(1),
        **options,
    )
    torch.cuda.synchronize()
    return compiled


def _warmup_loads(amdgcn):
    """Return warmup buffer loads emitted by the side-effecting inline asm."""
    loads = []
    in_inline_asm = False
    for line in amdgcn.splitlines():
        if "#ASMSTART" in line:
            in_inline_asm = True
        elif "#ASMEND" in line:
            in_inline_asm = False
        elif in_inline_asm and re.search(
            r"\bbuffer_load_dword\b.*\boffen\b", line
        ):
            loads.append(line)
    return loads


def test_utc_warmup():
    """Check final warmup instructions and GEMM correctness on BMZ/NMZ."""
    torch.manual_seed(0)
    M = N = 256
    K = 8192  # Each compact A/B operand panel spans two complete 2 MiB pages.
    a = (torch.randn((M, K), device="cuda", dtype=torch.float16) * 0.1)
    b = (
        torch.randn((N, K), device="cuda", dtype=torch.float16) * 0.1
    ).transpose(0, 1)
    cold = torch.empty((M, N), device="cuda", dtype=torch.float16)
    warm = torch.empty_like(cold)

    cold_kernel = _compile_and_run(a, b, cold, utc_warmup=False)
    warm_kernel = _compile_and_run(a, b, warm, utc_warmup=True)

    assert _warmup_loads(cold_kernel.asm["amdgcn"]) == []
    assert len(_warmup_loads(warm_kernel.asm["amdgcn"])) == 2

    reference = torch.matmul(a.float(), b.float()).to(torch.float16)
    torch.testing.assert_close(cold, reference, atol=2e-2, rtol=2e-2)
    assert torch.equal(warm, cold)
