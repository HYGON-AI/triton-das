"""
GEMM with MLS (tl.matrix_load) + WASP (warp_specialize).

For the full GEMM/FA × MLS/load × wasp4+4/wdra4+8 matrix, see
test_wasp_mls_regression.py.
"""

import argparse
import os

import pytest
import torch
import triton
import triton.language as tl

os.environ["AMDGCN_USE_BUFFER_OPS"] = "1"


def is_hcu_support_mls():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "hip" and (
        target.arch == "gfx938" or target.arch == "gfx946" or target.arch == "gfx92a"
    )


@triton.jit
def matmul_mls_wasp_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """C = A @ B with MLS loads inside a WASP-specialized K loop.

    A: [M, K], B: [K, N], C: [M, N]
    """
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(M > 0)
    tl.assume(N > 0)
    tl.assume(K > 0)

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    mls_offs_am = pid_m * BLOCK_SIZE_M
    mls_offs_bn = pid_n * BLOCK_SIZE_N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # WASP: specialize load vs MMA warps on the K-loop (see matmul_wasp.py).
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), warp_specialize=True):
        mls_offs_k = k * BLOCK_SIZE_K
        # MLS: matrix_load for A and B (see mls-unit-test.py, mix_load=0b00).
        a = tl.matrix_load(
            a_ptr,
            shape=[M, K],
            strides=[stride_am, stride_ak],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
            offsets=[mls_offs_am, mls_offs_k],
        )
        b = tl.matrix_load(
            b_ptr,
            shape=[K, N],
            strides=[stride_bk, stride_bn],
            block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
            offsets=[mls_offs_k, mls_offs_bn],
        )
        accumulator = tl.dot(a, b, accumulator)

    if ACTIVATION == "leaky_relu":
        accumulator = tl.where(accumulator >= 0, accumulator, 0.01 * accumulator)
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def default_wasp_config():
    """WASP 4+4 load/mma warps, WDRA off."""
    return {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "wasp_enabled": True,
        "wasp_num_load_warps": 4,
        "wasp_num_mma_warps": 4,
        "wdra_enabled": False,
    }


def matmul_mls_wasp(a, b, activation="", config=None, compile_only=False):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    if config is None:
        config = default_wasp_config()

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    args = (
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    if compile_only:
        matmul_mls_wasp_kernel.warmup(
            *args, ACTIVATION=activation, grid=grid, **config
        )
    else:
        matmul_mls_wasp_kernel[grid](
            *args, ACTIVATION=activation, **config
        )
    return c


# @pytest.mark.xfail(
#     reason="MLS (tl.matrix_load) + WASP (warp_specialize) not yet supported on HCU",
#     raises=Exception,
#     strict=False,
# )
def test_matmul_mls_wasp():
    if not is_hcu_support_mls():
        pytest.skip("skip: not support mls")

    torch.manual_seed(0)
    M, N, K = 128, 128, 4096
    # A row-major [M, K]; B col-major via (N, K).T -> [K, N] (matches matmul_wasp.py).
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((N, K), device="cuda", dtype=torch.float16).transpose(1, 0)

    triton_output = matmul_mls_wasp(a, b)
    torch_output = torch.matmul(a, b)
    torch.testing.assert_close(triton_output, torch_output, atol=1e-2, rtol=1e-2)


def run_standalone(M, N, K, compile_only=False):
    torch.manual_seed(0)
    a = torch.randn((M, K), device="cpu", dtype=torch.float16)
    b = torch.randn((N, K), device="cpu", dtype=torch.float16).transpose(1, 0)
    triton_output = matmul_mls_wasp(
        a.to("cuda"), b.to("cuda"), compile_only=compile_only
    ).to("cpu")
    if compile_only:
        print(f"compile-only OK for M={M} N={N} K={K}")
        return
    torch_output = torch.matmul(a, b)
    torch.testing.assert_close(triton_output, torch_output, atol=1e-2, rtol=1e-2)
    print(f"PASS: M={M} N={N} K={K}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GEMM MLS + WASP tracking test (expected to fail until support lands)"
    )
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()

    if not is_hcu_support_mls():
        raise RuntimeError("Current target does not support MLS")

    for shape in [(128, 128, 4096)]:
        run_standalone(*shape, compile_only=args.compile_only)
    print("Test done!")
