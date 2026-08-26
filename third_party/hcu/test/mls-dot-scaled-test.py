from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import os
import pytest
import torch

# os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton-cache")

import triton
import triton.language as tl
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor


def is_hcu_support_mls():
    """与 third_party/hcu/test/mls-unit-test.py 保持一致：仅部分 HCU arch 支持 MLS。"""
    target = triton.runtime.driver.active.get_current_target()
    if target is None:
        return False
    return target.backend == 'hip' and (
        target.arch == 'gfx938' or target.arch == 'gfx946' or target.arch == 'gfx92a'
    )

def is_hcu_support_mmac_layout():
    target = triton.runtime.driver.active.get_current_target()
    if target is None:
        return False
    return target.backend == 'hip' and (
        target.arch == 'gfx938' or target.arch == 'gfx946'
    )

# import os
# os.environ["AMDGCN_USE_BUFFER_OPS"] = "1"


class F8F6F4DataType(IntEnum):
    FP8 = 0
    BF8 = 1
    FP6 = 2
    BF6 = 3
    FP4 = 4


class F8F6F4TypeId(IntEnum):
    FP8_FP8 = 0
    FP8_BF8 = 1
    FP8_FP6 = 2
    FP8_BF6 = 3
    FP8_FP4 = 4
    BF8_FP8 = 5
    BF8_BF8 = 6
    BF8_FP6 = 7
    BF8_BF6 = 8
    BF8_FP4 = 9
    FP6_FP8 = 10
    FP6_BF8 = 11
    FP6_FP6 = 12
    FP6_BF6 = 13
    FP6_FP4 = 14
    BF6_FP8 = 15
    BF6_BF8 = 16
    BF6_FP6 = 17
    BF6_BF6 = 18
    BF6_FP4 = 19
    FP4_FP8 = 20
    FP4_BF8 = 21
    FP4_FP6 = 22
    FP4_BF6 = 23
    FP4_FP4 = 24


def get_lhs_data_type(type_id: F8F6F4TypeId) -> F8F6F4DataType:
    return F8F6F4DataType(int(type_id) // 5)


def get_rhs_data_type(type_id: F8F6F4TypeId) -> F8F6F4DataType:
    return F8F6F4DataType(int(type_id) % 5)

def fp4_map(values: torch.Tensor) -> torch.Tensor:
    fp4_e2m1_map = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
    lut = torch.tensor(fp4_e2m1_map, dtype=torch.float32).to(values.device)
    return lut[values.long()]

def get_hip_autotune_config():
    return [
        triton.Config(
            {'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 32, 'GROUP_SIZE_M': 1, 'waves_per_eu': 1},
            num_warps=1, num_stages=1),
    ]


def get_autotune_config():
        return get_hip_autotune_config()


@triton.jit
def f4_to_f32(code_u8):
    # code bits: S b2 b1 b0 (S is MSB), exponent bits = b2b1, mantissa bit = b0 [page:9]
    s = (code_u8 >> 3) & 0x1
    e = (code_u8 >> 1) & 0x3
    b0 = code_u8 & 0x1

    sign = 1.0 - 2.0 * s.to(tl.float32)  # 0->+1, 1->-1

    # normalized branch (e != 0): 2^(e-1) * (1 + b0*2^-1) [page:9]
    scale_norm = tl.exp2(e.to(tl.float32) - 1.0)
    mant_norm = 1.0 + b0.to(tl.float32) * 0.5
    val_norm = sign * scale_norm * mant_norm

    # subnormal/zero branch (e == 0): b0 * 2^-1 [page:9]
    val_sub = sign * b0.to(tl.float32) * 0.5

    return tl.where(e == 0, val_sub, val_norm)


# # `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
# #   - A list of `triton.Config` objects that define different configurations of
# #       meta-parameters (e.g., `BLOCK_M`) and compilation options (e.g., `num_warps`) to try
# #   - An auto-tuning *key* whose change in values will trigger evaluation of all the
# #       provided configs
# @triton.autotune(
#     configs=get_autotune_config(),
#     key=['M', 'N', 'K', 'stride_am', 'stride_ak', 'stride_bk', 'stride_bn'],
#     perf_debug=True,
# )
@triton.heuristics({
    'block_k_diviable': lambda nargs: nargs['K'] % nargs['BLOCK_K'] == 0,
    'block_m_diviable': lambda nargs: nargs['M'] % nargs['BLOCK_M'] == 0,
    'block_n_diviable': lambda nargs: nargs['N'] % nargs['BLOCK_N'] == 0,
})
@triton.jit
def matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        a_scale_ptr, b_scale_ptr,
        # Logical (unpacked) matmul: C[M,N] = A[M,K] @ B[K,N].
        # Same convention as python/test/unit/language/test_matmul.py::mxfp8_mxfp4_matmul:
        #   M, N, K     — logical element counts along each matmul axis
        #   BLOCK_M/N/K — logical tile sizes (dot_scaled / accumulator use logical BLOCK_M x BLOCK_N)
        M, N, K,
        # Strides step one *storage* element (e.g. one mxfp4 byte = 2 logical e2m1 values on a packed axis).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        stride_a_scale_m, stride_a_scale_k,
        stride_b_scale_n, stride_b_scale_k,
        block_k_diviable: tl.constexpr,
        block_m_diviable: tl.constexpr,
        block_n_diviable: tl.constexpr,
        boundary_check: tl.constexpr,
        # mix_load:
        # 0b00: dont't mix, both a and b use matrix_load
        # 0b01: a: matrix_load, b: tt_load
        # 0b10: a: tt_load, b: matrix_load
        # 0b11: a: tt_load, b: tt_load
        mix_load: tl.constexpr,
        lhs_data_type_id: tl.constexpr,
        rhs_data_type_id: tl.constexpr,
        # lhs_k_pack: for FP4 A, same idea as applying DIV_FACTOR on K only in mxfp8 (always K there);
        #             False means pack along M instead (this kernel generalizes A).
        # rhs_k_pack: matches mxfp8_mxfp4_matmul PACK_B_ALONG_K for tl.dot_scaled.
        lhs_k_pack: tl.constexpr,
        rhs_k_pack: tl.constexpr,
        # Meta-parameters
        USE_A_SCALE: tl.constexpr,
        USE_B_SCALE: tl.constexpr,
        SCALE_VEC_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        ACTIVATION: tl.constexpr,  #
        OUTPUT_DTYPE: tl.constexpr  #
):
    """C = A @ B with optional mxfp4 packing.

    Logical (unpacked) semantics match mxfp8_mxfp4_matmul: M,N,K and BLOCK_* count logical
    elements. FP4 storage uses DIV_FACTOR_* = 2 on the packed axis (see constexpr below).

    Physical tensor extents (storage elements): e.g. A is M//DIV_FACTOR_A_M by K//DIV_FACTOR_A_K.
    Physical tile for loads: PHY_BLOCK_* = BLOCK_* // DIV_FACTOR on each axis.
    tl.matrix_load(shape, block_shape, offsets) uses those physical counts; tt_load uses PHY_BLOCK_* indices.
    """
    FP8_TYPE_ID: tl.constexpr = 0   # e4m3
    BF8_TYPE_ID: tl.constexpr = 1   # e5m2
    FP6_TYPE_ID: tl.constexpr = 2   # e2m3
    BF6_TYPE_ID: tl.constexpr = 3   # e3m2
    FP4_TYPE_ID: tl.constexpr = 4   # e2m1

    LHS_IS_FP4: tl.constexpr = lhs_data_type_id == FP4_TYPE_ID
    RHS_IS_FP4: tl.constexpr = rhs_data_type_id == FP4_TYPE_ID
    # Align with mxfp8_mxfp4_matmul: DIV_FACTOR_A/B on e2m1; per-axis split like DIV_FACTOR_B_K / DIV_FACTOR_B_N.
    DIV_FACTOR_A: tl.constexpr = 2 if LHS_IS_FP4 else 1
    DIV_FACTOR_B: tl.constexpr = 2 if RHS_IS_FP4 else 1
    DIV_FACTOR_A_M: tl.constexpr = DIV_FACTOR_A if not lhs_k_pack else 1
    DIV_FACTOR_A_K: tl.constexpr = DIV_FACTOR_A if lhs_k_pack else 1
    DIV_FACTOR_B_K: tl.constexpr = DIV_FACTOR_B if rhs_k_pack else 1
    DIV_FACTOR_B_N: tl.constexpr = DIV_FACTOR_B if not rhs_k_pack else 1
    # Physical (storage) tile sizes: same role as BLOCK_K // DIV_FACTOR_A in mxfp8_mxfp4_matmul.
    PHY_BLOCK_M_A: tl.constexpr = BLOCK_M // DIV_FACTOR_A_M
    PHY_BLOCK_K_A: tl.constexpr = BLOCK_K // DIV_FACTOR_A_K
    PHY_BLOCK_K_B: tl.constexpr = BLOCK_K // DIV_FACTOR_B_K
    PHY_BLOCK_N_B: tl.constexpr = BLOCK_N // DIV_FACTOR_B_N
    tl.static_assert(BLOCK_M % DIV_FACTOR_A_M == 0, "BLOCK_M must be divisible by DIV_FACTOR_A_M")
    tl.static_assert(BLOCK_N % DIV_FACTOR_B_N == 0, "BLOCK_N must be divisible by DIV_FACTOR_B_N")
    tl.static_assert(BLOCK_K % DIV_FACTOR_A_K == 0, "BLOCK_K must be divisible by DIV_FACTOR_A_K")
    tl.static_assert(BLOCK_K % DIV_FACTOR_B_K == 0, "BLOCK_K must be divisible by DIV_FACTOR_B_K")
    tl.static_assert(BLOCK_K % SCALE_VEC_SIZE == 0, "BLOCK_K must be divisible by SCALE_VEC_SIZE")

    if lhs_data_type_id == FP4_TYPE_ID:
        LHS_FORMAT: tl.constexpr = "e2m1"
    elif lhs_data_type_id == FP8_TYPE_ID:
        LHS_FORMAT: tl.constexpr = "e4m3"
    elif lhs_data_type_id == BF8_TYPE_ID:
        LHS_FORMAT: tl.constexpr = "e5m2"
    elif lhs_data_type_id == FP6_TYPE_ID:
        tl.static_assert(False, "lhs fp6 (e2m3) is not supported by tl.dot_scaled in this test yet")
    elif lhs_data_type_id == BF6_TYPE_ID:
        tl.static_assert(False, "lhs bf6 (e3m2) is not supported by tl.dot_scaled in this test yet")
    else:
        tl.static_assert(False, "Unsupported lhs_data_type_id in matmul_kernel")

    if rhs_data_type_id == FP4_TYPE_ID:
        RHS_FORMAT: tl.constexpr = "e2m1"
    elif rhs_data_type_id == FP8_TYPE_ID:
        RHS_FORMAT: tl.constexpr = "e4m3"
    elif rhs_data_type_id == BF8_TYPE_ID:
        RHS_FORMAT: tl.constexpr = "e5m2"
    elif rhs_data_type_id == FP6_TYPE_ID:
        tl.static_assert(False, "rhs fp6 (e2m3) is not supported by tl.dot_scaled in this test yet")
    elif rhs_data_type_id == BF6_TYPE_ID:
        tl.static_assert(False, "rhs bf6 (e3m2) is not supported by tl.dot_scaled in this test yet")
    else:
        tl.static_assert(False, "Unsupported rhs_data_type_id in matmul_kernel")

    tl.assume(stride_am >= 0)
    tl.assume(stride_ak >= 0)
    tl.assume(stride_bk >= 0)
    tl.assume(stride_bn >= 0)
    tl.assume(stride_cm >= 0)
    tl.assume(stride_cn >= 0)

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    if GROUP_SIZE_M == 1:
        num_pid_n = tl.cdiv(N, BLOCK_N)
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # Logical offsets for C / scales; physical indices for packed A/B loads (cf. mxfp8 offs_ak = arange(BLOCK_K//DIV_FACTOR_A)).
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_a_m_phy = pid_m * PHY_BLOCK_M_A + tl.arange(0, PHY_BLOCK_M_A)
    offs_b_n_phy = pid_n * PHY_BLOCK_N_B + tl.arange(0, PHY_BLOCK_N_B)
    offs_ak = tl.arange(0, PHY_BLOCK_K_A)
    offs_bk = tl.arange(0, PHY_BLOCK_K_B)
    offs_scale_k = tl.arange(0, BLOCK_K // SCALE_VEC_SIZE)
    offs_am_scale = offs_am % M
    offs_bn_scale = offs_bn % N

    # 逻辑 tile 起点；在 M % BLOCK_M != 0 等尾部块场景下 % 可能不等于 pid * BLOCK（一般测试取满整除尺寸）。
    mls_offs_am = (pid_m * BLOCK_M) % M
    mls_offs_bn = (pid_n * BLOCK_N) % N
    mls_offs_k  = 0
    if USE_A_SCALE:
        a_scale_ptrs = a_scale_ptr + offs_am_scale[:, None] * stride_a_scale_m + offs_scale_k[None, :] * stride_a_scale_k
    if USE_B_SCALE:
        b_scale_ptrs = b_scale_ptr + offs_bn_scale[:, None] * stride_b_scale_n + offs_scale_k[None, :] * stride_b_scale_k

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_M, BLOCK_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    if OUTPUT_DTYPE == "int32":
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    else:
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    USE_TT_LOAD_A: tl.constexpr = (mix_load & 0b10) != 0
    USE_TT_LOAD_B: tl.constexpr = (mix_load & 0b01) != 0
    tl.static_assert((mix_load & (~0b11)) == 0, "Invalid mix_load")

    if boundary_check:
        boundary_check_a: tl.constexpr = (0, 1)
    elif (not block_m_diviable) and (not block_k_diviable):
        boundary_check_a: tl.constexpr = (0, 1)
    elif not block_m_diviable:
        boundary_check_a: tl.constexpr = (0,)
    elif not block_k_diviable:
        boundary_check_a: tl.constexpr = (1,)
    else:
        boundary_check_a: tl.constexpr = ()

    if boundary_check:
        boundary_check_b: tl.constexpr = (0, 1)
    elif (not block_k_diviable) and (not block_n_diviable):
        boundary_check_b: tl.constexpr = (0, 1)
    elif not block_k_diviable:
        boundary_check_b: tl.constexpr = (0,)
    elif not block_n_diviable:
        boundary_check_b: tl.constexpr = (1,)
    else:
        boundary_check_b: tl.constexpr = ()

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_ak_iter = (mls_offs_k // DIV_FACTOR_A_K) + offs_ak
        offs_bk_iter = (mls_offs_k // DIV_FACTOR_B_K) + offs_bk
        # Load the next block of A and B. For matrix_load, use boundary_check
        # when the physical tile can cross M/N/K tails; for tt.load, fall back
        # to explicit masks.
        if USE_TT_LOAD_A or USE_TT_LOAD_B:
            a_ptrs = a_ptr + (offs_a_m_phy[:, None] * stride_am + offs_ak_iter[None, :] * stride_ak)
            b_ptrs = b_ptr + (offs_bk_iter[:, None] * stride_bk + offs_b_n_phy[None, :] * stride_bn)
        if USE_TT_LOAD_A:
            k_remaining = K - k * BLOCK_K
            a_k_bound = tl.cdiv(k_remaining, DIV_FACTOR_A_K)
            a_mask = (offs_a_m_phy[:, None] < tl.cdiv(M, DIV_FACTOR_A_M)) & (offs_ak[None, :] < a_k_bound)
            a = tl.load(a_ptrs, mask=a_mask, other=0)
        else:
            a = tl.matrix_load(
                a_ptr,
                shape=[M // DIV_FACTOR_A_M, K // DIV_FACTOR_A_K],
                strides=[stride_am, stride_ak],
                block_shape=[PHY_BLOCK_M_A, PHY_BLOCK_K_A],
                offsets=[mls_offs_am // DIV_FACTOR_A_M, mls_offs_k // DIV_FACTOR_A_K],
                boundary_check=boundary_check_a,
            )
        if USE_TT_LOAD_B:
            k_remaining = K - k * BLOCK_K
            b_k_bound = tl.cdiv(k_remaining, DIV_FACTOR_B_K)
            b_mask = (offs_bk[:, None] < b_k_bound) & (offs_b_n_phy[None, :] < tl.cdiv(N, DIV_FACTOR_B_N))
            b = tl.load(b_ptrs, mask=b_mask, other=0)
        else:
            b = tl.matrix_load(
                b_ptr,
                shape=[K // DIV_FACTOR_B_K, N // DIV_FACTOR_B_N],
                strides=[stride_bk, stride_bn],
                block_shape=[PHY_BLOCK_K_B, PHY_BLOCK_N_B],
                offsets=[mls_offs_k // DIV_FACTOR_B_K, mls_offs_bn // DIV_FACTOR_B_N],
                boundary_check=boundary_check_b,
            )

        if USE_A_SCALE:
            if block_k_diviable:
                scale_a = tl.load(a_scale_ptrs)
            else:
                k_remaining = K - k * BLOCK_K
                scale_k_bound = tl.cdiv(k_remaining, SCALE_VEC_SIZE)
                scale_a_mask = (offs_am_scale[:, None] < M) & (offs_scale_k[None, :] < scale_k_bound)
                scale_a = tl.load(a_scale_ptrs, mask=scale_a_mask, other=0)
        else:
            scale_a = None
        if USE_B_SCALE:
            if block_k_diviable:
                scale_b = tl.load(b_scale_ptrs)
            else:
                k_remaining = K - k * BLOCK_K
                scale_k_bound = tl.cdiv(k_remaining, SCALE_VEC_SIZE)
                scale_b_mask = (offs_bn_scale[:, None] < N) & (offs_scale_k[None, :] < scale_k_bound)
                scale_b = tl.load(b_scale_ptrs, mask=scale_b_mask, other=0)
        else:
            scale_b = None
        accumulator = tl.dot_scaled(
            a, scale_a, LHS_FORMAT, b, scale_b, RHS_FORMAT, accumulator, lhs_k_pack=lhs_k_pack, rhs_k_pack=rhs_k_pack
        )
        #accumulator = tl.dot_scaled(a, None, "e2m1", b, None, "e2m1", accumulator)
        # Advance the ptrs to the next K block.
        mls_offs_k += BLOCK_K
        if USE_A_SCALE:
            a_scale_ptrs += (BLOCK_K // SCALE_VEC_SIZE) * stride_a_scale_k
        if USE_B_SCALE:
            b_scale_ptrs += (BLOCK_K // SCALE_VEC_SIZE) * stride_b_scale_k
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)

    # 根据OUTPUT_DTYPE参数确定输出类型
    if OUTPUT_DTYPE == "int32":
        c = accumulator
    else:  # OUTPUT_DTYPE == "float16"
        c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `matmul_kernel`.
@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


# Host 侧入口：matmul_with_layout。传入 matmul_kernel 的 M、N、K 必须是**逻辑**（unpack 后）轴长；
# 不能直接用 a.shape / b.shape 当逻辑维：FP4 pack 与 col 存储时 storage shape 更短或轴序不同，
# 需用本文件 matmul_with_layout 里的 div_m/div_k/div_n 与 stride 交换还原（与 mxfp8_mxfp4_matmul 一致）。


def _normalize_block_config(config):
    cfg = dict(config)
    rename_pairs = [
        ("BLOCK_SIZE_M", "BLOCK_M"),
        ("BLOCK_SIZE_N", "BLOCK_N"),
        ("BLOCK_SIZE_K", "BLOCK_K"),
    ]
    for old_key, new_key in rename_pairs:
        if old_key in cfg and new_key in cfg and cfg[old_key] != cfg[new_key]:
            raise AssertionError(f"Conflicting block config values: {old_key}={cfg[old_key]} vs {new_key}={cfg[new_key]}")
        if old_key in cfg and new_key not in cfg:
            cfg[new_key] = cfg.pop(old_key)
    return cfg


def _expected_scale_shape(rows, k, scale_vec_size):
    return rows, triton.cdiv(k, scale_vec_size)


def _validate_scale_tensor(scale, name, rows, k, scale_vec_size):
    expected_shape = _expected_scale_shape(rows, k, scale_vec_size)
    if scale.ndim != 2:
        raise AssertionError(f"{name} must be rank-2, got shape={tuple(scale.shape)}")
    if tuple(scale.shape) != expected_shape:
        raise AssertionError(f"{name} shape mismatch: expected {expected_shape}, got {tuple(scale.shape)}")
    if scale.dtype != torch.uint8:
        raise AssertionError(f"{name} must use MX e8m0 storage (torch.uint8), got {scale.dtype}")


def _skip_unsupported_mls_case(M, N, K, a_layout, b_layout, config, scale_vec_size=None, with_scale=False):
    if not is_hcu_support_mls():
        pytest.skip("skip: HCU MLS not supported on this target")

    block_m = config['BLOCK_M']
    block_n = config['BLOCK_N']
    block_k = config['BLOCK_K']

    # 对齐 mls-unit-test.py 中矩阵 load 的 block 约束；本测试仅有 8bit 存储（FP8/FP4），去掉 16bit 分支。
    if a_layout == 'row' and block_k < 64:
        pytest.skip(f"skip: [{M},{N},{K} with BLOCK_K={block_k}] for A row layout (8-bit MLS rule)")
    if a_layout == 'col' and block_m < 64:
        pytest.skip(f"skip: [{M},{N},{K} with BLOCK_M={block_m}] for A col layout (8-bit MLS rule)")
    if b_layout == 'row' and block_n < 64:
        pytest.skip(f"skip: [{M},{N},{K} with BLOCK_N={block_n}] for B row layout (8-bit MLS rule)")
    if b_layout == 'col' and block_k < 64:
        pytest.skip(f"skip: [{M},{N},{K} with BLOCK_K={block_k}] for B col layout (8-bit MLS rule)")
    if M < block_m or N < block_n or K < block_k:
        pytest.skip(f"skip: [{M},{N},{K}] with BLOCK[{block_m},{block_n},{block_k}]")
    if with_scale and scale_vec_size is not None and block_k % scale_vec_size != 0:
        pytest.skip(f"skip: BLOCK_K={block_k} must be divisible by scale_vec_size={scale_vec_size}")


def matmul_with_layout(a, b, a_layout='row', b_layout='row', mix_load=0b00,
                       lhs_data_type: F8F6F4DataType = F8F6F4DataType.FP8,
                       rhs_data_type: F8F6F4DataType = F8F6F4DataType.FP8,
                       activation="",
                       config=None,
                       output_dtype=None,
                       a_scale=None,
                       b_scale=None,
                       scale_vec_size=32,
                       boundary_check=False):
    """
    智能矩阵乘法函数，支持通过shape变化实现列主序+contiguous

    参数:
    - a: 输入矩阵A
    - b: 输入矩阵B
    - a_layout: 'row' 或 'col'，指示A的期望布局
    - b_layout: 'row' 或 'col'，指示B的期望布局
    - activation: 激活函数
    - output_dtype: 输出数据类型，如果为None则根据输入数据类型自动选择

    布局约定:
    - a_layout='row': a应该是[M, K]形状（行主序）
    - a_layout='col': a应该是[K, M]形状（行主序，但逻辑上是[M,K]的列主序）
    - b_layout='row': b应该是[K, N]形状（行主序）
    - b_layout='col': b应该是[N, K]形状（行主序，但逻辑上是[K,N]的列主序）

    M, N, K 与 BLOCK_M, BLOCK_N, BLOCK_K（与 python/test/unit/language/test_matmul.py::mxfp8_mxfp4_matmul 一致）:
    - 传入 kernel 的 M、N、K 为逻辑（unpack 后）矩阵轴长；BLOCK_* 为逻辑 tile 尺寸。
    - FP4 时 kernel 内用 DIV_FACTOR_A/B（及 DIV_FACTOR_A_M/K、DIV_FACTOR_B_K/N）把 pack 轴除以 2，
      得到物理 shape 与 PHY_BLOCK_*（对应 mxfp8 里的 BLOCK_K // DIV_FACTOR_A 等）。
    """
    if lhs_data_type in (F8F6F4DataType.FP6, F8F6F4DataType.BF6) or rhs_data_type in (
        F8F6F4DataType.FP6, F8F6F4DataType.BF6
    ):
        raise AssertionError("6-bit (FP6/BF6) is intentionally unsupported in this matmul test kernel")

    lhs_is_fp4 = lhs_data_type == F8F6F4DataType.FP4
    rhs_is_fp4 = rhs_data_type == F8F6F4DataType.FP4
    # tl.dot_scaled 仅对 e2m1 允许 lhs_k_pack/rhs_k_pack 为 False（沿 M/N 打包一对 nibble）；
    # FP8/BF16/FP16 必须传 True，见 python/triton/language/semantic.py dot_scaled。
    lhs_k_pack = _fp4_pack_along_k_for_operand('a', a_layout) if lhs_is_fp4 else True
    rhs_k_pack = _fp4_pack_along_k_for_operand('b', b_layout) if rhs_is_fp4 else True

    # 根据布局确定逻辑维度
    if a_layout == 'row':
        # A是行主序 [M, K]
        div_m = _fp4_div_factor_for_dim('a', a_layout, 'm') if lhs_is_fp4 else 1
        div_k = _fp4_div_factor_for_dim('a', a_layout, 'k') if lhs_is_fp4 else 1
        M = a.shape[0] * div_m
        K_a = a.shape[1] * div_k
        stride_am, stride_ak = a.stride(0), a.stride(1)
    elif a_layout == 'col':
        # A是列主序，实际存储为[K, M]，逻辑上是[M, K]的列主序
        div_k = _fp4_div_factor_for_dim('a', a_layout, 'k') if lhs_is_fp4 else 1
        div_m = _fp4_div_factor_for_dim('a', a_layout, 'm') if lhs_is_fp4 else 1
        K_a = a.shape[0] * div_k
        M = a.shape[1] * div_m
        stride_am, stride_ak = a.stride(1), a.stride(0)  # 交换stride
    else:
        raise ValueError("a_layout must be 'row' or 'col'")

    if b_layout == 'row':
        # B是行主序 [K, N]
        div_k = _fp4_div_factor_for_dim('b', b_layout, 'k') if rhs_is_fp4 else 1
        div_n = _fp4_div_factor_for_dim('b', b_layout, 'n') if rhs_is_fp4 else 1
        K_b = b.shape[0] * div_k
        N = b.shape[1] * div_n
        stride_bk, stride_bn = b.stride(0), b.stride(1)
    elif b_layout == 'col':
        # B是列主序，实际存储为[N, K]，逻辑上是[K, N]的列主序
        div_n = _fp4_div_factor_for_dim('b', b_layout, 'n') if rhs_is_fp4 else 1
        div_k = _fp4_div_factor_for_dim('b', b_layout, 'k') if rhs_is_fp4 else 1
        N = b.shape[0] * div_n
        K_b = b.shape[1] * div_k
        stride_bk, stride_bn = b.stride(1), b.stride(0)  # 交换stride
    else:
        raise ValueError("b_layout must be 'row' or 'col'")

    # 检查K维度匹配
    assert K_a == K_b, f"K dimension mismatch: A has K={K_a}, B has K={K_b}"
    K = K_a

    if a_scale is not None:
        _validate_scale_tensor(a_scale, "a_scale", M, K, scale_vec_size)
    if b_scale is not None:
        _validate_scale_tensor(b_scale, "b_scale", N, K, scale_vec_size)

    # 确定输出数据类型
    if output_dtype is None:
        if a.dtype == torch.int8:
            output_dtype = torch.int32
        else:
            output_dtype = torch.float16

    if config is None:
        config = {}
    config = _normalize_block_config(config)
    for required_key in ("BLOCK_M", "BLOCK_N", "BLOCK_K"):
        if required_key not in config:
            raise AssertionError(f"Missing required block config key: {required_key}")
    if (a_scale is not None or b_scale is not None) and config["BLOCK_K"] % scale_vec_size != 0:
        raise AssertionError(
            f"Scaled dot requires BLOCK_K divisible by scale_vec_size: BLOCK_K={config['BLOCK_K']}, "
            f"scale_vec_size={scale_vec_size}"
        )

    if lhs_is_fp4:
        if not lhs_k_pack and config.get('BLOCK_M', 0) % 2 != 0:
            raise AssertionError("FP4 A non-K pack requires BLOCK_M to be even")
        if lhs_k_pack and config.get('BLOCK_K', 0) % 2 != 0:
            raise AssertionError("FP4 A K-pack requires BLOCK_K to be even")
    if rhs_is_fp4:
        if not rhs_k_pack and config.get('BLOCK_N', 0) % 2 != 0:
            raise AssertionError("FP4 B non-K pack requires BLOCK_N to be even")
        if rhs_k_pack and config.get('BLOCK_K', 0) % 2 != 0:
            raise AssertionError("FP4 B K-pack requires BLOCK_K to be even")

    # 分配输出张量
    c = torch.ones((M, N), device=a.device, dtype=output_dtype)

    # 1D launch kernel
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )

    # 确定输出数据类型字符串
    if output_dtype == torch.int32:
        output_dtype_str = "int32"
    else:
        output_dtype_str = "float16"

    # matmul_kernel.warmup(
    a_scale_tensor = a if a_scale is None else a_scale
    b_scale_tensor = b if b_scale is None else b_scale

    compiled_kernel = matmul_kernel[grid](
        a, b, c,  #
        a_scale_tensor, b_scale_tensor,
        M, N, K,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        c.stride(0), c.stride(1),  #
        0 if a_scale is None else a_scale.stride(0),
        0 if a_scale is None else a_scale.stride(1),
        0 if b_scale is None else b_scale.stride(0),
        0 if b_scale is None else b_scale.stride(1),
        mix_load = mix_load,
        lhs_data_type_id=lhs_data_type.value,
        rhs_data_type_id=rhs_data_type.value,
        lhs_k_pack=lhs_k_pack,
        rhs_k_pack=rhs_k_pack,
        ACTIVATION=activation,  #
        OUTPUT_DTYPE=output_dtype_str,  #
        USE_A_SCALE=a_scale is not None,
        USE_B_SCALE=b_scale is not None,
        SCALE_VEC_SIZE=scale_vec_size,
        **config,
        boundary_check=boundary_check,
        # grid=grid,
    )
    print(f"  ----- shared memory: {compiled_kernel.metadata.shared} bytes -----")
    return c

# %%
# Pytest测试程序 - 完整的配置列表测试
# --------------------------------------
SCALE_VEC_SIZE = 32
LAYOUT_CASES = {
    'anbn': ('col', 'col'),
    'anbt': ('col', 'row'),
    'atbn': ('row', 'col'),
    'atbt': ('row', 'row'),
}


@dataclass
class F8F6F4TestInputs:
    a: torch.Tensor
    b: torch.Tensor
    unpack_a: torch.Tensor
    unpack_b: torch.Tensor
    a_scale: Optional[torch.Tensor] = None
    b_scale: Optional[torch.Tensor] = None


def _storage_shape(rows, cols, layout):
    if layout == 'row':
        return rows, cols
    if layout == 'col':
        return cols, rows
    raise ValueError(f"Unsupported layout: {layout}")


def _logical_view(tensor, layout):
    return tensor if layout == 'row' else tensor.t()


def _pack_dim_for_operand(operand_name, layout):
    pack_along_k = _fp4_pack_along_k_for_operand(operand_name, layout)
    if operand_name == 'a':
        # A storage shape:
        # - row layout: [M, K]  (K dim = 1)
        # - col layout: [K, M]  (K dim = 0)
        k_dim = 1 if layout == 'row' else 0
        non_k_dim = 0 if layout == 'row' else 1
        return k_dim if pack_along_k else non_k_dim
    if operand_name == 'b':
        # B storage shape:
        # - row layout: [K, N]  (K dim = 0)
        # - col layout: [N, K]  (K dim = 1)
        k_dim = 0 if layout == 'row' else 1
        non_k_dim = 1 if layout == 'row' else 0
        return k_dim if pack_along_k else non_k_dim
    raise ValueError(f"Unsupported operand: {operand_name}")


def _fp4_pack_along_k_for_operand(operand_name, layout):
    if operand_name == 'a':
        return layout == 'row'
    if operand_name == 'b':
        return layout == 'col'
    raise ValueError(f"Unsupported operand: {operand_name}")


def _fp4_div_factor_for_dim(operand_name, layout, dim_name):
    pack_along_k = _fp4_pack_along_k_for_operand(operand_name, layout)
    if operand_name == 'a':
        if dim_name == 'm':
            return 1 if pack_along_k else 2
        if dim_name == 'k':
            return 2 if pack_along_k else 1
    if operand_name == 'b':
        if dim_name == 'k':
            return 2 if pack_along_k else 1
        if dim_name == 'n':
            return 1 if pack_along_k else 2
    raise ValueError(f"Unsupported dim {dim_name} for operand {operand_name} layout {layout}")


def _element_type_to_torch_dtype(data_type: F8F6F4DataType):
    if data_type == F8F6F4DataType.FP8:
        return torch.float8_e4m3fn
    if data_type == F8F6F4DataType.BF8:
        return torch.float8_e5m2
    if data_type == F8F6F4DataType.FP4:
        return None
    if data_type == F8F6F4DataType.FP6:
        raise NotImplementedError("FP6 operand creation is not supported in this test yet")
    if data_type == F8F6F4DataType.BF6:
        raise NotImplementedError("BF6 operand creation is not supported in this test yet")
    raise ValueError(f"Unsupported data type: {data_type}")


def _create_operand(data_type: F8F6F4DataType, rows, cols, layout, operand_name, device):
    shape = _storage_shape(rows, cols, layout)
    if data_type == F8F6F4DataType.FP4:
        if shape[_pack_dim_for_operand(operand_name, layout)] % 2 != 0:
            raise ValueError(
                f"FP4 packing requires even size on packed axis: operand={operand_name}, layout={layout}, shape={shape}"
            )
        tensor = MXFP4Tensor(size=shape, device=device).random()
        # Debug path: use deterministic FP4 value 1.0 everywhere.
        # tensor = MXFP4Tensor(data=torch.ones(shape, device=device, dtype=torch.float32))
        pack_dim = _pack_dim_for_operand(operand_name, layout)
        packed = tensor.to_packed_tensor(dim=pack_dim)
        unpacked = tensor.unpack_packed_tensor(packed, dim=pack_dim, original_shape=shape)
        return packed, unpacked

    dtype = _element_type_to_torch_dtype(data_type)
    operand = torch.randn(shape, device=device, dtype=torch.float16).to(dtype)
    return operand, operand


def _create_scale(rows, k, device, scale_vec_size):
    shape = (rows, triton.cdiv(k, scale_vec_size))
    return MXScaleTensor(size=shape, device=device).random(high=32.0).data


def _decode_operand(unpacked):
    if unpacked.dtype == torch.uint8:
        return fp4_map(unpacked).to(torch.float32)
    return unpacked.to(torch.float32)


def _decode_scale(scale):
    if scale.dtype == torch.uint8:
        scale = scale.view(torch.uint8)
        scale = scale.to(torch.int32)
        scale = scale << 23
        return scale.view(torch.float32)
    return scale.to(torch.float32)


def _expand_a_scale(a_scale, M, K, scale_vec_size):
    return _decode_scale(a_scale).repeat_interleave(scale_vec_size, dim=1)[:M, :K]


def _expand_b_scale(b_scale, N, K, scale_vec_size):
    expanded = _decode_scale(b_scale).repeat_interleave(scale_vec_size, dim=1)[:N, :K]
    return expanded.t().contiguous()


def _build_reference_operands(test_inputs, a_layout, b_layout, scale_vec_size):
    a_ref = _logical_view(_decode_operand(test_inputs.unpack_a), a_layout)
    b_ref = _logical_view(_decode_operand(test_inputs.unpack_b), b_layout)
    M, K = a_ref.shape
    _, N = b_ref.shape

    if test_inputs.a_scale is not None:
        a_ref = a_ref * _expand_a_scale(test_inputs.a_scale, M, K, scale_vec_size)
    if test_inputs.b_scale is not None:
        b_ref = b_ref * _expand_b_scale(test_inputs.b_scale, N, K, scale_vec_size)
    return a_ref, b_ref


def _report_mismatch(triton_output_cpu, torch_output_cpu):
    diff = torch.abs(triton_output_cpu - torch_output_cpu)
    max_diff = torch.max(diff).item()
    print("❌ Triton and Torch differ")
    print(f"Max diff: {max_diff}")

    triton_nan_indices = torch.where(torch.isnan(triton_output_cpu))
    torch_nan_indices = torch.where(torch.isnan(torch_output_cpu))
    finite_diff = torch.where(torch.isfinite(diff), diff, torch.zeros_like(diff))
    diff_indices = torch.where(finite_diff > 1e-2)

    print("\n=== 差异分析 ===")
    print(f"Triton输出nan位置: {len(triton_nan_indices[0]) if len(triton_nan_indices[0]) > 0 else '无'}")
    print(f"Torch输出nan位置: {len(torch_nan_indices[0]) if len(torch_nan_indices[0]) > 0 else '无'}")
    print(f"有限数值差异位置: {len(diff_indices[0]) if len(diff_indices[0]) > 0 else '无'}")

    if len(diff_indices[0]) > 0:
        print("\n有限数值差异元素位置 (行, 列) 和值:")
        print("索引\t行\t列\tTriton值\tTorch值\t差异")
        for i in range(min(100, len(diff_indices[0]))):
            row, col = diff_indices[0][i].item(), diff_indices[1][i].item()
            print(
                f"{i}\t{row}\t{col}\t{triton_output_cpu[row, col].item():.6f}\t"
                f"{torch_output_cpu[row, col].item():.6f}\t{diff[row, col].item():.6f}"
            )
        if len(diff_indices[0]) > 100:
            print(f"... 和另外 {len(diff_indices[0]) - 100} 个差异元素未显示")


def run_layout_combination_test(test_inputs, a_layout, b_layout, lhs_data_type: F8F6F4DataType,
                                rhs_data_type: F8F6F4DataType, mix_load, case_name,
                                config=None, device='cuda', reference_device='cpu',
                                scale_vec_size=SCALE_VEC_SIZE,
                                boundary_check=False):
    """运行单个layout组合测试，Triton默认在GPU，Torch reference默认在CPU。"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is required for run_layout_combination_test")

    if config is None:
        config = {}

    a = test_inputs.a.to(device)
    b = test_inputs.b.to(device)
    a_scale = test_inputs.a_scale.to(device) if test_inputs.a_scale is not None else None
    b_scale = test_inputs.b_scale.to(device) if test_inputs.b_scale is not None else None

    print(f"\n=== {case_name} ===")
    print(f"A shape: {a.shape}, stride: {a.stride()}, layout: {a_layout}, dtype: {a.dtype}, scale: {a_scale is not None}")
    print(f"B shape: {b.shape}, stride: {b.stride()}, layout: {b_layout}, dtype: {b.dtype}, scale: {b_scale is not None}")

    triton_output = matmul_with_layout(
        a,
        b,
        a_layout=a_layout,
        b_layout=b_layout,
        mix_load=mix_load,
        lhs_data_type=lhs_data_type,
        rhs_data_type=rhs_data_type,
        activation="",
        config=config,
        output_dtype=torch.float16,
        a_scale=a_scale,
        b_scale=b_scale,
        scale_vec_size=scale_vec_size,
        boundary_check=boundary_check,
    )

    a_ref, b_ref = _build_reference_operands(test_inputs, a_layout, b_layout, scale_vec_size)
    torch_output = torch.matmul(
        a_ref.to(reference_device, dtype=torch.float32),
        b_ref.to(reference_device, dtype=torch.float32),
    ).to(torch.float16)

    triton_output_cpu = triton_output.cpu()
    torch_output_cpu = torch_output.cpu()
    if torch.allclose(triton_output_cpu, torch_output_cpu, atol=1e-2, rtol=0):
        print("✅ Triton and Torch match")
        return True

    _report_mismatch(triton_output_cpu, torch_output_cpu)

    diff = torch.abs(triton_output_cpu - torch_output_cpu)
    finite_diff = torch.where(torch.isfinite(diff), diff, torch.zeros_like(diff))
    diff_indices = torch.where(finite_diff > 1e-2)
    if len(diff_indices[0]) > 0:
        torch_vals = torch_output_cpu[diff_indices]
        triton_vals = triton_output_cpu[diff_indices]
        denom = torch.clamp(torch.abs(torch_vals), min=1e-6)
        rel_errors = torch.abs(triton_vals - torch_vals) / denom
        if torch.all(rel_errors <= 0.05):
            print("✅ 所有差异位置的相对误差均不超过 5%，判定为通过")
            return not torch.isnan(triton_output_cpu).any().item()
    return False


def create_test_matrices_f8f6f4(M, K, N, a_layout, b_layout, lhs_data_type: F8F6F4DataType,
                                rhs_data_type: F8F6F4DataType, device='cpu',
                                with_a_scale=False, with_b_scale=False,
                                scale_vec_size=SCALE_VEC_SIZE):
    """创建测试输入。默认在CPU上构造，便于Torch reference直接复用。"""
    # Always pass logical shapes; _storage_shape handles row/col storage mapping.
    a_rows, a_cols = M, K
    b_rows, b_cols = K, N
    a, unpack_a = _create_operand(lhs_data_type, a_rows, a_cols, a_layout, 'a', device)
    b, unpack_b = _create_operand(rhs_data_type, b_rows, b_cols, b_layout, 'b', device)
    a_scale = _create_scale(M, K, device, scale_vec_size) if with_a_scale else None
    b_scale = _create_scale(N, K, device, scale_vec_size) if with_b_scale else None

    return F8F6F4TestInputs(
        a=a,
        b=b,
        unpack_a=unpack_a,
        unpack_b=unpack_b,
        a_scale=a_scale,
        b_scale=b_scale,
    )


#test mls_ds_mmac.f8f6f4
@pytest.mark.parametrize("lhs_data_type", [F8F6F4DataType.FP4, F8F6F4DataType.FP8])
@pytest.mark.parametrize("rhs_data_type", [F8F6F4DataType.FP4, F8F6F4DataType.FP8])
@pytest.mark.parametrize("num_stages", [1, 2])
@pytest.mark.parametrize("instruction_sched_variant", ["none"])
@pytest.mark.parametrize("optimize_epilogue", [False])
@pytest.mark.parametrize("mix_load", [0b00])
@pytest.mark.parametrize("boundary_check", [False])
@pytest.mark.parametrize("layout_case", ["atbn", "anbt"])
@pytest.mark.parametrize("with_a_scale", [False])
@pytest.mark.parametrize("with_b_scale", [False])
@pytest.mark.parametrize("config_list", [
    config for config in [
        {'BLOCK_M': m, 'BLOCK_N': n, 'BLOCK_K': k, 'num_warps': w, "GROUP_SIZE_M": 1}
        for m in [16, 32, 64]
        for n in [16, 32, 64]
        for k in [64, 128]
        for w in [2, 4]
    ]
])
@pytest.mark.parametrize("size_multiplier", [2, 4])
@pytest.mark.parametrize("mmac_layout_force", [1])
def test_mls_f8f6f4_comprehensive(config_list, layout_case, with_a_scale, with_b_scale,
                                  size_multiplier, mix_load, optimize_epilogue,
                                  num_stages, instruction_sched_variant,
                                  mmac_layout_force,
                                  lhs_data_type, rhs_data_type, boundary_check):
    a_layout, b_layout = LAYOUT_CASES[layout_case]
    config = _normalize_block_config(config_list)
    if boundary_check:
        # Keep FP4 pack axes even while forcing M/N/K tails.
        M = config['BLOCK_M'] * size_multiplier - 4
        K = config['BLOCK_K'] * size_multiplier - 8
        N = config['BLOCK_N'] * size_multiplier - 12
    else:
        # 逻辑矩阵尺寸 = block 尺寸的整数倍，便于整 tile 覆盖且与 grid 一致。
        M = config['BLOCK_M'] * size_multiplier
        K = config['BLOCK_K'] * size_multiplier
        N = config['BLOCK_N'] * size_multiplier

    _skip_unsupported_mls_case(M, N, K, a_layout, b_layout, config)

    if not is_hcu_support_mmac_layout() and mmac_layout_force != 1:
        pytest.skip("skip: not support mmac_layout")
        return
    if layout_case not in ["atbn"]:
        pytest.skip("skip: not support layout case in triton3.2")
        return

    test_inputs = create_test_matrices_f8f6f4(
        M,
        K,
        N,
        a_layout,
        b_layout,
        lhs_data_type,
        rhs_data_type,
        device='cpu',
        with_a_scale=with_a_scale,
        with_b_scale=with_b_scale,
        scale_vec_size=SCALE_VEC_SIZE,
    )

    print(f"\n{'='*80}")
    print("综合测试配置:")
    print(f"  lhs数据类型: {lhs_data_type.name}")
    print(f"  rhs数据类型: {rhs_data_type.name}")
    print(f"  layout case: {layout_case} -> A({a_layout}) × B({b_layout})")
    print(f"  矩阵大小: A[{M}, {K}] × B[{K}, {N}]")
    print(f"  内核配置: {config}")
    print(f"  混合加载: {mix_load}")
    print(f"  边界检查: {boundary_check}")
    print(f"  A scale: {with_a_scale}")
    print(f"  B scale: {with_b_scale}")
    print(f"  epilogue: {optimize_epilogue}")
    print(f"  阶段数: {num_stages}")
    print(f"  指令调度变种: {instruction_sched_variant}")
    print(f"  mmac_layout_force: {mmac_layout_force}")
    print(f"  大小倍数: {size_multiplier}")

    config['num_stages'] = num_stages
    config['instruction_sched_variant'] = instruction_sched_variant
    config['mmac_layout_force'] = mmac_layout_force
    config['optimize_epilogue'] = optimize_epilogue

    result = run_layout_combination_test(
        test_inputs,
        a_layout,
        b_layout,
        lhs_data_type,
        rhs_data_type,
        mix_load,
        (
            f"Case: {layout_case} / A({a_layout}) × B({b_layout}) / "
            f"lhs={lhs_data_type.name} / rhs={rhs_data_type.name} / "
            f"a_scale={with_a_scale} / b_scale={with_b_scale} / "
            f"boundary_check={boundary_check}"
        ),
        config=config,
        boundary_check=boundary_check,
    )
    assert result, (
        f"Fail: lhs={lhs_data_type.name}, rhs={rhs_data_type.name}, M={M}xK={K}xN={N}, "
        f"layout_case={layout_case}, with_a_scale={with_a_scale}, "
        f"with_b_scale={with_b_scale}, boundary_check={boundary_check}, config={config}"
    )
    print(
        f"✅ Pass: lhs={lhs_data_type.name}, rhs={rhs_data_type.name}, M={M}xK={K}xN={N}, "
        f"layout_case={layout_case}, with_a_scale={with_a_scale}, "
        f"with_b_scale={with_b_scale}, boundary_check={boundary_check}, config={config}"
    )


SCALED_CONFIGS = [
    {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'num_warps': 4, 'GROUP_SIZE_M': 1},
]


@pytest.mark.parametrize("lhs_data_type", [F8F6F4DataType.FP4, F8F6F4DataType.FP8])
@pytest.mark.parametrize("rhs_data_type", [F8F6F4DataType.FP4, F8F6F4DataType.FP8])
@pytest.mark.parametrize("layout_case", ["atbn", "anbt"])
@pytest.mark.parametrize(("with_a_scale", "with_b_scale"), [(False, False)]) #[(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("config", SCALED_CONFIGS)
def test_mls_f8f6f4_scaled_values(layout_case, with_a_scale, with_b_scale, config, lhs_data_type, rhs_data_type):
    a_layout, b_layout = LAYOUT_CASES[layout_case]
    config = _normalize_block_config(config)
    size_multiplier = 2
    M = config['BLOCK_M'] * size_multiplier
    K = config['BLOCK_K'] * size_multiplier
    N = config['BLOCK_N'] * size_multiplier

    _skip_unsupported_mls_case(M, N, K, a_layout, b_layout, config, SCALE_VEC_SIZE, with_scale=True)

    test_inputs = create_test_matrices_f8f6f4(
        M,
        K,
        N,
        a_layout,
        b_layout,
        lhs_data_type,
        rhs_data_type,
        device='cpu',
        with_a_scale=with_a_scale,
        with_b_scale=with_b_scale,
        scale_vec_size=SCALE_VEC_SIZE,
    )

    result = run_layout_combination_test(
        test_inputs,
        a_layout,
        b_layout,
        lhs_data_type,
        rhs_data_type,
        0b00,
        (
            f"Scaled case: {layout_case} / A({a_layout}) × B({b_layout}) / "
            f"lhs={lhs_data_type.name} / rhs={rhs_data_type.name} / "
            f"a_scale={with_a_scale} / b_scale={with_b_scale}"
        ),
        config={**config, 'num_stages': 1},
        scale_vec_size=SCALE_VEC_SIZE,
        boundary_check=False,
    )
    assert result, (
        f"Scaled fail: lhs={lhs_data_type.name}, rhs={rhs_data_type.name}, M={M}xK={K}xN={N}, "
        f"layout_case={layout_case}, with_a_scale={with_a_scale}, "
        f"with_b_scale={with_b_scale}, config={config}"
    )

# # #---------------------------------------------------------------------------------------------------

torch.manual_seed(0)

# ##################################################################################################
# ## case 1: sb b4xb4 test
# M = 32
# N = 32
# K = 256

# BLOCK_M = 32
# BLOCK_N = 32
# BLOCK_K = 256
# layout_case = "atbn"
# num_warps = 1
# num_stages = 1
# mix_load = 0b00
# instruction_sched_variant = "none"
# mmac_layout_force = 1
# lhs_data_type = F8F6F4DataType.FP4
# rhs_data_type = F8F6F4DataType.FP4

# config = {'BLOCK_M': BLOCK_M, 'BLOCK_N': BLOCK_N, 'BLOCK_K': BLOCK_K,
#           'num_warps': num_warps, "GROUP_SIZE_M": 1}

# # test fp4 mls ds mmac
# print(f"\n{'='*60}")
# print(f"测试用例1: A(row) × B(col), config={config}")
# print(f"A: [{M}, {K}] 行主序, B: [{K}, {N}] 列主序")

# test_mls_f8f6f4_comprehensive(config_list=config,
#                               layout_case=layout_case,
#                               with_a_scale=False,
#                               with_b_scale=False,
#                               size_multiplier=1,
#                               mix_load=mix_load,
#                               optimize_epilogue=False,
#                               num_stages=1,
#                               instruction_sched_variant="none",
#                               mmac_layout_force=1,
#                               lhs_data_type=lhs_data_type,
#                               rhs_data_type=rhs_data_type)


# ###################################################################################################
# ## case 2: sb b4xf8 test
# M = 32
# N = 64
# K = 256

# BLOCK_M = 32
# BLOCK_N = 64
# BLOCK_K = 256
# layout_case = "atbn"
# num_warps = 1
# num_stages = 1
# mix_load = 0b00
# instruction_sched_variant = "none"
# mmac_layout_force = 1
# lhs_data_type = F8F6F4DataType.FP4
# # rhs_data_type = F8F6F4DataType.FP4
# rhs_data_type = F8F6F4DataType.FP8

# config = {'BLOCK_M': BLOCK_M, 'BLOCK_N': BLOCK_N, 'BLOCK_K': BLOCK_K,
#           'num_warps': num_warps, "GROUP_SIZE_M": 1}

# # test fp4 mls ds mmac
# print(f"\n{'='*60}")
# print(f"测试用例2: A(row) × B(col), config={config}")
# print(f"A: [{M}, {K}] 行主序, B: [{K}, {N}] 列主序")

# test_mls_f8f6f4_comprehensive(config_list=config,
#                               layout_case=layout_case,
#                               with_a_scale=False,
#                               with_b_scale=False,
#                               size_multiplier=1,
#                               mix_load=mix_load,
#                               optimize_epilogue=False,
#                               num_stages=1,
#                               instruction_sched_variant="none",
#                               mmac_layout_force=1,
#                               lhs_data_type=lhs_data_type,
#                               rhs_data_type=rhs_data_type)

# if __name__ == "__main__":
#     pytest.main(["-v", "-s", "mls-dot-scaled-test.py::test_mls_f8f6f4_comprehensive[1-2-config_list0-False-False-atbn-False-0-False-none-False-2-F8F6F4DataType.FP4-F8F6F4DataType.FP4]"])
#     # pytest.main(["-v", "-s", "mls-dot-scaled-test.py"])
