import torch
from enum import Enum

import triton
import triton.language as tl
import triton.testing

import os
os.environ["AMDGCN_USE_BUFFER_OPS"] = "1"


def is_hcu_support_mls():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == 'hip' and (target.arch == 'gfx938' or target.arch == 'gfx946' or target.arch == 'gfx92a')


regression_run = True

def get_autotune_config():
    return [
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
            num_warps=4, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
            num_warps=8, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
            num_warps=8, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
            num_warps=8, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
            num_warps=4, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
            num_warps=4, num_stages=2),
        # triton.Config(
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
            num_warps=4, num_stages=2),
        # triton.Config(
        #     {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'waves_per_eu': 3},
        #     num_warps=4, num_stages=2),  # <E8><BF><99><E4><B8><AA> config<E4><BC><9A><E6><8E><89><E5><8D><A1>, <E5><9C><A8> (2048, 7168, 256) <E6><97><B6><E4><BC><9A><E6><8E><89><E5><<8D><A1><EF><BC><8C> <E9><9C><80><E8><A6><81><E6><94><B9><E7><94><B5><E5><8E><8B><E5><92><8C><E9><A2><91><E7><8E><87> <E6><89><8D><E6><AD><A3><E5><B8><B8><E3><80><82>

        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'waves_per_eu': 3},
            num_warps=4, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'waves_per_eu': 3},
            num_warps=4, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'waves_per_eu': 4},
            num_warps=4, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'waves_per_eu': 4},
            num_warps=4, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'waves_per_eu': 4},
            num_warps=2, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'waves_per_eu': 4},
            num_warps=4, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'waves_per_eu': 4},
            num_warps=4, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'waves_per_eu': 4},
            num_warps=4, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'waves_per_eu': 4},
            num_warps=4, num_stages=2),
    ]

    configs = []

    block_sizes = [16, 32, 64, 128, 256]
    num_warps_list = [4, 8]
    num_stages_list = [1, 2]

    for block_m in block_sizes:
        for block_n in block_sizes:
            for block_k in block_sizes:
                for num_warps in num_warps_list:
                    for num_stages in num_stages_list:
                        configs.append(
                            triton.Config(
                                {
                                    'BLOCK_SIZE_M': block_m,
                                    'BLOCK_SIZE_N': block_n,
                                    'BLOCK_SIZE_K': block_k,
                                    'GROUP_SIZE_M': 1,
                                    'waves_per_eu': 1,
                                },
                                num_warps=num_warps,
                                num_stages=num_stages
                            )
                        )

    return configs


@triton.jit
def matrix_load_store_kernel(
        a_ptr, c_ptr,
        M, K,
        stride_am, stride_ak,
        stride_cm, stride_ck,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        boundary_check: tl.constexpr,
        use_mask: tl.constexpr):
    """Copy a tile from matrix A into matrix C using tl.matrix_load."""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    tile_m = pid_m * BLOCK_SIZE_M
    tile_k = pid_k * BLOCK_SIZE_K

    # 计算 offsets（标量形式，与 matmul_kernel 一致）
    mls_offs_m = tile_m
    mls_offs_k = tile_k

    offs_m = tile_m + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tile_k + tl.arange(0, BLOCK_SIZE_K)


    # # 使用 tl.load，需要构造指针数组
    # a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    # a_tile = tl.load(a_ptrs)
    if boundary_check:
        a_tile = tl.matrix_load(
            a_ptr,
            shape=[M, K],
            strides=[stride_am, stride_ak],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
            offsets=[mls_offs_m, mls_offs_k],
            boundary_check=(0, 1))
    else:
        a_tile = tl.matrix_load(
            a_ptr,
            shape=[M, K],
            strides=[stride_am, stride_ak],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
            offsets=[mls_offs_m, mls_offs_k])
    # tl.device_print("a_tile_matrix_load", a_tile)

    if use_mask:
        mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a_tile = tl.where(mask, a_tile, 0) # use 0 (not 0.0) to let triton to infer its type.
    # tl.device_print("a_tile_after_mask", a_tile)


    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_k[None, :] * stride_ck
    mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    tl.store(c_ptrs, a_tile, mask=mask)


def run_matrix_load_store_test(a, a_layout, boundary_check=False, use_mask=False, config=None):
    """
    参数:
    - a: 输入矩阵A
    - a_layout: 'row' 或 'col'，指示A的期望布局

    布局约定:
    - a_layout='row': a应该是[M, K]形状（行主序）
    - a_layout='col': a应该是[K, M]形状（行主序，但逻辑上是[M,K]的列主序）
    """

    # 根据布局确定逻辑维度
    if a_layout == 'row':
        # A是行主序 [M, K]
        M, K_a = a.shape
        stride_am, stride_ak = a.stride(0), a.stride(1)
    elif a_layout == 'col':
        # A是列主序，实际存储为[K, M]，逻辑上是[M, K]的列主序
        K_a, M = a.shape
        stride_am, stride_ak = a.stride(1), a.stride(0)  # 交换stride
    else:
        raise ValueError("a_layout must be 'row' or 'col'")
    if config is None:
        config = {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_K': 32}

    # 分配输出张量
    if a_layout == 'row':
        c = torch.empty((M, K_a), device=a.device, dtype=a.dtype)
        stride_cm, stride_ck = c.stride(0), c.stride(1)
    elif a_layout == 'col':
        c = torch.empty((K_a, M), device=a.device, dtype=a.dtype)
        stride_cm, stride_ck = c.stride(1), c.stride(0)
    else:
        raise ValueError("a_layout must be 'row' or 'col'")

    # 2D launch kernel (使用 lambda META 形式，与 matmul_with_layout 一致)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(K_a, META['BLOCK_SIZE_K']))

    matrix_load_store_kernel[grid](
        a, c,
        M, K_a,
        stride_am, stride_ak,
        stride_cm, stride_ck,
        boundary_check=boundary_check,
        use_mask=use_mask,
        **config,
    )

    if torch.allclose(c.cpu(), a.cpu(), atol=1e-3, rtol=0):
        print("✅ matrix_load_store_kernel: Copy matches reference")
        return True
    else:
        diff_tensor = torch.abs(c - a)
        max_diff = torch.max(diff_tensor).item()

        print(f"最大差异: {max_diff}")

        # 检查有限数值的差异
        finite_diff = torch.where(torch.isfinite(diff_tensor), diff_tensor, torch.zeros_like(diff_tensor))
        diff_indices = torch.where(finite_diff > 1e-4)

        # 打印有限数值的差异
        if len(diff_indices[0]) > 0:
            print("\n有限数值差异元素位置 (行, 列) 和值:")
            print("索引\t行\t列\t原始值\t复制值\t差异")
            num_diff = min(100, len(diff_indices[0]))
            for i in range(num_diff):  # 最多打印100个差异
                row = int(diff_indices[0][i].item())
                col = int(diff_indices[1][i].item())
                a_val = a[row, col].item()
                c_val = c[row, col].item()
                diff_val = diff_tensor[row, col].item()
                print(f"{i}\t{row}\t{col}\t{a_val:.6f}\t{c_val:.6f}\t{diff_val:.6f}")

            if len(diff_indices[0]) > 100:
                print(f"... 和另外 {len(diff_indices[0]) - 100} 个差异元素未显示")
        else:
            print("没有找到有限数值的差异元素")
        return False



####################################################################################################


def prune_configs(configs, nargs, **kwargs):
    # Get bitwidth from nargs (dtype_bits parameter) or default to 16
    bitwidth = nargs.get('dtype_bits', 16)

    def _prune(config):
        _config = config.all_kwargs()
        if _config["BLOCK_SIZE_M"] > nargs["M"]  or _config["BLOCK_SIZE_K"]  > nargs["K"] or _config["BLOCK_SIZE_N"]  > nargs["N"]:
            return True

        # Determine threshold based on bitwidth
        if bitwidth == 8:
            threshold = 64
        else:  # 16bit or default
            threshold = 32

        # Check matrix A constraints
        a_is_row_major = nargs.get("a_is_row_major", True)
        if a_is_row_major:
            if _config["BLOCK_SIZE_K"] < threshold:
                return True
        else:
            if _config["BLOCK_SIZE_M"] < threshold:
                return True

        # Check matrix B constraints
        b_is_row_major = nargs.get("b_is_row_major", True)
        if b_is_row_major:
            if _config["BLOCK_SIZE_N"] < threshold:
                return True
        else:
            if _config["BLOCK_SIZE_K"] < threshold:
                return True

        return False
    return [c for c in configs if not _prune(c)]

# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
# @triton.autotune(
#     configs=get_autotune_config(),
#     key=['M', 'N', 'K', 'stride_am', 'stride_ak', 'stride_bk', 'stride_bn', 'block_k_diviable', 'mix_load',
#          'boundary_check', 'use_mask', 'dtype_bits', 'a_is_row_major', 'b_is_row_major'],
#     prune_configs_by={"early_config_prune": prune_configs},
#     perf_debug=True,
# )
@triton.heuristics({
    'block_k_diviable': lambda nargs: nargs['K'] % nargs['BLOCK_SIZE_K'] == 0,
    'block_m_diviable': lambda nargs: nargs['M'] % nargs['BLOCK_SIZE_M'] == 0,
    'block_n_diviable': lambda nargs: nargs['N'] % nargs['BLOCK_SIZE_N'] == 0,
})
@triton.jit
def matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        block_k_diviable: tl.constexpr,
        block_m_diviable: tl.constexpr,
        block_n_diviable: tl.constexpr,
        a_is_row_major: tl.constexpr,
        b_is_row_major: tl.constexpr,
        # mix_load:
        # 0b00: dont't mix, both a and b use matrix_load
        # 0b01: a: matrix_load, b: tt_load
        # 0b10: a: tt_load, b: matrix_load
        # 0b11: a: tt_load, b: tt_load
        mix_load: tl.constexpr,
        boundary_check: tl.constexpr,   # force boundary check ?
        use_mask: tl.constexpr,         # use mask (tl.where)?
        dtype_bits: tl.constexpr,       # data type bitwidth (8 or 16)
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        ACTIVATION: tl.constexpr,  #
        OUTPUT_DTYPE: tl.constexpr  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(M > 0)
    tl.assume(N > 0)
    tl.assume(K > 0) # ENote: important, it affect the global_load coalescing for prologue of tt.load

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    if GROUP_SIZE_M == 1:
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
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
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M) % M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = (pid_n * BLOCK_SIZE_N) % N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    offs_am_base = offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    offs_bn_base = offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    a_ptrs = a_ptr + offs_am_base
    b_ptrs = b_ptr + offs_bn_base

    mls_offs_am = (pid_m * BLOCK_SIZE_M) % M
    mls_offs_bn = (pid_n * BLOCK_SIZE_N) % N
    mls_offs_k  = 0

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    if OUTPUT_DTYPE == "int32":
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


    if boundary_check:
        tl.static_assert(use_mask == False, "boundary_check and use_mask cannot be True at the same time")
        boundary_check_a : tl.constexpr = (0, 1)
    elif use_mask:
        boundary_check_a : tl.constexpr = ()
    elif (not block_m_diviable) and (not block_k_diviable):
        boundary_check_a : tl.constexpr = (0, 1)
    elif not block_m_diviable:
        boundary_check_a : tl.constexpr = (0,)
    elif not block_k_diviable:
        boundary_check_a : tl.constexpr = (1,)
    else:
        boundary_check_a : tl.constexpr = ()

    # B: [K, N]，0 对应 K，1 对应 N
    if boundary_check:
        boundary_check_b : tl.constexpr = (0, 1)
    elif use_mask:
        boundary_check_b : tl.constexpr = ()
    elif (not block_k_diviable) and (not block_n_diviable):
        boundary_check_b : tl.constexpr = (0, 1)
    elif not block_k_diviable:
        boundary_check_b : tl.constexpr = (0,)
    elif not block_n_diviable:
        boundary_check_b : tl.constexpr = (1,)
    else:
        boundary_check_b : tl.constexpr = ()

    mask_m = None
    if not block_m_diviable:
        mask_m = offs_am[:, None] < M

    mask_n = None
    if not block_n_diviable:
        mask_n = offs_bn[None, :] < N

    # 定义是否需要 mask 的 constexpr 标志
    need_mask_a: tl.constexpr = (not block_m_diviable) or (not block_k_diviable)
    need_mask_b: tl.constexpr = (not block_n_diviable) or (not block_k_diviable)

    # ENote 1: this iters can make prologue of tt.load can still gen coalescing of load for buffer_load disable case, due to we "tlassume(K > 0)".
    # compared with "for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):", you need "tl.assume((K + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K > 0)" if you
    # also want to make prologue of tt.load can still gen coalescing of load.
    for k in range(0, K, BLOCK_SIZE_K):
        mls_offs_k = k

        if not block_k_diviable:
            mask_k_a = offs_k[None, :] + k < K      # [1, BLOCK_SIZE_K]
            mask_k_b = offs_k[:, None] + k < K      # [BLOCK_SIZE_K, 1]

        # 组合 mask_a 和 mls_mask_a
        if not block_k_diviable:
            if not block_m_diviable:
                mask_a = mask_m & mask_k_a
            else:
                mask_a = mask_k_a
        else:
            mask_a = mask_m

        # 组合 mask_b 和 mls_mask_b
        if not block_k_diviable:
            if not block_n_diviable:
                mask_b = mask_n & mask_k_b
            else:
                mask_b = mask_k_b
        else:
            mask_b = mask_n

        # 根据 mix_load 加载数据
        if mix_load == 0b00:
            a = tl.matrix_load(a_ptr, shape=[M, K], strides=[stride_am, stride_ak],
                               block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
                               offsets=[mls_offs_am, mls_offs_k],
                               boundary_check=boundary_check_a, mask=mask_a if use_mask else None)
            b = tl.matrix_load(b_ptr, shape=[K, N], strides=[stride_bk, stride_bn],
                               block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
                               offsets=[mls_offs_k, mls_offs_bn],
                               boundary_check=boundary_check_b, mask=mask_b if use_mask else None)
        elif mix_load == 0b01:
            a = tl.matrix_load(a_ptr, shape=[M, K], strides=[stride_am, stride_ak],
                               block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
                               offsets=[mls_offs_am, mls_offs_k],
                               boundary_check=boundary_check_a, mask=mask_a if use_mask else None)
            if need_mask_b:
                b = tl.load(b_ptrs, mask=mask_b, other=0.0)
            else:
                b = tl.load(b_ptrs)
        elif mix_load == 0b10:
            if need_mask_a:
                a = tl.load(a_ptrs, mask=mask_a, other=0.0)
            else:
                a = tl.load(a_ptrs)
            b = tl.matrix_load(b_ptr, shape=[K, N], strides=[stride_bk, stride_bn],
                               block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
                               offsets=[mls_offs_k, mls_offs_bn],
                               boundary_check=boundary_check_b, mask=mask_b if use_mask else None)
        elif mix_load == 0b11:
            if need_mask_a:
                # mask_a = tl.max_constancy(mask_a, [1, 16])
                a = tl.load(a_ptrs, mask=mask_a, other=0.0)
            else:
                a = tl.load(a_ptrs)
            if need_mask_b:
                # mask_b = tl.max_constancy(mask_b, [16, 1])
                b = tl.load(b_ptrs, mask=mask_b, other=0.0)
            else:
                b = tl.load(b_ptrs)

        # We accumulate along the K dimension.
        if OUTPUT_DTYPE == "int32":
            accumulator = tl.dot(a, b, accumulator, out_dtype=tl.int32)
        else:
            accumulator = tl.dot(a, b, accumulator)

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
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
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]

    need_mask_c: tl.constexpr = (not block_m_diviable) or (not block_n_diviable)
    if need_mask_c:
        if not block_m_diviable:
            if not block_n_diviable:
                c_mask = mask_m & mask_n
            else:
                c_mask = mask_m
        else:
            if not block_n_diviable:
                c_mask = mask_n
            else:
                c_mask = None
    else:
        c_mask = None
    tl.store(c_ptrs, c, mask=c_mask)

# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `matmul_kernel`.
@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


# %%
# We can now create a convenience wrapper function that only takes two input tensors,
# and (1) checks any shape constraint; (2) allocates the output; (3) launches the above kernel.

def matmul_with_layout(a, b, a_layout='row', b_layout='row', mix_load=0b00, boundary_check=False,
                       use_mask=False, activation="", config=None, output_dtype=None):
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
    """

    # 根据布局确定逻辑维度
    if a_layout == 'row':
        # A是行主序 [M, K]
        M, K_a = a.shape
        stride_am, stride_ak = a.stride(0), a.stride(1)
    elif a_layout == 'col':
        # A是列主序，实际存储为[K, M]，逻辑上是[M, K]的列主序
        K_a, M = a.shape
        stride_am, stride_ak = a.stride(1), a.stride(0)  # 交换stride
    else:
        raise ValueError("a_layout must be 'row' or 'col'")

    if b_layout == 'row':
        # B是行主序 [K, N]
        K_b, N = b.shape
        stride_bk, stride_bn = b.stride(0), b.stride(1)
    elif b_layout == 'col':
        # B是列主序，实际存储为[N, K]，逻辑上是[K, N]的列主序
        N, K_b = b.shape
        stride_bk, stride_bn = b.stride(1), b.stride(0)  # 交换stride
    else:
        raise ValueError("b_layout must be 'row' or 'col'")

    # 检查K维度匹配
    assert K_a == K_b, f"K dimension mismatch: A has K={K_a}, B has K={K_b}"
    K = K_a

    # 确定输出数据类型
    if output_dtype is None:
        if a.dtype == torch.int8:
            output_dtype = torch.int32
        else:
            output_dtype = torch.float16

    # 分配输出张量
    c = torch.empty((M, N), device=a.device, dtype=output_dtype)

    # 1D launch kernel
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )

    # 确定输出数据类型字符串
    if output_dtype == torch.int32:
        output_dtype_str = "int32"
    else:
        output_dtype_str = "float16"

    # 获取输入数据类型的 bitwidth
    try:
        dtype_bits = torch.finfo(a.dtype).bits
    except Exception:
        dtype_bits = torch.iinfo(a.dtype).bits

    # # 为 benchmark / 测试提供一个简单的默认配置
    if config is None:
        config = {}

    matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        c.stride(0), c.stride(1),  #
        a_is_row_major = a_layout == 'row',
        b_is_row_major = b_layout == 'row',
        mix_load = mix_load,
        boundary_check = boundary_check,
        use_mask = use_mask,
        dtype_bits = dtype_bits,
        ACTIVATION=activation,  #
        OUTPUT_DTYPE=output_dtype_str,  #
        **config,
    )
    return c


# %%
# Unit Test - 用shape变化实现contiguous列主序 (全新方法)
# ---------
#
# 通过改变shape来实现列主序且contiguous的张量

def run_layout_combination_test(a, b, a_layout, b_layout, mix_load, boundary_check, use_mask, case_name="", config=None):
    """测试单个布局组合（新方法）"""
    print(f"\n=== {case_name} ===")
    print(f"A shape: {a.shape}, stride: {a.stride()}, contiguous: {a.is_contiguous()}, layout: {a_layout}, dtype: {a.dtype}")
    print(f"B shape: {b.shape}, stride: {b.stride()}, contiguous: {b.is_contiguous()}, layout: {b_layout}, dtype: {b.dtype}")

    # 使用新的matmul_with_layout函数
    if a.dtype == torch.int8:
        triton_output = matmul_with_layout(a, b, a_layout, b_layout, mix_load, boundary_check, use_mask,
                                           activation="", config=config, output_dtype=torch.int32)
    else:
        triton_output = matmul_with_layout(a, b, a_layout, b_layout, mix_load, boundary_check, use_mask,
                                           activation="", config=config)

    # 为了与torch.matmul比较，需要创建等价的张量
    if a_layout == 'row':
        a_torch = a
    else:  # a_layout == 'col'
        a_torch = a.t()  # [K,M] -> [M,K]

    if b_layout == 'row':
        b_torch = b
    else:  # b_layout == 'col'
        b_torch = b.t()  # [N,K] -> [K,N]

    # 根据输入数据类型确定torch.matmul的输出类型
    if a.dtype == torch.int8:
        # 对于int8输入，转换为int32进行计算
        a_torch_cpu = a_torch.cpu().to(torch.int32)
        b_torch_cpu = b_torch.cpu().to(torch.int32)
        torch_output = torch.matmul(a_torch_cpu, b_torch_cpu).cuda()
    elif a.dtype == torch.float8_e5m2 or a.dtype == torch.float8_e4m3fn:
        a_torch_cpu = a_torch.cpu().to(torch.float16)
        b_torch_cpu = b_torch.cpu().to(torch.float16)
        torch_output = torch.matmul(a_torch_cpu, b_torch_cpu).cuda()
    else:
        # 对于float16输入，保持float16
        torch_output = torch.matmul(a_torch.cpu(), b_torch.cpu()).cuda()

    # 比较结果
    rtol = 0
    triton_output_cpu = triton_output.cpu()
    torch_output_cpu = torch_output.cpu()
    torch.set_printoptions(profile="full")
    if torch.allclose(triton_output_cpu, torch_output_cpu, atol=1e-2, rtol=rtol):
        print("✅ Triton and Torch match")
        return True
    else:
        # print(f"triton_output_cpu={triton_output_cpu}")
        # print(f"torch_output_cpu={torch_output_cpu}")
        print("❌ Triton and Torch differ")
        print(f"Max diff: {torch.max(torch.abs(triton_output_cpu - torch_output_cpu))}")

        print("❌ Triton and Torch differ")
        diff = torch.abs(triton_output_cpu - torch_output_cpu)
        max_diff = torch.max(diff).item()
        print(f"最大差异: {max_diff}")

        # 检查是否有nan值
        triton_nan_count = torch.isnan(triton_output_cpu).sum().item()
        torch_nan_count = torch.isnan(torch_output_cpu).sum().item()
        print(f"Triton输出中nan值数量: {triton_nan_count}")
        print(f"Torch输出中nan值数量: {torch_nan_count}")

        # 找出所有差异大于阈值的元素（包括nan值）
        threshold = 1e-2 if a.dtype == torch.float16 else 1
        # 检查nan值
        triton_nan_indices = torch.where(torch.isnan(triton_output_cpu))
        torch_nan_indices = torch.where(torch.isnan(torch_output_cpu))

        # 检查有限数值的差异
        finite_diff = torch.where(torch.isfinite(diff), diff, torch.zeros_like(diff))
        diff_indices = torch.where(finite_diff > threshold)

        print(f"\n=== 差异分析 ===")
        print(f"Triton输出nan位置: {len(triton_nan_indices[0]) if len(triton_nan_indices[0]) > 0 else '无'}")
        print(f"Torch输出nan位置: {len(torch_nan_indices[0]) if len(torch_nan_indices[0]) > 0 else '无'}")
        print(f"有限数值差异位置: {len(diff_indices[0]) if len(diff_indices[0]) > 0 else '无'}")

        # 打印nan位置
        if len(triton_nan_indices[0]) > 0:
            print("\nTriton输出中的nan位置:")
            for i in range(min(100, len(triton_nan_indices[0]))):
                row, col = triton_nan_indices[0][i].item(), triton_nan_indices[1][i].item()
                torch_val = torch_output_cpu[row, col].item()
                print(f"  [{row}, {col}]: Triton=nan, Torch={torch_val:.6f}")

        if len(torch_nan_indices[0]) > 0:
            print("\nTorch输出中的nan位置:")
            for i in range(min(100, len(torch_nan_indices[0]))):
                row, col = torch_nan_indices[0][i].item(), torch_nan_indices[1][i].item()
                triton_val = triton_output_cpu[row, col].item()
                print(f"  [{row}, {col}]: Triton={triton_val:.6f}, Torch=nan")

        # 打印有限数值的差异
        if len(diff_indices[0]) > 0:
            print("\n有限数值差异元素位置 (行, 列) 和值:")
            print("索引\t行\t列\tTriton值\tTorch值\t差异")
            for i in range(min(100, len(diff_indices[0]))):  # 最多打印100个差异
                row, col = diff_indices[0][i].item(), diff_indices[1][i].item()
                triton_val = triton_output_cpu[row, col].item()
                torch_val = torch_output_cpu[row, col].item()
                print(f"{i}\t{row}\t{col}\t{triton_val:.6f}\t{torch_val:.6f}\t{diff[row, col].item():.6f}")

            if len(diff_indices[0]) > 100:
                print(f"... 和另外 {len(diff_indices[0]) - 100} 个差异元素未显示")
        else:
            print("没有找到有限数值的差异元素")

        # 针对所有 diff_indices 做相对误差判定：若全部 <= 5% 则视为通过
        if len(diff_indices[0]) > 0:
            torch_vals = torch_output_cpu[diff_indices]
            triton_vals = triton_output_cpu[diff_indices]
            denom = torch.clamp(torch.abs(torch_vals), min=1e-6)
            rel_errors = torch.abs(triton_vals - torch_vals) / denom
            if torch.all(rel_errors <= 0.05):
                print("✅ 所有差异位置的相对误差均不超过 5%，判定为通过")
                return False if len(triton_nan_indices[0]) > 0 else True
            else:
                print("❌ 所有差异位置的相对误差均超过 5%，判定为失败")
                return False

        return True

# %%
# Pytest测试程序 - 完整的配置列表测试
# --------------------------------------
import pytest

def create_test_matrices(M, K, N, a_layout, b_layout, dtype):
    """根据布局创建测试矩阵"""
    if dtype == torch.int8:
        # 对于int8类型，使用randint生成-8到7的随机整数
        if a_layout == 'row':
            a = torch.randint(-8, 8, (M, K), device='cuda', dtype=dtype)
        else:  # col
            a = torch.randint(-8, 8, (K, M), device='cuda', dtype=dtype)

        if b_layout == 'row':
            b = torch.randint(-8, 8, (K, N), device='cuda', dtype=dtype)
        else:  # col
            b = torch.randint(-8, 8, (N, K), device='cuda', dtype=dtype)
    elif dtype == torch.float8_e5m2 or dtype == torch.float8_e4m3fn:
        if a_layout == 'row':
            a = torch.randn((M, K), device='cuda', dtype=torch.float16)
        else:  # col
            a = torch.randn((K, M), device='cuda', dtype=torch.float16)
        if b_layout == 'row':
            b = torch.randn((K, N), device='cuda', dtype=torch.float16)
        else:  # col
            b = torch.randn((N, K), device='cuda', dtype=torch.float16)
        a = a.to(dtype)
        b = b.to(dtype)
    else:
        # 对于float16类型，使用randn生成正态分布随机数
        if a_layout == 'row':
            a = torch.randn((M, K), device='cuda', dtype=dtype)
        else:  # col
            a = torch.randn((K, M), device='cuda', dtype=dtype)

        if b_layout == 'row':
            b = torch.randn((K, N), device='cuda', dtype=dtype)
        else:  # col
            b = torch.randn((N, K), device='cuda', dtype=dtype)

    return a, b


@pytest.mark.parametrize("use_mask", [False, True] if regression_run else [False, True])
@pytest.mark.parametrize("boundary_check", [False, True] if regression_run else [False, True])
@pytest.mark.parametrize("num_stages", [1, 2] if regression_run else [1, 2])
@pytest.mark.parametrize("async_copy_use_single_buffer", [True])
@pytest.mark.parametrize("schedule_hint", ["none"] if regression_run else ["none", "local-prefetch"])
@pytest.mark.parametrize("optimize_epilogue", [True] if regression_run else [False, True])
@pytest.mark.parametrize("mix_load", [0b00, 0b01] if regression_run else [0b00, 0b01, 0b10, 0b11])
@pytest.mark.parametrize("dtype", [
    torch.float16,
    torch.int8,
    torch.float8_e5m2,
    torch.float8_e4m3fn
])
@pytest.mark.parametrize("layout_combination", [
    # ('row', 'row'),
    ('row', 'col'),
    ('col', 'row'),
    # ('col', 'col')
])
@pytest.mark.parametrize("config_list", [
    # RegressionRun 模式下使用精简配置
    config for config in [
        {'BLOCK_SIZE_M': m, 'BLOCK_SIZE_N': n, 'BLOCK_SIZE_K': k, 'num_warps': w, "GROUP_SIZE_M": 1}
        for m in ([32] if regression_run else [16, 32, 64])
        for n in ([32, 64] if regression_run else [16, 32, 64])
        for k in ([64] if regression_run else [32, 64, 128, 256])
        for w in ([4] if regression_run else [1, 2, 4, 8])
    ]
])
@pytest.mark.parametrize("size_multiplier", [16] if regression_run else [8, 16])
@pytest.mark.parametrize("mmac_layout_force", [1, 4] if regression_run else [1, 4])
def test_mls_comprehensive(dtype, config_list, layout_combination,
                           size_multiplier, mix_load, optimize_epilogue,
                           num_stages, async_copy_use_single_buffer,
                           schedule_hint, mmac_layout_force,
                           boundary_check, use_mask):

    if not is_hcu_support_mls():
        pytest.skip("skip: not support mls")
        return

    print(f"config_list={config_list}")
    print(f"dtype={dtype}")

    base_size = 16

    # Note:
    # 1> from Spec, it requires the non-major dim bases to be 16-byte aligned !!!
    # 2> HW issue: for nmz, when matrix_load with mfilter|nfilter = dim size, result is incorrect.
    M = base_size * size_multiplier + (-4 if (boundary_check or use_mask) else 0)
    K = base_size * size_multiplier * 2 + (-8 if (boundary_check or use_mask) else 0)
    N = base_size * size_multiplier + (-12 if (boundary_check or use_mask) else 0)

    # 检查矩阵尺寸是否小于block尺寸，如果是则跳过测试
    BLOCK_SIZE_M = config_list['BLOCK_SIZE_M']
    BLOCK_SIZE_N = config_list['BLOCK_SIZE_N']
    BLOCK_SIZE_K = config_list['BLOCK_SIZE_K']

    # 通过位宽判断 dtype（优先尝试浮点，其次整数）
    try:
        dtype_bits = torch.finfo(dtype).bits
    except Exception:
        dtype_bits = torch.iinfo(dtype).bits

    if (layout_combination[0] == 'row'):
        if (dtype_bits == 8 and BLOCK_SIZE_K < 64) or (dtype_bits == 16 and BLOCK_SIZE_K < 32):
            pytest.skip(f"skip: [{M},{N},{K} with BLOCK_SIZE_K={BLOCK_SIZE_K}] for row layout")
    if (layout_combination[0] == 'col'):
        if (dtype_bits == 8 and BLOCK_SIZE_M < 64) or (dtype_bits == 16 and BLOCK_SIZE_M < 32):
            pytest.skip(f"skip: [{M},{N},{K} with BLOCK_SIZE_M={BLOCK_SIZE_M}] for col layout")
    if (layout_combination[1] == 'row'):
        if (dtype_bits == 8 and BLOCK_SIZE_N < 64) or (dtype_bits == 16 and BLOCK_SIZE_N < 32):
            pytest.skip(f"skip: [{M},{N},{K} with BLOCK_SIZE_N={BLOCK_SIZE_N}] for row layout")
    if (layout_combination[1] == 'col'):
        if (dtype_bits == 8 and BLOCK_SIZE_K < 64) or (dtype_bits == 16 and BLOCK_SIZE_K < 32):
            pytest.skip(f"skip: [{M},{N},{K} with BLOCK_SIZE_K={BLOCK_SIZE_K}] for col layout")

    if M < BLOCK_SIZE_M or N < BLOCK_SIZE_N or K < BLOCK_SIZE_K:
        pytest.skip(f"skip: [{M},{N},{K}] with BLOCK[{BLOCK_SIZE_M},{BLOCK_SIZE_N},{BLOCK_SIZE_K}]")

    if (num_stages == 3 and async_copy_use_single_buffer):
        pytest.skip(f"skip: [{M},{N},{K}] with num_stages={num_stages} and async_copy_use_single_buffer={async_copy_use_single_buffer}")
    if (num_stages == 1 and schedule_hint == "local-prefetch"):
        pytest.skip(f"skip: [{M},{N},{K}] with num_stages={num_stages} and schedule_hint={schedule_hint}")

    if boundary_check or use_mask:
        if (mix_load != 0b11):
            pytest.skip(f"skip: [{M},{N},{K}] with boundary_check=True and mix_load=0b11")
        if (M % BLOCK_SIZE_M == 0 and N % BLOCK_SIZE_N == 0 and K % BLOCK_SIZE_K == 0) or (boundary_check and use_mask):
            pytest.skip(f"skip: [{M},{N},{K}] with boundary_check=True and BLOCK[{BLOCK_SIZE_M},{BLOCK_SIZE_N},{BLOCK_SIZE_K}]")


    a2, b2 = create_test_matrices(M, K, N, layout_combination[0], layout_combination[1], dtype)

    print(f"\n{'='*80}")
    print(f"综合测试配置:")
    print(f"  数据类型: {dtype}")
    print(f"  矩阵大小: A[{M}, {K}] × B[{K}, {N}]")
    print(f"  布局组合: A({layout_combination[0]}) × B({layout_combination[1]})")
    print(f"  大小倍数: {size_multiplier}")
    print(f"  内核配置: {config_list}")
    print(f"  混合加载: {mix_load}")
    print(f"  边界检查: {boundary_check}")
    print(f"  使用掩码: {use_mask}")
    print(f"  epilogue: {optimize_epilogue}")
    print(f"  阶段数: {num_stages}")
    print(f"  异步拷贝单缓冲区: {async_copy_use_single_buffer}")
    print(f"  指令调度变种: {schedule_hint}")
    print(f"  mmac_layout_force: {mmac_layout_force}")

    config_list['optimize_epilogue'] = optimize_epilogue
    config_list['mmac_layout_force'] = mmac_layout_force
    config_list['num_stages'] = num_stages
    config_list['async_copy_use_single_buffer'] = async_copy_use_single_buffer
    config_list['schedule_hint'] = schedule_hint

    result2 = run_layout_combination_test(a2, b2, layout_combination[0], layout_combination[1],
                    mix_load, boundary_check, use_mask,
                    f"Case: A({layout_combination[0]}) × B({layout_combination[1]}) with {dtype} mix_load={mix_load}, boundary_check={boundary_check}, epilogue={optimize_epilogue}",
                    config=config_list)
    assert result2, f"Fail: dtype={dtype}, M={M}xK={K}xN={N}, A({layout_combination[0]})×B({layout_combination[1]}), config={config_list}"
    print(f"✅ Pass: dtype={dtype}, M={M}xK={K}xN={N}, A({layout_combination[0]})×B({layout_combination[1]}), config={config_list}")


def create_test_matrix_for_load_store(M, K, a_layout, dtype):
    """为 run_matrix_load_store_test 创建测试矩阵"""
    if dtype == torch.int8:
        # 对于int8类型，使用randint生成-8到7的随机整数
        if a_layout == 'row':
            a = torch.randint(-8, 8, (M, K), device='cuda', dtype=dtype)
        else:  # col
            a = torch.randint(-8, 8, (K, M), device='cuda', dtype=dtype)
    elif dtype == torch.float8_e5m2 or dtype == torch.float8_e4m3fn:
        if a_layout == 'row':
            a = torch.randn((M, K), device='cuda', dtype=torch.float16)
        else:  # col
            a = torch.randn((K, M), device='cuda', dtype=torch.float16)
        a = a.to(dtype)
    else:
        # 对于float16类型，使用randn生成正态分布随机数
        if a_layout == 'row':
            a = torch.randn((M, K), device='cuda', dtype=dtype)
        else:  # col
            a = torch.randn((K, M), device='cuda', dtype=dtype)
    return a



@pytest.mark.parametrize("boundary_check", [False] if regression_run else [False])
@pytest.mark.parametrize("use_mask", [True] if regression_run else [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.int8])
@pytest.mark.parametrize("a_layout", ['row', 'col'])
@pytest.mark.parametrize("config_list", [
    # 使用for循环生成配置列表
    config for config in [
        {'BLOCK_SIZE_M': m, 'BLOCK_SIZE_K': k}
        for m in ([32] if regression_run else [16, 32, 64])
        for k in ([64] if regression_run else [32, 64, 128, 256])
    ]
])
@pytest.mark.parametrize("size_multiplier", [16] if regression_run else [1, 2, 4, 8, 16])
def test_matrix_load_store_comprehensive(dtype, config_list, a_layout,
                                         size_multiplier, use_mask, boundary_check):

    if not is_hcu_support_mls():
        pytest.skip("skip: not support mls")
        return

    print(f"config_list={config_list}")
    print(f"dtype={dtype}")

    base_size = 16

    # Note:
    # 1> from Spec, it requires the base ptr to be 16-byte aligned !!!
    # 2> HW issue: for nmz, when matrix_load with mfilter|nfilter = dim size, result is incorrect.
    M = base_size * size_multiplier + (-4 if (boundary_check or use_mask) else 0)
    K = base_size * size_multiplier * 2 + (-8 if (boundary_check or use_mask) else 0)

    # 检查矩阵尺寸是否小于block尺寸，如果是则跳过测试
    BLOCK_SIZE_M = config_list['BLOCK_SIZE_M']
    BLOCK_SIZE_K = config_list['BLOCK_SIZE_K']

    # 通过位宽判断 dtype（优先尝试浮点，其次整数）
    try:
        dtype_bits = torch.finfo(dtype).bits
    except Exception:
        dtype_bits = torch.iinfo(dtype).bits

    if (a_layout == 'row'):
        if (dtype_bits == 8 and BLOCK_SIZE_K < 64) or (dtype_bits == 16 and BLOCK_SIZE_K < 32):
            pytest.skip(f"skip: [{M},{K}] with BLOCK_SIZE_K={BLOCK_SIZE_K}] for row layout")
    if (a_layout == 'col'):
        if (dtype_bits == 8 and BLOCK_SIZE_M < 64) or (dtype_bits == 16 and BLOCK_SIZE_M < 32):
            pytest.skip(f"skip: [{M},{K}] with BLOCK_SIZE_M={BLOCK_SIZE_M}] for col layout")

    if M < BLOCK_SIZE_M or K < BLOCK_SIZE_K:
        pytest.skip(f"skip: [{M},{K}] with BLOCK[{BLOCK_SIZE_M},{BLOCK_SIZE_K}]")

    if boundary_check:
        if (M % BLOCK_SIZE_M == 0 and K % BLOCK_SIZE_K == 0) or use_mask:
            pytest.skip(f"skip: [{M},{K}] with boundary_check=True and BLOCK[{BLOCK_SIZE_M},{BLOCK_SIZE_K}]")

    a = create_test_matrix_for_load_store(M, K, a_layout, dtype)

    print(f"\n{'='*80}")
    print(f"综合测试配置:")
    print(f"  数据类型: {dtype}")
    print(f"  矩阵大小: A[{M}, {K}]")
    print(f"  布局: A({a_layout})")
    print(f"  大小倍数: {size_multiplier}")
    print(f"  内核配置: {config_list}")
    print(f"  边界检查: {boundary_check}")
    print(f"  使用掩码: {use_mask}")

    result = run_matrix_load_store_test(a, a_layout, boundary_check=boundary_check,
                                        use_mask=use_mask, config=config_list)
    assert result, f"Fail: dtype={dtype}, M={M}xK={K}, A({a_layout}), config={config_list}, boundary_check={boundary_check}, use_mask={use_mask}"
    print(f"✅ Pass: dtype={dtype}, M={M}xK={K}, A({a_layout}), config={config_list}, boundary_check={boundary_check}, use_mask={use_mask}")

# #---------------------------------------------------------------------------------------------------


torch.manual_seed(0)

# # # 测试用例1: A(row) × B(row)
# M = 16
# N = 32
# K = 40
# dtype = torch.float16

# config1 = {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'num_warps': 2, "GROUP_SIZE_M": 1}
# config1["num_stages"] = 1

# print(f"\n{'='*60}")
# print(f"测试用例1: A(row) × B(row), config={config1}")
# print(f"A: [{M}, {K}] 行主序, B: [{K}, {N}] 行主序")
# a1, b1 = create_test_matrices(M, K, N, 'row', 'row', dtype)
# # a1[:, 0:64]  = 1
# # b1[:, 0:16]  = 0
# # b1[:, 16:32] = 1
# # b1[:, 32:48] = -1
# # b1[:, 48:64] = 0
# result1 = run_layout_combination_test(a1, b1, 'row', 'row', 0b00, True, True, "Case 1: A(row) × B(row)", config1)


# 测试用例2: A(row) × B(col)
M = 16
N = 16
K = 40
dtype = torch.float16

config2 = {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 64, 'num_warps': 1, "GROUP_SIZE_M": 1}
config2['optimize_epilogue'] = 1
config2['num_stages'] = 2
config2['async_copy_use_single_buffer'] = True
# config2['schedule_hint'] = "local-prefetch"

print(f"\n{'='*60}")
print(f"测试用例2: A(row) × B(col), config={config2}")
print(f"A: [{M}, {K}] 行主序, B: [{K}, {N}] 行主序但逻辑上是[{N}, {K}]列主序")
a2, b2 = create_test_matrices(M, K, N, 'row', 'col', dtype)
result2 = run_layout_combination_test(a2, b2, 'row', 'col', 0b11, False, True, "Case 2: A(row) × B(col)", config2)



# # # 测试用例3: A(col) × B(row)

# M = 64
# N = 64
# K = 60
# dtype = torch.int8 #torch.float16

# config3 = {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'num_warps': 2, "GROUP_SIZE_M": 1}


# print(f"\n{'='*60}")
# print(f"测试用例3: A(col) × B(row), config={config3}")
# print(f"A: [{K}, {M}] 列主序, B: [{K}, {N}] 行主序")
# a3, b3 = create_test_matrices(M, K, N, 'col', 'row', dtype)
# # a3[:, 0:16]  = 1
# # a3[:, 16:32] = -1
# # b3[:, 0:16] = 1
# # b3[:, 16:32] = -1

# result3 = run_layout_combination_test(a3, b3, 'col', 'row', 0b00, True, True, "Case 3: A(col) × B(row)", config3)


# # 测试用例4: A(col) × B(col)

# M = 64
# N = 64
# K = 112
# dtype = torch.int8

# config4 = {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'num_warps': 2, "GROUP_SIZE_M": 1}
# config4['optimize_epilogue'] = 1

# print(f"\n{'='*60}")
# print(f"测试用例4: A(col) × B(col), config={config4}")
# print(f"A: [{K}, {M}] 列主序, B: [{K}, {N}] 列主序")
# a4, b4 = create_test_matrices(M, K, N, 'col', 'col', dtype)

# result4 = run_layout_combination_test(a4, b4, 'col', 'col', 0b00, True, True, "Case 4: A(col) × B(col)", config4)


####################################################################################################

# # matrix_load_store_kernel 测试用例1: row layout

# M = 32
# K = 92
# dtype = torch.int8

# config = {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_K': 64}
# config['num_warps'] = 1
# config['num_stages'] = 1

# print(f"\n{'='*60}")
# print("run_matrix_load_store_test 测试用例1: A(row)")
# print(f"A: [{M}, {K}] 行主序")
# print(f"config: {config}")

# a = create_test_matrix_for_load_store(M, K, 'row', dtype)
# # 根据 a 的 shape，生成从 0 到 size 的值
# # a = torch.arange(0, a.numel(), device=a.device, dtype=a.dtype).reshape(a.shape)

# run_matrix_load_store_test(a, 'row', boundary_check=True, use_mask=True, config=config)



# # run_matrix_load_store_test 测试用例2: A(col)

# M = 64
# K = 32
# N_dummy = 64
# dtype = torch.int8

# config = {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 32}
# config['num_warps'] = 1
# config['num_stages'] = 1

# print(f"\n{'='*60}")
# print("run_matrix_load_store_test 测试用例2: A(col)")
# print(f"A: [{M}, {K}] 列主序")
# print(f"config: {config}")

# a = create_test_matrix_for_load_store(M, K, 'col', dtype)
# # 根据 a 的 shape，生成从 0 到 size 的值
# # a = torch.arange(0, a.numel(), device=a.device, dtype=a.dtype).reshape(a.shape)

# run_matrix_load_store_test(a, 'col', boundary_check=True, use_mask=True, config=config)

####################################################################################################


# %%
# Benchmark - mix_load 对比测试
# -----------------------------

matrix_mls_test_sizes = [
    # (M, N, K)
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
    (16384, 16384, 16384),
    # moe gemm1 case
    (32, 256, 7168),
    (64, 256, 7168),
    (128, 256, 7168),
    (256, 256, 7168),
    (512, 256, 7168),
    (1024, 256, 7168),
    (2048, 256, 7168),
    (4096, 256, 7168),
    (8192, 256, 7168),
    (16384, 256, 7168),
    # moe gemm2 case
    (32, 7168, 256),
    (64, 7168, 256),
    (128, 7168, 256),
    (256, 7168, 256),
    (512, 7168, 256),
    (1024, 7168, 256),
    (2048, 7168, 256),
    (4096, 7168, 256),
    (8192, 7168, 256),
    (16384, 7168, 256),
    # # fa case
    (128, 128, 128),
    (256, 256, 128),
    (512, 512, 128),
    (1024, 1024, 128),
    (2048, 2048, 128),
    (4096, 4096, 128),
    (8192, 8192, 128),
    (16384, 16384, 128),
]

# matrix_mls_test_sizes = [
#     # (M, N, K)
#     (128, 128, 128-4),
#     (256, 256, 256-4),
#     (512, 512, 512-4),
#     (1024, 1024, 1024-4),
#     (2048, 2048, 2048-4),
#     (4096, 4096, 4096-4),
#     (8192, 8192, 8192-4),
#     (16384, 16384, 16384-4),
#     # moe gemm1 case
#     (32, 256, 7168-4),
#     (64, 256, 7168-4),
#     (128, 256, 7168-4),
#     (256, 256, 7168-4),
#     (512, 256, 7168-4),
#     (1024, 256, 7168-4),
#     (2048, 256, 7168-4),
#     (4096, 256, 7168-4),
#     (8192, 256, 7168-4),
#     (16384, 256, 7168-4),
#     # moe gemm2 case
#     (32, 7168, 256-4),
#     (64, 7168, 256-4),
#     (128, 7168, 256-4),
#     (256, 7168, 256-4),
#     (512, 7168, 256-4),
#     (1024, 7168, 256-4),
#     (2048, 7168, 256-4),
#     (4096, 7168, 256-4),
#     (8192, 7168, 256-4),
#     (16384, 7168, 256-4),
#     # # fa case
#     (128, 128, 128-4),
#     (256, 256, 128-4),
#     (512, 512, 128-4),
#     (1024, 1024, 128-4),
#     (2048, 2048, 128-4),
#     (4096, 4096, 128-4),
#     (8192, 8192, 128-4),
#     (16384, 16384, 128-4),
# ]
matrix_load_line_vals = ["mls", "buffer_load"]
matrix_load_line_names = ["MLS(tflops)", "BUF_LOAD(tflops)"]
benchmark_matmul_mls_configs = []
for fp8_inputs in [False, True]:
    for a_layout in ['row']:
        for b_layout in ['col', 'row']:
            benchmark_matmul_mls_configs.append(
                triton.testing.Benchmark(
                    x_names=["M", "N", "K"],
                    x_vals=matrix_mls_test_sizes,
                    line_arg="matrix_load_mode",
                    line_vals=matrix_load_line_vals,
                    line_names=matrix_load_line_names,
                    styles=[("green", "-"), ("orange", "-")],
                    ylabel="TFLOPS",
                    plot_name="matmul-load-performance-" + a_layout + "-" + b_layout +  "-" + ("fp8" if fp8_inputs else "fp16"),
                    args={"fp8_inputs": fp8_inputs, "a_layout": a_layout, "b_layout": b_layout},
                )
            )

@triton.testing.perf_report(benchmark_matmul_mls_configs)
def benchmark_matmul_mls(M, N, K, matrix_load_mode, fp8_inputs, a_layout, b_layout):
    dtype = torch.float8_e5m2 if fp8_inputs else torch.float16
    if matrix_load_mode == "mls":
        matrix_load_mode_value = 0b00
    elif matrix_load_mode == "buffer_load":
        matrix_load_mode_value = 0b11
    else:
        raise ValueError(f"未知的 matrix_load_mode: {matrix_load_mode}")

    a, b = create_test_matrices(M, K, N, a_layout, b_layout, dtype)
    quantiles = [0.5, 0.2, 0.8]

    def _matmul_mls():
        return matmul_with_layout(
            a, b,
            a_layout=a_layout,
            b_layout=b_layout,
            mix_load=matrix_load_mode_value,
            boundary_check=False,
            use_mask=False,
            activation="",
        )

    ms, min_ms, max_ms = triton.testing.do_bench(_matmul_mls, quantiles=quantiles)
    perf = lambda t: 2 * M * N * K * 1e-12 / (t * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


# if __name__ == "__main__":
#     benchmark_matmul_mls.run(show_plots=True, print_data=True)


# if __name__ == "__main__":
#     pytest.main(["-v", "-s", "mls-unit-test.py::test_matrix_load_store_comprehensive[16-config_list11-col-dtype1-True-True]"])
