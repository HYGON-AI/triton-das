# SPDX-License-Identifier: Apache-2.0

# Adapted from https://github.com/sgl-project/sglang/pull/3730
import functools
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import triton
import triton.language as tl

# from sglang.srt.utils import get_device_name
from vllm.platforms import current_platform

logger = logging.getLogger(__name__)


@triton.jit
def _per_token_quant_int8(
    x_ptr,
    xq_ptr,
    scale_ptr,
    stride_x,
    stride_xq,
    N,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)

    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row_id * stride_x + cols,
                mask=mask, other=0.0).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(x)), 1e-10)
    scale_x = absmax / 127
    x_q = x * (127 / absmax)
    x_q = tl.extra.cuda.libdevice.round(x_q).to(tl.int8)

    tl.store(xq_ptr + row_id * stride_xq + cols, x_q, mask=mask)
    tl.store(scale_ptr + row_id, scale_x)


def per_token_quant_int8(x):
    M = x.numel() // x.shape[-1]
    N = x.shape[-1]
    x_q = torch.empty_like(x, device=x.device, dtype=torch.int8)
    scales = torch.empty(x.shape[:-1] + (1,),
                         device=x.device, dtype=torch.float32)
    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)

    assert x.is_contiguous()
    _per_token_quant_int8[(M,)](
        x,
        x_q,
        scales,
        stride_x=x.stride(-2),
        stride_xq=x_q.stride(-2),
        N=N,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )

    return x_q, scales

configs = [
    triton.Config({"BLOCK_SIZE_M": BLOCK_SIZE_M}, num_warps=num_warps, num_stages=num_stages)
    for BLOCK_SIZE_M in [1, 2, 3, 4] for num_warps in [1, 2, 4, 8] for num_stages in [1, 2]
]
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 1}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_SIZE_M": 2}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_SIZE_M": 3}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_SIZE_M": 3}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_SIZE_M": 4}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_SIZE_M": 4}, num_warps=1, num_stages=2),
    ],
    key=["M", "GROUP_SIZE"],
    #perf_debug=True
)
@triton.heuristics(
    values={
        "GROUP_SIZE_ALIGNED": lambda args: triton.next_power_of_2(args["GROUP_SIZE"]),
        "MASK_FREE": lambda args: args['GROUP_SIZE'] == args['GROUP_SIZE_ALIGNED']
    }
)
@triton.jit
def _per_token_group_quant_int8(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    # Avoid to divide zero
    eps,
    # Information for int8
    int8_min,
    int8_max,
    M,
    GROUP_SIZE: tl.constexpr,
    GROUP_SIZE_ALIGNED: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    MASK_FREE: tl.constexpr
):
    """A Triton-accelerated function to perform
    per-token-group quantization on a tensor.

    This function converts the tensor values into int8 values.
    """
    # Map the program id to the row of X and Y it should compute.
    tl.assume(BLOCK_SIZE_M > 0)
    tl.assume(GROUP_SIZE_ALIGNED > 0)

    g_id = tl.program_id(0)

    BLOCK_SIZE_M = min(BLOCK_SIZE_M, M - g_id * BLOCK_SIZE_M)

    for group_idx in range(BLOCK_SIZE_M):
        # Calculate base pointers for this iteration
        curr_y_ptr = y_ptr + (g_id * BLOCK_SIZE_M + group_idx) * GROUP_SIZE
        curr_y_q_ptr = y_q_ptr + (g_id * BLOCK_SIZE_M + group_idx) * GROUP_SIZE
        curr_y_s_ptr = y_s_ptr + g_id * BLOCK_SIZE_M + group_idx

        # TODO: xukang, cols be const?
        cols = tl.arange(0, GROUP_SIZE_ALIGNED)
        mask = cols < GROUP_SIZE
        if MASK_FREE:
            y = tl.load(curr_y_ptr + cols).to(tl.float32)
        else:
            y = tl.load(curr_y_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        # Quant
        _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
        y_s = _absmax / int8_max
        y_q = tl.clamp(y / y_s, int8_min, int8_max).to(y_q_ptr.dtype.element_ty)

        if MASK_FREE:
            tl.store(curr_y_q_ptr + cols, y_q)
        else:
            tl.store(curr_y_q_ptr + cols, y_q, mask=mask)

        tl.store(curr_y_s_ptr, y_s)


def per_token_group_quant_int8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype = torch.int8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Function to perform per-token-group quantization on an input tensor `x`.

    It converts the tensor values into signed int8 values and returns the
    quantized tensor along with the scaling factor used for quantization.

    Args:
        x: The input tenosr with ndim >= 2.
        group_size: The group size used for quantization.
        eps: The minimum to avoid dividing zero.
        dtype: The dype of output tensor. Note that only `torch.int8`
        is supported for now.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The quantized tensor and the
        scaling factor for quantization.
    """
    assert (
        x.shape[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    iinfo = torch.iinfo(dtype)
    int8_max = iinfo.max
    int8_min = iinfo.min

    x_q = torch.empty_like(x, device=x.device, dtype=dtype)
    M = x.numel() // group_size
    x_s = torch.empty(
        x.shape[:-1] + (x.shape[-1] // group_size,),
        device=x.device,
        dtype=torch.float32,
    )

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']),
    )
    _per_token_group_quant_int8[grid](
        x,
        x_q,
        x_s,
        eps,
        int8_min=int8_min,
        int8_max=int8_max,
        M=M,
        GROUP_SIZE=group_size,
    )

    return x_q, x_s


'''
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": block_size[0],
            "BLOCK_SIZE_K": block_size[1],
            "GROUP_SIZE_M": 32,
            "num_warps": 4,
            "num_stages": 3,
        }


'''
configs = [
    triton.Config(
        {"BLOCK_SIZE_M": BLOCK_SIZE_M, "BLOCK_SIZE_N": BLOCK_SIZE_N, "BLOCK_SIZE_K": BLOCK_SIZE_K,
         "GROUP_SIZE_M": GROUP_SIZE_M, "COMBINE_SCALE_LOAD": COMBINE_SCALE_LOAD},
        num_warps=num_warps, num_stages=num_stages)
        for BLOCK_SIZE_M in [16, 32, 64, 128, 256]
        for BLOCK_SIZE_N in [16, 32, 64, 128]
        for BLOCK_SIZE_K in [128]
        for GROUP_SIZE_M in [32, 64, 128]
        for COMBINE_SCALE_LOAD in [True, False]
        for num_warps in [1, 2, 4, 8, 16]
        for num_stages in [1, 2, 3]
]
'''
@triton.autotune(
    configs=configs,
    key=["M", "K", "N", "group_n", "group_k"],
    perf_debug=True
)
'''
@triton.heuristics(
    values={
        "DIVISIBLE_M": lambda args: args["M"] % args["BLOCK_SIZE_M"] == 0,
        "DIVISIBLE_N": lambda args: args["N"] % args["BLOCK_SIZE_N"] == 0,
    }
)
@triton.jit
def _w8a8_block_int8_matmul(
    # Pointers to inputs and output
    A,
    B,
    C,
    As,
    Bs,
    # Shape for matmul
    M,
    N,
    K,
    # Block size for block-wise quantization
    group_n,
    group_k,
    # Stride for inputs and output
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_k,
    stride_Bs_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
    COMBINE_SCALE_LOAD: tl.constexpr = False
):
    """Triton-accelerated function used to perform linear operations (dot
    product) on input tensors `A` and `B` with block-wise quantization,
    and store the result in output tensor `C`.
    """

    # TODO: xukang, add split_k_iters
    # TODO: xukang, if there are many buffer_store_dword, make OPTIMIZE_EPILOGUE autotune param
    # TODO: xukang, now there are many buffer_load_dword, try optimizing load data coelescing
    # TODO: xukang, now there are many ds_read_b64, try optimize to ds_read_b128
    # TODO: xukang, add autotune params to tune the position of the load of A,B,As,Bs
    # TODO: xukang, make autotune params larger range

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    tl.assume(pid > 0)
    tl.assume(pid_m > 0)
    tl.assume(pid_n > 0)
    tl.assume(group_n > 0)
    tl.assume(group_k > 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(stride_As_m > 0)
    tl.assume(stride_As_k > 0)
    tl.assume(stride_Bs_k > 0)
    tl.assume(stride_Bs_n > 0)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    if COMBINE_SCALE_LOAD:
        As_ptrs = As + offs_am[:, None] * stride_As_m
        offs_bsn = offs_bn // group_n
        Bs_ptrs = Bs + offs_bsn[:, None] * stride_Bs_n
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K), 2):
            a0 = tl.load(a_ptrs,
                        mask=offs_k[None, :] < K - k * BLOCK_SIZE_K,
                        other=0.0)
            b0 = tl.load(b_ptrs,
                        mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                        other=0.0)

            k_start = k * BLOCK_SIZE_K
            offs_ks = k_start // group_k + tl.arange(0, 2)
            a_s = tl.load(As_ptrs + offs_ks[None, :] * stride_As_k,
                          mask=offs_ks[None, :] <= (K - 1) // group_k,
                          other=0.0)
            b_s = tl.load(Bs_ptrs + offs_ks[None, :] * stride_Bs_k,
                          mask=offs_ks[None, :] <= (K - 1) // group_k,
                          other=0.0)
            a_s0, a_s1 = tl.split(a_s)
            b_s0, b_s1 = tl.split(b_s)

            accumulator += tl.dot(a0, b0).to(tl.float32) * a_s0[:, None] * b_s0[None, :]

            a0 = tl.load(a_ptrs + BLOCK_SIZE_K * stride_ak,
                        mask=offs_k[None, :] < K - (k + 1)* BLOCK_SIZE_K,
                        other=0.0)
            b0 = tl.load(b_ptrs + BLOCK_SIZE_K * stride_bk,
                        mask=offs_k[:, None] < K - (k + 1) * BLOCK_SIZE_K,
                        other=0.0)

            accumulator += tl.dot(a0, b0).to(tl.float32) * a_s1[:, None] * b_s1[None, :]

            a_ptrs += BLOCK_SIZE_K * stride_ak * 2
            b_ptrs += BLOCK_SIZE_K * stride_bk * 2
    else:
        As_ptrs = As + offs_am * stride_As_m
        offs_bsn = offs_bn // group_n
        Bs_ptrs = Bs + offs_bsn * stride_Bs_n
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(a_ptrs,
                        mask=offs_k[None, :] < K - k * BLOCK_SIZE_K,
                        other=0.0)
            b = tl.load(b_ptrs,
                        mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                        other=0.0)

            k_start = k * BLOCK_SIZE_K
            offs_ks = k_start // group_k
            a_s = tl.load(As_ptrs + offs_ks * stride_As_k)
            b_s = tl.load(Bs_ptrs + offs_ks * stride_Bs_k)

            accumulator += tl.dot(a, b).to(tl.float32) * a_s[:, None] * b_s[None, :]

            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
 

    # TODO: xukang, make this heuristic consts
    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    #c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    offs_c = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_ptrs = C + offs_c

    STORE_MASK_FREE: tl.constexpr = DIVISIBLE_M & DIVISIBLE_N
    if STORE_MASK_FREE:
        tl.store(c_ptrs, c)
    else:
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

@functools.lru_cache
def get_w8a8_block_int8_configs(
    N: int, K: int, block_n: int, block_k: int
) -> Optional[Dict[int, Any]]:
    """
    Return optimized configurations for the w8a8 block fp8 kernel.

    The return value will be a dictionary that maps an irregular grid of
    batch sizes to configurations of the w8a8 block fp8 kernel. To evaluate the
    kernel on a given batch size bs, the closest batch size in the grid should
    be picked and the associated configuration chosen to invoke the kernel.
    """

    # First look up if an optimized configuration is available in the configs
    # directory
    device_name = current_platform.get_device_name().replace(" ", "_")
    json_file_name = f"N={N},K={K},device_name={device_name},dtype=int8_w8a8,block_shape=[{block_n}, {block_k}].json" # noqa: E501

    config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name
    )
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            logger.info(
                "Using configuration from %s for W8A8 Block INT8 kernel.",
                config_file_path,
            )
            # If a configuration has been found, return it
            return {int(key): val for key, val in json.load(f).items()}

    # If no optimized configuration is available, we will use the default
    # configuration
    logger.warning(
        (
            "Using default W8A8 Block INT8 kernel config. Performance might "
            "be sub-optimal! Config file not found at %s"
        ),
        config_file_path,
    )
    return None


def w8a8_block_int8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: List[int],
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """matrix multiplication with block-wise quantization.

    It takes two input tensors `A` and `B` with scales `As` and `Bs`.
    The output is returned in the specified `output_dtype`.

    Args:
        A: The input tensor, e.g., activation.
        B: The input tensor, e.g., weight.
        As: The per-token-group quantization scale for `A`.
        Bs: The per-block quantization scale for `B`.
        block_size: The block size for per-block quantization. It should
        be 2-dim, e.g., [128, 128].
        output_dytpe: The dtype of the returned tensor.

    Returns:
        torch.Tensor: The result of matmul.
    """
    # print("A,B,As,Bs,block_size0,block_size1",A.shape,B.shape,As.shape,Bs.shape,block_size[0], block_size[1])
    assert len(block_size) == 2
    block_n, block_k = block_size[0], block_size[1]

    assert A.shape[-1] == B.shape[-1]
    assert A.shape[:-1] == As.shape[:-1] and A.is_contiguous()
    assert triton.cdiv(A.shape[-1], block_k) == As.shape[-1]
    M = A.numel() // A.shape[-1]

    assert B.ndim == 2 and B.is_contiguous() and Bs.ndim == 2
    N, K = B.shape
    assert triton.cdiv(N, block_n) == Bs.shape[0]
    assert triton.cdiv(K, block_k) == Bs.shape[1]

    C_shape = A.shape[:-1] + (N,)
    C = A.new_empty(C_shape, dtype=output_dtype)

    configs = get_w8a8_block_int8_configs(N, K, block_size[0], block_size[1])
    if configs:
        # If an optimal configuration map has been found, look up the
        # optimal config
        config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
    else:
        # Default config
        # Block-wise quant: BLOCK_SIZE_K must be divisable by block_size[1]
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": block_size[0],
            "BLOCK_SIZE_K": block_size[1],
            "GROUP_SIZE_M": 32,
            "COMBINE_SCALE_LOAD": False,
            "num_warps": 4,
            "num_stages": 3,
        }

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) *
            triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    _w8a8_block_int8_matmul[grid](
        A,
        B,
        C,
        As,
        Bs,
        M,
        N,
        K,
        block_n,
        block_k,
        A.stride(-2),
        A.stride(-1),
        B.stride(1),
        B.stride(0),
        C.stride(-2),
        C.stride(-1),
        As.stride(-2),
        As.stride(-1),
        Bs.stride(1),
        Bs.stride(0),
        **config,
    )

    return C


def apply_w8a8_block_int8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: List[int],
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert input_scale is None
    # View input as 2D matrix for fp8 methods
    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[0]]

    q_input, x_scale = per_token_group_quant_int8(input_2d, block_size[1])
    output = w8a8_block_int8_matmul(
        q_input, weight, x_scale, weight_scale, block_size,
        output_dtype=input.dtype
    )

    if bias is not None:
        output = output + bias
    return output.to(dtype=input.dtype).view(*output_shape)


def input_to_int8(
    x: torch.Tensor, dtype: torch.dtype = torch.int8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """This function quantizes input values to
    int8 values with tensor-wise quantization.
    """
    iinfo = torch.iinfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    int8_min, int8_max = iinfo.min, iinfo.max
    scale = int8_max / amax
    x_scl_sat = (x * scale).clamp(min=int8_min, max=int8_max)
    return x_scl_sat.to(dtype).contiguous(), scale.float().reciprocal()


def block_dequant(
    x_q_block: torch.Tensor,
    x_s: torch.Tensor,
    block_size: List[int],
) -> torch.Tensor:
    """This function conducts block-wise dequantization.
    The inputs are block-wise quantization tensor `x_q_block`,
    block-wise quantization scale and the block size.
    The outputs are dequantized tensor.
    """
    block_n, block_k = block_size[0], block_size[1]
    n, k = x_q_block.shape
    n_tiles = (n + block_n - 1) // block_n
    k_tiles = (k + block_k - 1) // block_k
    assert n_tiles == x_s.shape[0]
    assert k_tiles == x_s.shape[1]

    x_dq_block = x_q_block.to(torch.float32)

    for i in range(k_tiles):
        for j in range(n_tiles):
            x_dq_block[
                j * block_n : min((j + 1) * block_n, n),
                i * block_k : min((i + 1) * block_k, k),
            ] *= x_s[j][i]

    return x_dq_block
