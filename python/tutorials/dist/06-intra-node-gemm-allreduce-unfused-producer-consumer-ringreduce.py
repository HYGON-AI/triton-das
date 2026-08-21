# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
# Modified by Hygon Information Technology Co., Ltd., 2026.


import triton
import torch

import argparse
import random
import numpy as np
import os
import datetime
from typing import Optional

import triton.language as tl
import pyrocshmem

from triton.language.extra import libshmem_device
from triton.language.extra.hip import libdevice

from hip import hip

import ctypes
libc = ctypes.CDLL('libc.so.6')


from utils import (
    generate_data,
    get_torch_prof_ctx,
    perf_func,
    dist_print,
)

from functools import partial

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
    "s8": torch.int8,
    "s32": torch.int32,
}

THRESHOLD_MAP = {
    torch.float16: 1e-2,
    torch.bfloat16: 1e-2,
    torch.float8_e4m3fn: 1e-2,
    torch.float8_e5m2: 1e-2,
}

def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    return result

def _make_data(M, N, K, scale, dtype, device):
    data_config = [
        ((M, K), dtype, (0.01 * scale, 0), device),  # A
        ((N, K), dtype, (0.01 * scale, 0), device),  # B
        None, #((M, N), dtype, (1, 0), device),
        # (  # bias
        #     None if not args.has_bias else ((M, args.N), dtype, (1, 0))),
    ]
    generator = generate_data(data_config)
    input, weight, bias = next(generator)
    return input, weight, bias


def init():
    os.environ["TRITON_HIP_USE_BLOCK_PINGPONG"] = "1"
    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(LOCAL_RANK)
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=WORLD_SIZE,
        rank=RANK,
        timeout=datetime.timedelta(seconds=1800),
    )
    assert torch.distributed.is_initialized()
    TP_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")
    torch.distributed.barrier(TP_GROUP)

    torch.use_deterministic_algorithms(False, warn_only=True)
    torch.set_printoptions(precision=5)
    torch.manual_seed(3 + RANK)
    torch.cuda.manual_seed_all(3 + RANK)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    np.random.seed(3 + RANK)
    random.seed(5)

    torch.cuda.synchronize()
    torch.distributed.barrier()
    
    torch.manual_seed(3)
    torch.cuda.manual_seed_all(3)
    
    return RANK, LOCAL_RANK, WORLD_SIZE, TP_GROUP

def torch_gemm_rs(
    input: torch.Tensor,  # [M, local_k]
    weight: torch.Tensor,  # [N, local_K]
    bias: Optional[torch.Tensor],
    TP_GROUP,
    transpose_weight,
    all_reduce=False
):
    M, local_K = input.shape
    if not transpose_weight:
        weight = weight.T
    N = weight.shape[1]
    output = torch.matmul(input, weight)

    if bias:
        output = output + bias
    
    # return output
    if not all_reduce:
        rs_output = torch.empty((M // WORLD_SIZE, N), dtype=output.dtype, device=input.device)
        torch.distributed.reduce_scatter_tensor(rs_output, output, group=TP_GROUP)
        return rs_output
    else:
        torch.distributed.all_reduce(output, group=TP_GROUP)
        return output

@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n

configs=[
    triton.Config(
        {
            'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 4,
        }, num_warps=4, num_stages=1, num_ctas=1),
]

@triton.autotune(
    configs=configs,
    key=['M', 'N', 'K'],
    # use_cuda_graph=True,
)
@triton.jit
def kernel_gemm_rs_producer_persistent(
    a_ptr, b_ptr,
    scatter_buf_ptr,
    counter_ptr,
    barrier_ptr,
    rank, num_ranks,
    M, N, K,  #
    stride_am, stride_ak,  #
    stride_bk, stride_bn,  #
    stride_cm, stride_cn,  #
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    NUM_SMS: tl.constexpr,  #
    RANK_SWIZZLE: tl.constexpr = True,
    ):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    # _compute_pid
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    M_per_rank = M // num_ranks
    num_pid_m_per_rank = M_per_rank // BLOCK_SIZE_M
    rank_offset = pid_m // num_pid_m_per_rank

    if RANK_SWIZZLE:
        rank_offset = (rank_offset + rank + 1) % num_ranks
        pid_m = pid_m - first_pid_m + num_pid_m_per_rank * rank_offset
    
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    dtype = a_ptr.dtype.element_ty
    c = accumulator.to(dtype)
    
    # store to local scatter buffer
    c_ptr = scatter_buf_ptr
    
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    c = accumulator.to(dtype)
    tl.store(c_ptrs, c, mask=c_mask)
    
    val = tl.atomic_add(counter_ptr + rank_offset, 1, sem="release", scope="gpu")
    if val == num_pid_m_per_rank * num_pid_n - 1:
        tl.store(barrier_ptr + rank_offset, 1, cache_modifier=".wt")
        # reset counter_ptr
        tl.store(counter_ptr + rank_offset, 0, cache_modifier=".wt")

@triton.jit
def add_continuous_kernel(
    lhs,
    rhs,
    out,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    n_blocks = tl.cdiv(N, BLOCK_SIZE)
    num_pid = tl.num_programs(axis=0)

    lhs_block_ptr = tl.make_block_ptr(
        base=lhs,
        shape=(N, ),
        strides=(1, ),
        offsets=(block_start, ),
        block_shape=(BLOCK_SIZE, ),
        order=(0, ),
    )
    rhs_block_ptr = tl.make_block_ptr(
        base=rhs,
        shape=(N, ),
        strides=(1, ),
        offsets=(block_start, ),
        block_shape=(BLOCK_SIZE, ),
        order=(0, ),
    )
    out_block_ptr = tl.make_block_ptr(
        base=out,
        shape=(N, ),
        strides=(1, ),
        offsets=(block_start, ),
        block_shape=(BLOCK_SIZE, ),
        order=(0, ),
    )
    for _ in range(pid, n_blocks, num_pid):
        tl.store(
            out_block_ptr,
            tl.load(lhs_block_ptr, boundary_check=(0, )) + tl.load(rhs_block_ptr, boundary_check=(0, )),
            boundary_check=(0, ),
        )
        lhs_block_ptr = tl.advance(lhs_block_ptr, [BLOCK_SIZE * num_pid])
        rhs_block_ptr = tl.advance(rhs_block_ptr, [BLOCK_SIZE * num_pid])
        out_block_ptr = tl.advance(out_block_ptr, [BLOCK_SIZE * num_pid])


# // CUDA kernel for P2P copy using shared memory
# __global__ static void copyp2p(int4 *__restrict__ dest, int4 const *__restrict__ src,
#                         size_t num_elems) {
#   size_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
#   size_t gridSize = blockDim.x * gridDim.x;

# #pragma unroll(5)
#   for (size_t i = globalId; i < num_elems; i += gridSize) {
#     dest[i] = src[i];
#   }
# }
@triton.jit
def copyp2p_kernel(
    dest,
    src,
    num_elems,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    n_blocks = tl.cdiv(num_elems, BLOCK_SIZE)
    num_pid = tl.num_programs(axis=0)

    dest_block_ptr = tl.make_block_ptr(
        base=dest,
        shape=(num_elems, ),
        strides=(1, ),
        offsets=(block_start, ),
        block_shape=(BLOCK_SIZE, ),
        order=(0, ),
    )
    src_block_ptr = tl.make_block_ptr(
        base=src,
        shape=(num_elems, ),
        strides=(1, ),
        offsets=(block_start, ),
        block_shape=(BLOCK_SIZE, ),
        order=(0, ),
    )

    for _ in range(pid, n_blocks, num_pid):
        tl.store(
            dest_block_ptr,
            tl.load(src_block_ptr, boundary_check=(0, )),
            boundary_check=(0, ),
        )
        dest_block_ptr = tl.advance(dest_block_ptr, [BLOCK_SIZE * num_pid])
        src_block_ptr = tl.advance(src_block_ptr, [BLOCK_SIZE * num_pid])

def add_continuous(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    out: Optional[torch.Tensor],
    stream: torch.cuda.Stream,
    num_ctas=16,
    num_warps=8,
):
    assert lhs.dtype == rhs.dtype and lhs.numel() == rhs.numel()
    if out is None:
        out = torch.empty_like(lhs)
    
    with torch.cuda.stream(stream):
        add_continuous_kernel[(num_ctas, )](  # local memory bw is very high. use many blocks
            lhs, rhs, out, out.numel(), num_warps=num_warps,
            BLOCK_SIZE=32 * 8 * 4,  # per thread has 8*4 elements
        )
    return out

@triton.jit
def kernel_set_signal(
    ptr,
    val
):
    # tl.atomic_cas(ptr, 0, 1, sem="release", scope="gpu")
    while tl.atomic_cas(ptr, 0, 1, sem="release", scope="gpu") != 0:
        pass
    # tl.store(ptr, 1, cache_modifier=".wt")

@triton.jit
def kernel_wait_eq_cuda(ptr, cmp, val):
    # while tl.load(ptr, cache_modifier=".cv", volatile=True) != 1:
    #     pass
    while tl.atomic_cas(ptr, cmp, val, sem="acquire", scope="sys") == 0:
        pass

def _wait_eq_cuda(ptr, stream):
    grid = lambda meta: (1, )
    with torch.cuda.stream(stream):
        kernel_wait_eq_cuda[grid](ptr, 1, 0)
    
@triton.jit
def barrier_all_ipc(rank, num_ranks, comm_buf_base_ptrs):
    tid = libdevice.thread_idx(axis=0)  # noqa: F841
    for i in range(num_ranks):
        remote_base_ptr = tl.load(comm_buf_base_ptrs + i).to(tl.pointer_type(tl.int32))
        # remote_base_ptr = comm_buf_base_ptrs[i]
        while tl.atomic_cas(remote_base_ptr + rank, 0, 1, scope="sys", sem="release") != 0:
            pass

    for i in range(num_ranks):
        local_base_ptr = tl.load(comm_buf_base_ptrs + rank).to(tl.pointer_type(tl.int32))
        # local_base_ptr = comm_buf_base_ptrs[rank]
        while tl.atomic_cas(local_base_ptr + i, 1, 0, scope="sys", sem="acquire") != 1:
            pass

    tl.debug_barrier()

def barrier_all_on_stream(
    rank,
    num_ranks,
    sync_bufs_ptr,
    stream,
):
    with torch.cuda.stream(stream):
        barrier_all_ipc[(1, )](rank, num_ranks, sync_bufs_ptr)

def triton_gemm_rs(
    a, b,
    gemm_bufs,
    barrier_bufs,
    scatter_bufs,
    scatter_signal_bufs,
    sync_bufs_ptr,
    counter_buf,
    rank, num_ranks, M, N, K,
    num_sms,
    current_stream,
    scatter_stream
):
    scatter_stream.wait_stream(current_stream)
    # counter_buf = torch.zeros((num_ranks, ), dtype=torch.int32, device=device)
    m_per_rank = a.shape[0] // num_ranks
    
    # final output
    output = torch.empty((m_per_rank, N), dtype=a.dtype, device=a.device)
    num_elems_per_rank = m_per_rank * N
    # nbytes_per_rank = m_per_rank * N * a.dtype.itemsize
    # local_buf_base_ptr = gemm_buf.data_ptr()
    # remote_offset = rank * nbytes_per_rank

    num_sms_gemm = 64
    # num_sms_rs = num_sms - num_sms_gemm
    num_sms_rs = 64

    # grid = lambda META: (min(
    #             num_sms_gemm,
    #             triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    #         ), )
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    with torch.cuda.stream(current_stream):
        kernel_gemm_rs_producer_persistent[grid](
                a, b,
                gemm_bufs[rank],
                counter_buf,
                barrier_bufs[rank],
                rank, num_ranks, M, N, K,
                a.stride(0), a.stride(1),
                b.stride(1), b.stride(0),
                N, 1,
                NUM_SMS=num_sms_gemm
            )

    # ring scatter and reduce
    num_ctas_rs = 16
    num_warps_rs = 16

    # optim 2: with copyp2p kernel (require stores to remote)
    # NOTE: gemm kernel not affected (because we limit CU for copyp2p_kernel)
    with torch.cuda.stream(scatter_stream):
        to_rank = (rank - 1 + num_ranks) % num_ranks
        # for stage in range(num_ranks):
        for stage in range(num_ranks + num_ranks): # extent to all-gather
            segment = (rank + stage + 1) % num_ranks
            m_start = segment * m_per_rank
            m_end = m_start + m_per_rank
            src = gemm_bufs[rank][m_start:m_end]
            dst = scatter_bufs[to_rank][m_start:m_end]
            # wait gemm buf ready (local buf)
            if stage < num_ranks:
                _wait_eq_cuda(barrier_bufs[rank][segment], scatter_stream)
            
            # cur_out = output if output is not None and stage == num_ranks - 1 else dst
            cur_out = dst
            
            # P2P_KERNEL_BLOCK_SIZE = 1024
            if stage == 0:
                # only copy (gemm -> scatter)
                copyp2p_kernel[(num_ctas_rs, )](
                    cur_out, src, num_elems_per_rank,
                    num_warps=num_warps_rs,
                    BLOCK_SIZE=num_warps_rs * 64 * 16)
            elif stage > num_ranks - 1:
                # all gather phase, just copy
                _wait_eq_cuda(scatter_signal_bufs[rank][segment], scatter_stream)
                copyp2p_kernel[(num_ctas_rs, )](
                    cur_out,
                    scatter_bufs[rank][m_start:m_end],
                    num_elems_per_rank,
                    num_warps=num_warps_rs,
                    BLOCK_SIZE=num_warps_rs * 64 * 16)
            else:
                # add and copy to peer rank
                _wait_eq_cuda(scatter_signal_bufs[rank][segment], scatter_stream)
                add_continuous_kernel[(num_ctas_rs, )](
                    src, scatter_bufs[rank][m_start:m_end],
                    cur_out, num_elems_per_rank,
                    num_warps=num_warps_rs,
                    BLOCK_SIZE=num_warps_rs * 64 * 16)
            if stage == (num_ranks - 1) + (num_ranks - 1):
                break
            # set signal
            kernel_set_signal[(1, )](
                scatter_signal_bufs[to_rank][segment], 1)

    current_stream.wait_stream(scatter_stream)

    # return output
    return scatter_bufs[rank]

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    parser.add_argument("-m", type=int, default=2048, help="Number of rows in matrix A")
    parser.add_argument("-n", type=int, default=3584, help="Number of columns in matrix B")
    parser.add_argument("-k", type=int, default=14336, help="Common dimension between matrices A and B")
    parser.add_argument("--iters", default=10, type=int, help="perf iterations")
    parser.add_argument("--bench", default=False, action="store_true", help="benchmark")
    parser.add_argument("--check", default=False, action="store_true", help="correctness check")
    return parser.parse_args()

configs = []
configs.append(
triton.testing.Benchmark(
    x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
    x_vals=[
        # GEMM RS
        # (1024, 4096, 12288),
        (1024, 4096, 14336),
        (2048, 3584, 14336),
        # (2048, 3584, 20480),
        # (4096, 8192, 36864),
        # (8192, 8192, 36864),
    ],
    line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
    # Possible values for `line_arg`
    # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
    line_vals=["torch", "triton", "speedup"],  # Label name for the lines
    line_names=["Torch", "Triton", "SpeedUp"],  # Line styles
    # line_vals=["triton"],  # Label name for the lines
    # line_names=["Triton"],  # Line styles
    styles=[("green", "-"), ("blue", "-"), ("yellow", "-")],
    ylabel="ms",  # Label name for the y-axis
    plot_name="matmul-rs-" + "fp16",
    # args={"fp8_inputs": fp8_inputs},
    args={},
))

class BenchmarkRunner:
    def __init__(self):
        self.results_torch = [{}]
        self.results_triton = [{}]

    def run_benchmark(self):
        @triton.testing.perf_report(configs)
        def benchmark(M, N, K, provider):
            local_K = K // TP_GROUP.size()
            quantiles = [0.5, 0.2, 0.8]
            scale = TP_GROUP.rank() + 10
            input, weight, bias = _make_data(M, N, local_K, scale, dtype, device)

            M_PER_CHUNK = M // WORLD_SIZE
            m_chunk_num = (M + M_PER_CHUNK - 1) // M_PER_CHUNK

            gemm_bufs = pyrocshmem.rocshmem_create_tensor_list_intra_node((M, N), dtype=dtype)
            # barrier_bufs = pyrocshmem.rocshmem_create_tensor_list_intra_node((m_chunk_num,), dtype=rocSHMEM_SIGNAL_DTYPE)
            barrier_bufs = pyrocshmem.rocshmem_create_tensor_list_intra_node((m_chunk_num,), dtype=torch.int)
            barrier_bufs[RANK].fill_(0)

            # scatter bufs
            scatter_bufs = pyrocshmem.rocshmem_create_tensor_list_intra_node((M, N), dtype=dtype)
            # scatter_signal_bufs = pyrocshmem.rocshmem_create_tensor_list_intra_node((WORLD_SIZE,), dtype=rocSHMEM_SIGNAL_DTYPE)
            scatter_signal_bufs = pyrocshmem.rocshmem_create_tensor_list_intra_node((WORLD_SIZE,), dtype=torch.int)
            scatter_signal_bufs[RANK].fill_(0)

            # sync bufs
            sync_bufs = pyrocshmem.rocshmem_create_tensor_list_intra_node((WORLD_SIZE,), dtype=torch.int)
            sync_bufs[RANK].fill_(0)
            sync_bufs_ptr = torch.tensor([t.data_ptr() for t in sync_bufs], device=torch.cuda.current_device(),
                                            requires_grad=False)

            current_stream = torch.cuda.current_stream()
            scatter_stream = torch.cuda.Stream(priority=-1)

            torch.cuda.synchronize()
            torch.distributed.barrier()
            torch.cuda.empty_cache()

            if provider == "torch":
                ms, min_ms, max_ms = \
                    triton.testing.do_bench_multiprocess(
                        lambda: torch_gemm_rs(
                            input, weight, bias, TP_GROUP, transpose_weight=False, all_reduce=True), quantiles=quantiles)
                self.results_torch[0] = {'ms': ms}

            if provider == 'triton':
                ms, min_ms, max_ms = triton.testing.do_bench_multiprocess(
                    lambda: triton_gemm_rs(
                        input, weight, 
                        gemm_bufs,
                        barrier_bufs,
                        scatter_bufs,
                        scatter_signal_bufs,
                        sync_bufs_ptr,
                        RANK, WORLD_SIZE, M, N, local_K,
                        num_sms,
                        current_stream,
                        scatter_stream), quantiles=quantiles)
                self.results_triton[0] = {'ms': ms}

            if provider == 'speedup':
                ms1 = self.results_torch[0]['ms']
                ms2 = self.results_triton[0]['ms']
                perf = lambda ms1, ms2: (1.0/ms2 - 1.0/ms1) / (1.0/ms1)
                return perf(ms1, ms2), perf(ms1, ms2), perf(ms1, ms2)

            perf = lambda ms: ms * 1e-3
            return perf(ms), perf(max_ms), perf(min_ms)
        benchmark.run(show_plots=False, print_data=(RANK==0))

if __name__ == "__main__":
    # init
    args = parse_args()
    RANK, LOCAL_RANK, WORLD_SIZE, TP_GROUP = init()
    print(f"RANK: {RANK} LOCAL_RANK:{LOCAL_RANK} WORLD_SIZE:{WORLD_SIZE}")

    perf_warmups = 10
    perf_iters = 10
    
    torch.cuda.synchronize()
    # shmem init
    pyrocshmem.init_rocshmem_by_uniqueid(TP_GROUP)

    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    device = "cuda"

    if args.bench:
        dtype = torch.float16
        runner = BenchmarkRunner()
        runner.run_benchmark()
        torch.distributed.destroy_process_group()
        exit(0)

    # M, N, K = 512, 512, 64
    M, N, K = 2048, 3584, 14336
    # M, N, K = 2048, 2048, 12288
    
    local_K = K // TP_GROUP.size()
    scale = 10
    dtype = torch.float16
    a, b, bias = _make_data(M, N, local_K, scale, dtype, device)
    assert bias is None

    # check data
    atol, rtol = THRESHOLD_MAP[dtype], THRESHOLD_MAP[dtype]

    m_per_rank = M // WORLD_SIZE
    M_PER_CHUNK = M // WORLD_SIZE
    m_chunk_num = (M + M_PER_CHUNK - 1) // M_PER_CHUNK
    
    gemm_bufs = pyrocshmem.rocshmem_create_tensor_list_intra_node((M, N), dtype=dtype)
    # barrier_bufs = pyrocshmem.rocshmem_create_tensor_list_intra_node((m_chunk_num,), dtype=rocSHMEM_SIGNAL_DTYPE)
    barrier_bufs = pyrocshmem.rocshmem_create_tensor_list_intra_node((m_chunk_num,), dtype=torch.int)
    barrier_bufs[RANK].fill_(0)
    
    # scatter bufs
    scatter_bufs = pyrocshmem.rocshmem_create_tensor_list_intra_node((M, N), dtype=dtype)
    # scatter_signal_bufs = pyrocshmem.rocshmem_create_tensor_list_intra_node((WORLD_SIZE,), dtype=rocSHMEM_SIGNAL_DTYPE)
    scatter_signal_bufs = pyrocshmem.rocshmem_create_tensor_list_intra_node((WORLD_SIZE * 2,), dtype=torch.int)
    scatter_signal_bufs[RANK].fill_(0)
    
    # sync bufs
    sync_bufs = pyrocshmem.rocshmem_create_tensor_list_intra_node((WORLD_SIZE,), dtype=torch.int)
    sync_bufs[RANK].fill_(0)
    sync_bufs_ptr = torch.tensor([t.data_ptr() for t in sync_bufs], device=torch.cuda.current_device(),
                                    requires_grad=False)
    # counter_buf
    counter_buf = torch.zeros((WORLD_SIZE, ), dtype=torch.int32, device=device)

    current_stream = torch.cuda.current_stream()
    scatter_stream = torch.cuda.Stream(priority=-1)

    torch.cuda.synchronize()
    torch.distributed.barrier()
    
    if args.check:
        torch.cuda.empty_cache()
        input_list = [_make_data(M, N, local_K, scale, dtype, device) for _ in range(args.iters)]
        dist_out_list, torch_out_list = [], []

        # torch impl
        for input, weight, bias in input_list:
            torch_out = torch_gemm_rs(
                input, weight, bias, TP_GROUP, transpose_weight=False, all_reduce=True)
            torch_out_list.append(torch_out.cpu())

        # barrier_all_on_stream(RANK, WORLD_SIZE, sync_bufs_ptr, current_stream)
        pyrocshmem.rocshmem_barrier_all()
        torch.cuda.synchronize()
        torch.distributed.barrier()

        # dist triton impl
        for idx, (input, weight, bias) in enumerate(input_list):
            # if RANK == 0:
            #     print(f"iter {idx}")
            dist_out = triton_gemm_rs(
                        input, weight,
                        gemm_bufs,
                        barrier_bufs,
                        scatter_bufs,
                        scatter_signal_bufs,
                        sync_bufs_ptr,
                        counter_buf,
                        RANK, WORLD_SIZE, M, N, local_K,
                        num_sms,
                        current_stream,
                        scatter_stream)
            torch.cuda.synchronize()
            dist_out_list.append(dist_out.cpu())

        # verify
        for idx, (torch_out, dist_out) in enumerate(zip(torch_out_list, dist_out_list)):
            try:
                torch.testing.assert_close(torch_out, dist_out, atol=atol, rtol=rtol)
                print(f"RANK[{RANK}]: pass.")
            except Exception as e:
                raise e
        # print(f"RANK[{RANK}]: pass.")
        torch.distributed.destroy_process_group()
        exit(0)
    
    ctx = get_torch_prof_ctx(False)
    with ctx:
        # torch.cuda.empty_cache()
        torch_out, torch_perf = perf_func(
            partial(
                torch_gemm_rs, a, b, bias, TP_GROUP, transpose_weight=False, all_reduce=True),
                iters = perf_iters,
                warmup_iters = perf_warmups)

        torch.cuda.synchronize()
        torch.distributed.barrier()
        triton_out, triton_perf = perf_func(
            partial(triton_gemm_rs,
                        a, b,
                        gemm_bufs,
                        barrier_bufs,
                        scatter_bufs,
                        scatter_signal_bufs,
                        sync_bufs_ptr,
                        counter_buf,
                        RANK, WORLD_SIZE, M, N, local_K,
                        num_sms,
                        current_stream,
                        scatter_stream),
            iters = perf_iters,
            warmup_iters = perf_warmups)
    
    try:
        torch.testing.assert_close(triton_out, torch_out, atol=atol, rtol=rtol)
    except Exception as e:
        raise e

    tflops = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    dist_print(f"triton #{RANK}", triton_perf, need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))
    dist_print(f"torch #{RANK}", torch_perf, need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))
    # dist_print(f"triton #{RANK}", tflops(triton_perf), need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))
    # dist_print(f"torch #{RANK}", tflops(torch_perf), need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))
    

    pyrocshmem.rocshmem_finalize()
    torch.distributed.destroy_process_group()

    # best_config = kernel_gemm_rs_producer_persistent.best_config
    # print("Best Config:", best_config)