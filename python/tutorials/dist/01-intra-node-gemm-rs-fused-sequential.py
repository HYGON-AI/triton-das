import triton
import torch

import argparse
import random
import numpy as np
import os
import datetime
from typing import Optional

import triton.language as tl
import pynvshmem

from triton.language.extra import libshmem_device
from triton.language.extra.hip import libdevice

from cuda import cudart
# from triton_dist.utils import HIP_CHECK

# from triton_dist.kernels.amd.gemm_reduce_scatter import get_hip_autotune_config

# from triton_dist.utils import NVSHMEM_SIGNAL_DTYPE

from hip import hip

# from triton_dist.utils import (
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

@triton.autotune(
    configs=[
        triton.Config(
            {
                'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'M_PER_COPY_CHUNK':
                1024, 'waves_per_eu': 2
            }, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
    # use_cuda_graph=True,
)
# @triton.autotune(
#     configs=get_hip_autotune_config(),
#     key=['M', 'N', 'K'],
# )
@triton.jit
def kernel_gemm_rs_producer_persistent(
    a_ptr, b_ptr,
    scatter_buf_ptr,
    # counter_ptr,
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
    M_PER_COPY_CHUNK: tl.constexpr, #
    NUM_SMS: tl.constexpr,  #
    ):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tid = libdevice.thread_idx(axis=0)  # noqa: F841

    # NOTE: There is currently a bug in blackwell pipelining that means it can't handle a value being
    # used in both the prologue and epilogue, so we duplicate the counters as a work-around.
    tile_id_c = start_pid - NUM_SMS

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    M_per_rank = M // num_ranks
    num_pid_m_per_rank = M_per_rank // BLOCK_SIZE_M

    ENABLE_SWIZZLE = True
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=False):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)

        if ENABLE_SWIZZLE:
            m_rank = pid_m // num_pid_m_per_rank
            pid_m_intra_rank = pid_m - m_rank * num_pid_m_per_rank
            # original rank and node_id
            m_node_id = m_rank // num_ranks
            m_local_rank = m_rank % num_ranks
            swizzle_m_node_id = (m_node_id + rank + 1) % 1 #nnodes
            swizzle_m_local_rank = (m_local_rank + rank + 1) % num_ranks
            swizzle_m_rank = swizzle_m_node_id * num_ranks + swizzle_m_local_rank
            # perform swizzle
            pid_m = swizzle_m_rank * num_pid_m_per_rank + pid_m_intra_rank
            rank_offset = pid_m // num_pid_m_per_rank
        else:
            rank_offset = pid_m // num_pid_m_per_rank

        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

            a = tl.load(a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0)
            accumulator = tl.dot(a, b, accumulator)

        # tile_id_c += NUM_SMS
        # pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)

        # offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        target_m = ((pid_m * BLOCK_SIZE_M % M_per_rank) + M_per_rank * rank)
        offs_cm = target_m + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        # # Modif: get scatter buffer ptr
        dtype = a_ptr.dtype.element_ty
        # c_ptr = tl.load(scatter_buf_ptr + rank_offset).to(tl.pointer_type(dtype))
        c_ptr = libshmem_device.remote_ptr(scatter_buf_ptr, rank_offset).to(tl.pointer_type(dtype))
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        c = accumulator.to(dtype)

        tl.store(c_ptrs, c, mask=c_mask)


@triton.autotune(
    configs=[
            triton.Config(
                {
                    'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'M_PER_COPY_CHUNK':
                    128, 'waves_per_eu': 2
                }, num_warps=8, num_stages=2),
        ],
        key=['M', 'N', 'K'],
        # use_cuda_graph=True,
    )
# @triton.autotune(
#     configs=get_hip_autotune_config(),
#     key=['M', 'N', 'K'],
# )
@triton.jit
def kernel_gemm_rs_producer_fuse_scatter(
    # Pointers to matrices
    a_ptr, b_ptr, scatter_bufs_ptr, rank, num_ranks,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am, stride_ak,  #
    stride_bk, stride_bn,  #
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr, M_PER_COPY_CHUNK: tl.constexpr  #
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
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # rank swizzle
    M_per_rank = M // num_ranks
    num_pid_m_per_copy_chunk = M_PER_COPY_CHUNK // BLOCK_SIZE_M
    chunk_offset = pid_m // (num_pid_m_per_copy_chunk * num_ranks)
    rank_offset = pid_m % (num_pid_m_per_copy_chunk * num_ranks) // num_pid_m_per_copy_chunk
    block_offset = pid_m % num_pid_m_per_copy_chunk

    # rank_swizzle_offset = M_per_rank * nxt_rank // BLOCK_SIZE_M
    # pid_m = (pid_m + rank_swizzle_offset) % num_pid_m
    rank_offset = (rank_offset + rank + 1) % num_ranks
    pid_m = (rank_offset * M_per_rank + chunk_offset * M_PER_COPY_CHUNK + block_offset * BLOCK_SIZE_M) // BLOCK_SIZE_M
    thread_idx = libdevice.thread_idx(axis=0)  # noqa: F841

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

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    target_m = ((pid_m * BLOCK_SIZE_M % M_per_rank) + M_per_rank * rank)
    offs_cm = target_m + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptr = tl.load(scatter_bufs_ptr + rank_offset).to(tl.pointer_type(dtype))
    c_ptr = tl.multiple_of(c_ptr, 16)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

@triton.jit
def kernel_consumer_reduce(
    c_ptr,  # [M, N]
    out_ptr,  # [M_per_rank, N]
    # shape of matrix
    M_per_rank,
    N,
    rank,
    num_ranks: tl.constexpr,
    # tile size
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = tl.where(offs < M_per_rank * N, offs, 0)

    out_ptrs = out_ptr + offs

    accum = tl.zeros((BLOCK_SIZE, ), dtype=out_ptr.dtype.element_ty)
    for i in range(0, num_ranks):
        cur_rank = (i + rank + 1) % num_ranks
        c_ptrs = c_ptr + offs + cur_rank * M_per_rank * N
        data = tl.load(c_ptrs)
        accum += data

    tl.store(out_ptrs, accum)

def ring_reduce_after_scatter(
    rank,
    num_ranks,
    scatter_out,  # [M, N]
    stream,
):
    M, N = scatter_out.shape
    M_per_rank = M // num_ranks
    output = torch.empty((M_per_rank, N), dtype=scatter_out.dtype, device=scatter_out.device)
    grid = lambda META: (triton.cdiv(M_per_rank * N, META["BLOCK_SIZE"]), )
    with torch.cuda.stream(stream):
        kernel_consumer_reduce[grid](
            scatter_out,
            output,
            M_per_rank,
            N,
            rank=rank,
            num_ranks=num_ranks,
            BLOCK_SIZE=2048,
            num_warps=2,
        )
    return output

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
    scatter_bufs,
    scatter_bufs_ptr,
    sync_bufs_ptr,
    rank, num_ranks, M, N, K,
    num_sms,
    current_stream,
    scatter_stream
):
    barrier_all_on_stream(rank, num_ranks, sync_bufs_ptr, current_stream)
    M_per_rank = M // num_ranks

    with torch.cuda.stream(current_stream):
        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
        kernel_gemm_rs_producer_fuse_scatter[grid](
            a, b,
            scatter_bufs_ptr,
            rank, num_ranks, M, N, K,
            a.stride(0), a.stride(1),
            b.stride(1), b.stride(0),
            N, 1
        )

    barrier_all_on_stream(rank, num_ranks, sync_bufs_ptr, current_stream)

    output = torch.empty((M_per_rank, N), dtype=a.dtype, device=a.device)
    grid = lambda META: (triton.cdiv(M_per_rank * N, META["BLOCK_SIZE"]), )
    with torch.cuda.stream(current_stream):
        kernel_consumer_reduce[grid](
            scatter_bufs[rank],
            output,
            M_per_rank,
            N,
            rank=rank,
            num_ranks=num_ranks,
            BLOCK_SIZE=2048,
            num_warps=2,
        )

    # output = ring_reduce_after_scatter(rank, num_ranks, scatter_buf, current_stream)

    return output

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("M", type=int)
    # parser.add_argument("N", type=int)
    # parser.add_argument("K", type=int)
    # parser.add_argument("--warmup", default=5, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=10, type=int, help="perf iterations")
    parser.add_argument("--bench", default=False, action="store_true", help="benchmark")
    
    # parser.add_argument("--verify-iters", default=10, type=int)
    # parser.add_argument("--dtype", default="float16", type=str, help="data type")
    # parser.add_argument("--profile", default=False, action="store_true", help="dump torch.profiler.profile")
    parser.add_argument("--check", default=False, action="store_true", help="correctness check")
    return parser.parse_args()

def test_nvshmem_signal(rank):

    @triton.jit
    def _pingpong(t, iters):
        # pingpong for rank 0-1, 2-3, ...
        mype = libshmem_device.my_pe()
        # thread_idx = tid(axis=0)
        thread_idx = libdevice.thread_idx(axis=0)
        if thread_idx == 0:
            for n in range(iters):
                if mype == 0:
                    libshmem_device.signal_wait_until(t, libshmem_device.NVSHMEM_CMP_EQ, 1 + n)
                    libshmem_device.signal_op(
                        t,
                        1 + n,
                        libshmem_device.NVSHMEM_SIGNAL_SET,
                        1,
                    )
                elif mype == 1:
                    libshmem_device.signal_op(
                        t,
                        1 + n,
                        libshmem_device.NVSHMEM_SIGNAL_SET,
                        0,
                    )
                    libshmem_device.signal_wait_until(t, libshmem_device.NVSHMEM_CMP_EQ, 1 + n)
        # __syncthreads()
        libdevice.__syncthreads()

    print("test nvshmemx_signal with pingpong...")
    t = pynvshmem.nvshmem_create_tensor((1, ), NVSHMEM_SIGNAL_DTYPE)
    t.fill_(0)
    # nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    pynvshmem.nvshmem_barrier_all()
    _pingpong[(1, )](t, 100, num_warps=1)
    # nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    pynvshmem.nvshmem_barrier_all()
    
    torch.cuda.synchronize()
    # if libshmem_device.my_pe() == 0:
    if rank == 0:
        print('result: ', t.to(torch.int32))
        try:
            torch.testing.assert_close(t.to(torch.int32), torch.ones([1], dtype=torch.int32, device="cuda") * 100)
        except Exception as e:
            print("❌ nvshmemx_signal with pingpong failed")
            raise e
        else:
            print("✅ nvshmemx_signal with pingpong pass")
    # nvshmem_free_tensor_sync(t)
    torch.distributed.barrier()
    torch.cuda.synchronize()
    del t

configs = []
configs.append(
triton.testing.Benchmark(
    x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
    x_vals=[
        ## GEMM RS
        (1024, 4096, 12288),
        (1024, 4096, 14336),
        (2048, 3584, 14336),
        (2048, 3584, 20480),
        (4096, 8192, 36864),
        (8192, 8192, 36864),
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

            # M_PER_CHUNK = M // WORLD_SIZE
            # m_chunk_num = (M + M_PER_CHUNK - 1) // M_PER_CHUNK
            
            scatter_bufs = pynvshmem.nvshmem_create_tensor_list_intra_node((M, N), dtype=dtype)
            scatter_bufs_ptr = torch.tensor([t.data_ptr() for t in scatter_bufs], device=torch.cuda.current_device(),
                                            requires_grad=False)

            # sync buf
            sync_bufs = pynvshmem.nvshmem_create_tensor_list_intra_node((WORLD_SIZE,), dtype=torch.int)
            sync_bufs[RANK].fill_(0)
            sync_bufs_ptr = torch.tensor([t.data_ptr() for t in sync_bufs], device=torch.cuda.current_device(),
                                            requires_grad=False)
            
            current_stream = torch.cuda.current_stream()
            scatter_stream = torch.cuda.Stream(priority=-1)

            if provider == "torch":
                ms, min_ms, max_ms = \
                    triton.testing.do_bench_multiprocess(
                        lambda: torch_gemm_rs(
                            input, weight, bias, TP_GROUP, transpose_weight=False, all_reduce=False), quantiles=quantiles)
                self.results_torch[0] = {'ms': ms}

            if provider == 'triton':
                ms, min_ms, max_ms = triton.testing.do_bench_multiprocess(
                    lambda: triton_gemm_rs(
                        input, weight, 
                        scatter_bufs,
                        scatter_bufs_ptr,
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
    os.environ["TRITON_HIP_USE_BLOCK_PINGPONG"] = "1"
    RANK, LOCAL_RANK, WORLD_SIZE, TP_GROUP = init()
    print(f"RANK: {RANK} LOCAL_RANK:{LOCAL_RANK} WORLD_SIZE:{WORLD_SIZE}")

    perf_warmups = 10
    perf_iters = 10
    
    torch.cuda.synchronize()
    # shmem init
    pynvshmem.init_nvshmem_by_uniqueid(TP_GROUP)

    # test_nvshmem_signal(RANK)
    # torch.distributed.destroy_process_group()
    # print("exit")
    # exit(0)

    # M, N, K = 512, 512, 64
    M, N, K = 2048, 3584, 14336
    # M, N, K = 2048, 2048, 12288
    
    local_K = K // TP_GROUP.size()
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count

    scale = 10
    device = "cuda"
    dtype = torch.float16
    a, b, bias = _make_data(M, N, local_K, scale, dtype, device)
    assert bias is None
    stride_bk, stride_bn = b.stride(1), b.stride(0)
    # check data
    atol, rtol = THRESHOLD_MAP[dtype], THRESHOLD_MAP[dtype]

    m_per_rank = M // WORLD_SIZE
    M_PER_CHUNK = M // WORLD_SIZE
    m_chunk_num = (M + M_PER_CHUNK - 1) // M_PER_CHUNK

    scatter_bufs = pynvshmem.nvshmem_create_tensor_list_intra_node((M, N), dtype=dtype)
    scatter_bufs_ptr = torch.tensor([t.data_ptr() for t in scatter_bufs], device=torch.cuda.current_device(),
                                    requires_grad=False)

    # sync buf
    sync_bufs = pynvshmem.nvshmem_create_tensor_list_intra_node((WORLD_SIZE,), dtype=torch.int)
    sync_bufs[RANK].fill_(0)
    sync_bufs_ptr = torch.tensor([t.data_ptr() for t in sync_bufs], device=torch.cuda.current_device(),
                                    requires_grad=False)

    # scatter signal buf
    # scatter_signal_bufs = pynvshmem.nvshmem_create_tensor_list_intra_node((WORLD_SIZE,), dtype=torch.int)
    
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
                input, weight, bias, TP_GROUP, transpose_weight=False, all_reduce=False)
            torch_out_list.append(torch_out)

        # barrier_all_on_stream(RANK, WORLD_SIZE, sync_bufs_ptr, current_stream)
        pynvshmem.nvshmem_barrier_all()
        torch.cuda.synchronize()
        torch.distributed.barrier()

        # dist triton impl
        torch.cuda.empty_cache()
        for idx, (input, weight, bias) in enumerate(input_list):
            if RANK == 0:
                print(f"iter {idx}")
            dist_out = triton_gemm_rs(
                        input, weight,
                        scatter_bufs,
                        scatter_bufs_ptr,
                        sync_bufs_ptr,
                        RANK, WORLD_SIZE, M, N, local_K,
                        num_sms,
                        current_stream,
                        scatter_stream)
            torch.cuda.synchronize()
            dist_out_list.append(dist_out)

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
        
    if args.bench:
        device = "cuda"
        dtype = torch.float16
        runner = BenchmarkRunner()
        runner.run_benchmark()
        torch.distributed.destroy_process_group()
        exit(0)
    
    ctx = get_torch_prof_ctx(False)
    with ctx:
        torch.cuda.empty_cache()
        torch_out, torch_perf = perf_func(
            partial(
                torch_gemm_rs, a, b, bias, TP_GROUP, transpose_weight=False, all_reduce=False),
                iters = perf_iters,
                warmup_iters = perf_warmups)

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.distributed.barrier()
        triton_out, triton_perf = perf_func(
            partial(triton_gemm_rs,
                        a, b,
                        scatter_bufs,
                        scatter_bufs_ptr,
                        sync_bufs_ptr,
                        RANK, WORLD_SIZE, M, N, local_K,
                        num_sms,
                        current_stream,
                        scatter_stream),
            iters = perf_iters,
            warmup_iters = perf_warmups)

    torch.cuda.synchronize()
    torch.distributed.barrier()
    torch.cuda.synchronize()
    
    # try:
    #     torch.testing.assert_close(triton_out, torch_out, atol=atol, rtol=rtol)
    # except Exception as e:
    #     # print(f"rank {RANK} torch: {torch_out}\n")
    #     # print(f"rank {RANK} triton: {triton_out}\n")
    #     raise e

    dist_print(f"triton #{RANK}", triton_perf, need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))
    dist_print(f"torch #{RANK}", torch_perf, need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))

    torch.distributed.destroy_process_group()

    # best_config = kernel_gemm_rs_producer_persistent.best_config
    # print("Best Config:", best_config)