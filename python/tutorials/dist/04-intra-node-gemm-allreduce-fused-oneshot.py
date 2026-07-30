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

from triton.language.extra.hip.libdevice import (thread_idx, load_acquire_system, store_release_system)

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
def tile_id_to_index_range(
    tile_id,
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    tile_in_group = tile_id % num_pid_in_group
    pid_m = first_pid_m + (tile_in_group % group_size_m)
    pid_n = tile_in_group // group_size_m

    rm_start = pid_m * BLOCK_SIZE_M
    rn_start = pid_n * BLOCK_SIZE_N

    # clamp to the maximum valid index (M-1, N-1)
    max_m = M - 1
    max_n = N - 1

    # generate indices
    rm = rm_start + tl.arange(0, BLOCK_SIZE_M)
    rn = rn_start + tl.arange(0, BLOCK_SIZE_N)

    rm = tl.minimum(rm, max_m)
    rn = tl.minimum(rn, max_n)

    return rm, rn, rm_start, rn_start

@triton.jit
def offset_for_tile(local_tile_id, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M, M_local, N_local):
    rm, rn, rm_start, rn_start = tile_id_to_index_range(
        local_tile_id, M_local, N_local, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )
    c_mask = (rm[:, None] < M_local) & (rn[None, :] < N_local)
    return rm, rn, c_mask, rm_start, rn_start

@triton.jit
def extract_submask_and_offset(
    rm,
    rn,
    mask,
    rm_start,
    rn_start,
    start_row,
    start_col,
    SUB_BLOCK_SIZE_M: tl.constexpr,
    SUB_BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    stride_cm_local: tl.constexpr,
    stride_cn_local: tl.constexpr,
):
    # Create indices for the sub-block
    sub_rm = tl.arange(0, SUB_BLOCK_SIZE_M) + start_row
    sub_rn = tl.arange(0, SUB_BLOCK_SIZE_N) + start_col

    # Create a 2D grid of indices for the sub-block
    sub_rm_2d = sub_rm[:, None]  # Shape: (SUB_BLOCK_SIZE_M, 1)
    sub_rn_2d = sub_rn[None, :]  # Shape: (1, SUB_BLOCK_SIZE_N)

    # Compute the sub-mask
    sub_mask = (sub_rm_2d < BLOCK_SIZE_M) & (sub_rn_2d < BLOCK_SIZE_N)

    # Compute the sub-offset relative to the start of the tile
    sub_offset = ((rm_start + sub_rm_2d) * stride_cm_local) + ((rn_start + sub_rn_2d) * stride_cn_local)

    return sub_mask, sub_offset


# autotune
'''
configs = [
    triton.Config(
        {"BLOCK_SIZE_M": BLOCK_SIZE_M, "BLOCK_SIZE_N": BLOCK_SIZE_N, "BLOCK_SIZE_K": BLOCK_SIZE_K,
         "GROUP_SIZE_M": GROUP_SIZE_M},
        num_warps=num_warps, num_stages=num_stages)
        for BLOCK_SIZE_M in [16, 32, 64, 128, 256]
        for BLOCK_SIZE_N in [16, 32, 64, 128, 256]
        for BLOCK_SIZE_K in [128]
        for GROUP_SIZE_M in [32, 64, 128]
        for num_warps in [1, 2, 4, 8, 16]
        for num_stages in [1, 2]
]
'''
configs = [
    triton.Config(
        {
            'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 4,
        }, num_warps=4, num_stages=1, num_ctas=1),
]
# '''
@triton.autotune(
    configs=configs,
    key=['M', 'N', 'K'],
    # perf_debug=False,
    # use_cuda_graph=True,
)
@triton.jit
def persistent_gemm_all_reduce(
    A, B,
    output, #c_global,
    scatter_buf_ptr,
    scatter_bufs_ptr, #C
    # bias_ptr,
    # P,
    # locks,
    scatter_signal_buf_ptr,
    scatter_signal_bufs_ptr, #tile_completed,
    M,
    N,
    K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # stride_bias,
    NUM_SMS: tl.constexpr,
    # STREAMK_TILES: tl.constexpr,
    # NUM_XCDS: tl.constexpr,
    # BIAS: tl.constexpr,
    EVEN_K: tl.constexpr,
    # heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    # pid = tl.program_id(0)
    tile_id = tl.program_id(0)
    
    dtype = A.dtype.element_ty

    # if NUM_XCDS != 1:
    #     pid = (pid % NUM_XCDS) * (NUM_SMS // NUM_XCDS) + (pid // NUM_XCDS)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    acc_dtype = tl.float32 #if C.type.element_ty != tl.int8 else tl.int32

    # '''
    # for tile_id in range(pid, total_tiles, NUM_SMS):
    # if COLLECT_TIMESTAMPS:
    #     timestamp = read_realtime()
    #     tl.atomic_min(mm_begin_timestamp_ptr + tile_id, timestamp)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m

    rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

    rk = tl.arange(0, BLOCK_SIZE_K)
    rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
    rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
    B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

    tl.assume(pid_m > 0)
    tl.assume(pid_n > 0)

    loop_k = tl.cdiv(K, BLOCK_SIZE_K)
    if not EVEN_K:
        loop_k -= 1

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
    for k in range(0, loop_k):
        a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
        b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
        acc += tl.dot(a, b)
        A_BASE += BLOCK_SIZE_K * stride_ak
        B_BASE += BLOCK_SIZE_K * stride_bk

    if not EVEN_K:
        k = loop_k
        rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
        A_BASE = tl.multiple_of(A_BASE, (1, 16))
        B_BASE = tl.multiple_of(B_BASE, (16, 1))
        a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
        b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0)
        acc += tl.dot(a, b)

    # Store results so other ranks can see them
    # c = acc.to(C.type.element_ty)
    c = acc.to(dtype)
    rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
    rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    C = scatter_buf_ptr
    # C = tl.load(scatter_bufs_ptr + cur_rank).to(tl.pointer_type(dtype))
    offs_c = rm[:, None] * stride_cm + rn[None, :] * stride_cn
    C_ = C + offs_c
    tl.store(C_, c, c_mask)

    # Signal to other ranks that we have completed this tile
    for remote in range(world_size):
        if remote != cur_rank:
            remote_base_ptr = tl.load(scatter_signal_bufs_ptr + remote).to(tl.pointer_type(tl.int32))
            tl.atomic_add(remote_base_ptr + tile_id, 1, scope="sys", sem="release")

    # local_base_ptr = tl.load(scatter_signal_bufs_ptr + cur_rank).to(tl.pointer_type(tl.int32))
    while tl.atomic_cas(scatter_signal_buf_ptr + tile_id, world_size - 1, 0, scope="sys", sem="acquire") != (world_size - 1):
        pass

    acc1 = acc
    # intra-node reduce
    for i in range(1, world_size):
        rank_id = (cur_rank + i) % world_size
        remote_ptr = tl.load(scatter_bufs_ptr + rank_id).to(tl.pointer_type(tl.float16))
        acc1 += tl.load(remote_ptr + offs_c, mask=c_mask)
    acc_f16 = acc1.to(tl.float16)
    tl.store(output + offs_c, acc_f16, mask=c_mask, cache_modifier=".wt")

    '''
    # All ranks have completed this tile, so we can reduce
    rm, rn, mask, rm_start, rn_start = offset_for_tile(tile_id, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M, M, N)

    # Calculate the number of sub-tiles in each dimension
    num_sub_tiles_m = tl.cdiv(BLOCK_SIZE_M, BLOCK_SIZE_M)
    num_sub_tiles_n = tl.cdiv(BLOCK_SIZE_N, BLOCK_SIZE_N)
    total_sub_tiles = num_sub_tiles_m * num_sub_tiles_n

    for sub_tile_idx in range(0, total_sub_tiles):
        # Calculate start_row and start_col for the current sub-tile
        start_row = (sub_tile_idx // num_sub_tiles_n) * BLOCK_SIZE_M
        start_col = (sub_tile_idx % num_sub_tiles_n) * BLOCK_SIZE_N

        # Translate to global
        sub_mask, sub_offset = extract_submask_and_offset(
            rm,
            rn,
            mask,
            rm_start,
            rn_start,
            start_row,
            start_col,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            stride_cm,
            stride_cn,
        )

        # Pull data from remote and do acc, then store to output
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for remote_rank in range(world_size):
            remote_ptr = tl.load(scatter_bufs_ptr + remote_rank).to(tl.pointer_type(dtype))
            acc += tl.load(remote_ptr + sub_offset, mask=sub_mask)

        tl.store(output + sub_offset, acc, mask=sub_mask, cache_modifier=".wt")
    '''


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


# NOTE: this is for tl.atomic across rank test
# @triton.jit
# def barrier_all_ipc(rank, num_ranks, comm_buf_base_ptrs):
#     tid = libdevice.thread_idx(axis=0)  # noqa: F841
#     for i in range(num_ranks):
#         if i != rank:
#             remote_base_ptr = tl.load(comm_buf_base_ptrs + i).to(tl.pointer_type(tl.int32))
#             tl.atomic_add(remote_base_ptr, 1, scope="sys", sem="release")

#     local_base_ptr = tl.load(comm_buf_base_ptrs + rank).to(tl.pointer_type(tl.int32))
#     while tl.atomic_cas(local_base_ptr, num_ranks - 1, 0, scope="sys", sem="acquire") != (num_ranks - 1):
#         pass

#     tl.debug_barrier()


def barrier_all_on_stream(
    rank,
    num_ranks,
    sync_bufs_ptr,
    stream,
):
    with torch.cuda.stream(stream):
        barrier_all_ipc[(1, )](rank, num_ranks, sync_bufs_ptr)


def triton_gemm_allreduce(
    a, b,
    output,
    scatter_buf_ptr,
    scatter_bufs_ptr,
    scatter_signal_buf_ptr,
    scatter_signal_bufs_ptr,
    sync_bufs_ptr,
    rank, num_ranks, M, N, K,
    BLK_M: int,
    BLK_N: int,
    BLK_K: int,
    gsize_m: int,
    num_sms,
    current_stream,
    scatter_stream
):
    # barrier_all_on_stream(rank, num_ranks, scatter_signal_bufs_ptr, current_stream)
    
    # # for debug
    # torch.cuda.synchronize()
    # print(f"RANK: {RANK} scatter_signal_bufs: {scatter_signal_bufs[RANK]}")

    with torch.cuda.stream(current_stream):
        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
        # grid = lambda META: (min(
        #         num_sms,
        #         triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        #     ), )
        persistent_gemm_all_reduce[grid](
            a, b,
            output,
            scatter_buf_ptr,
            scatter_bufs_ptr,
            scatter_signal_buf_ptr,
            scatter_signal_bufs_ptr,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(1), b.stride(0),
            N, 1,
            # BLK_M, BLK_N, BLK_K, gsize_m,
            num_sms,
            K % BLK_K == 0,
            rank, num_ranks,
            # waves_per_eu=2, # to be tunned
            # num_warps=8,
            # num_stages=2,
        )

    # barrier_all_on_stream(rank, num_ranks, sync_bufs_ptr, current_stream)

    return output

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    parser.add_argument("-m", type=int, default=2048, help="Number of rows in matrix A")
    parser.add_argument("-n", type=int, default=3584, help="Number of columns in matrix B")
    parser.add_argument("-k", type=int, default=14336, help="Common dimension between matrices A and B")
    
    # default block size
    # For One Shot, use: 256x256x64
    parser.add_argument("--BLK_M", type=int, default=128, help="Block size M")
    parser.add_argument("--BLK_N", type=int, default=128, help="Block size N")
    parser.add_argument("--BLK_K", type=int, default=64, help="Block size K")
    # Best to try 1, 6 or 8
    parser.add_argument("--gsize_m", type=int, default=4, help="Grid size M")
    
    # parser.add_argument("--warmup", default=5, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=10, type=int, help="perf iterations")
    parser.add_argument("--bench", default=False, action="store_true", help="benchmark")
    
    # parser.add_argument("--verify-iters", default=10, type=int)
    # parser.add_argument("--dtype", default="float16", type=str, help="data type")
    # parser.add_argument("--profile", default=False, action="store_true", help="dump torch.profiler.profile")
    parser.add_argument("--check", default=False, action="store_true", help="correctness check")
    return vars(parser.parse_args())

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
            
            scatter_bufs = pyrocshmem.rocshmem_create_tensor_list_intra_node((M, N), dtype=dtype)
            scatter_bufs_ptr = torch.tensor([t.data_ptr() for t in scatter_bufs], device=torch.cuda.current_device(),
                                            requires_grad=False)

            # sync buf
            sync_bufs = pyrocshmem.rocshmem_create_tensor_list_intra_node((WORLD_SIZE,), dtype=torch.int)
            sync_bufs[RANK].fill_(0)
            sync_bufs_ptr = torch.tensor([t.data_ptr() for t in sync_bufs], device=torch.cuda.current_device(),
                                            requires_grad=False)
            
            current_stream = torch.cuda.current_stream()
            scatter_stream = torch.cuda.Stream(priority=-1)

            if provider == "torch":
                ms, min_ms, max_ms = \
                    triton.testing.do_bench_multiprocess(
                        lambda: torch_gemm_rs(
                            input, weight, bias, TP_GROUP, transpose_weight=False, all_reduce=True), quantiles=quantiles)
                self.results_torch[0] = {'ms': ms}

            if provider == 'triton':
                ms, min_ms, max_ms = triton.testing.do_bench_multiprocess(
                    lambda: triton_gemm_allreduce(
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

    perf_warmups = 20
    perf_iters = 10
    
    torch.cuda.synchronize()
    # shmem init
    pyrocshmem.init_rocshmem_by_uniqueid(TP_GROUP)
    
    if args["bench"]:
        dtype = torch.float16
        runner = BenchmarkRunner()
        runner.run_benchmark()
        torch.distributed.destroy_process_group()
        exit(0)
        
    M, N, K = args["m"], args["n"], args["k"]
    local_K = K // TP_GROUP.size()
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count

    scale = 10
    device = torch.cuda.current_device()
    dtype = torch.float16
    a, b, bias = _make_data(M, N, local_K, scale, dtype, device)
    assert bias is None
    stride_bk, stride_bn = b.stride(1), b.stride(0)
    # check data
    atol, rtol = THRESHOLD_MAP[dtype], THRESHOLD_MAP[dtype]
    
    max_blocks_M = triton.cdiv(args["m"], 16)
    max_blocks_N = triton.cdiv(args["n"], 16)
    total_tiles = max_blocks_M * max_blocks_N
    
    # output
    output = torch.zeros((args["m"], args["n"]), device=device, dtype=dtype)
    
    # local C
    scatter_bufs = pyrocshmem.rocshmem_create_tensor_list_intra_node((args["m"], args["n"]), dtype=dtype)
    scatter_bufs_ptr = torch.tensor([t.data_ptr() for t in scatter_bufs], device=device, requires_grad=False)
    
    # signal buf
    scatter_signal_bufs = pyrocshmem.rocshmem_create_tensor_list_intra_node((total_tiles,), dtype=torch.int)
    scatter_signal_bufs[RANK].fill_(0)
    scatter_signal_bufs_ptr = torch.tensor([t.data_ptr() for t in scatter_signal_bufs], device=device, requires_grad=False)

    # sync buf
    sync_bufs = pyrocshmem.rocshmem_create_tensor_list_intra_node((WORLD_SIZE,), dtype=torch.int)
    sync_bufs[RANK].fill_(0)
    sync_bufs_ptr = torch.tensor([t.data_ptr() for t in sync_bufs], device=device, requires_grad=False)
    
    current_stream = torch.cuda.current_stream()
    scatter_stream = torch.cuda.Stream(priority=-1)

    torch.cuda.synchronize()
    torch.distributed.barrier()
    
    if args["check"]:
        torch.cuda.empty_cache()
        input_list = [_make_data(M, N, local_K, scale, dtype, device) for _ in range(args["iters"])]
        dist_out_list, torch_out_list = [], []

        # torch impl
        for input, weight, bias in input_list:
            torch_out = torch_gemm_rs(
                input, weight, bias, TP_GROUP, transpose_weight=False, all_reduce=True)
            torch_out_list.append(torch_out)

        barrier_all_on_stream(RANK, WORLD_SIZE, sync_bufs_ptr, current_stream)
        torch.cuda.synchronize()
        torch.distributed.barrier()

        # dist triton impl
        torch.cuda.empty_cache()
        for idx, (input, weight, bias) in enumerate(input_list):
            dist_out = triton_gemm_allreduce(
                        input, weight,
                        output,
                        scatter_bufs[RANK],
                        scatter_bufs_ptr,
                        scatter_signal_bufs[RANK],
                        scatter_signal_bufs_ptr,
                        sync_bufs_ptr,
                        RANK, WORLD_SIZE, M, N, local_K,
                        args["BLK_M"],
                        args["BLK_N"],
                        args["BLK_K"],
                        args["gsize_m"],
                        num_sms,
                        current_stream,
                        scatter_stream)
            torch.cuda.synchronize()
            torch.distributed.barrier()
            dist_out_list.append(dist_out.clone())

        # if RANK == 0:
        #     print(f"RANK: {RANK} dist: {dist_out_list[0]} torch: {torch_out_list[0]}")

        # verify
        for idx, (torch_out, dist_out) in enumerate(zip(torch_out_list, dist_out_list)):
            try:
                torch.testing.assert_close(torch_out, dist_out, atol=atol, rtol=rtol)
                print(f"RANK[{RANK}]: pass.")
            except Exception as e:
                raise e
        torch.distributed.destroy_process_group()
        print(f"RANK[{RANK}]: pass.")
        exit(0)
    
    ctx = get_torch_prof_ctx(False)
    with ctx:
        torch.cuda.empty_cache()
        torch_out, torch_perf = perf_func(
            partial(
                torch_gemm_rs, a, b, bias, TP_GROUP, transpose_weight=False, all_reduce=True),
                iters = perf_iters,
                warmup_iters = perf_warmups)

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.distributed.barrier()
        triton_out, triton_perf = perf_func(
            partial(triton_gemm_allreduce,
                        a, b,
                        output,
                        scatter_bufs[RANK],
                        scatter_bufs_ptr,
                        scatter_signal_bufs[RANK],
                        scatter_signal_bufs_ptr,
                        sync_bufs_ptr,
                        RANK, WORLD_SIZE, M, N, local_K,
                        args["BLK_M"],
                        args["BLK_N"],
                        args["BLK_K"],
                        args["gsize_m"],
                        num_sms,
                        current_stream,
                        scatter_stream),
            iters = perf_iters,
            warmup_iters = perf_warmups)

    torch.cuda.synchronize()
    torch.distributed.barrier()
    torch.cuda.synchronize()
    
    try:
        torch.testing.assert_close(triton_out, torch_out, atol=atol, rtol=rtol)
    except Exception as e:
        # print(f"rank {RANK} torch: {torch_out}\n")
        # print(f"rank {RANK} triton: {triton_out}\n")
        raise e

    tflops = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    # dist_print(f"triton #{RANK}", triton_perf, need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))
    # dist_print(f"torch #{RANK}", torch_perf, need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))
    dist_print(f"triton #{RANK}", tflops(triton_perf), need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))
    dist_print(f"torch #{RANK}", tflops(torch_perf), need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))

    pyrocshmem.rocshmem_finalize()
    torch.distributed.destroy_process_group()

    best_config = persistent_gemm_all_reduce.best_config
    print("Best Config:", best_config)