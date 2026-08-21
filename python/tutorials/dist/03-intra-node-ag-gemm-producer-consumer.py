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
from typing import Optional, List

import triton.language as tl
import pyrocshmem

from triton.language.extra import libshmem_device
from triton.language.extra.hip import libdevice

from utils import HIP_CHECK

from hip import hip

from triton.language.extra.hip.libdevice import (thread_idx, load_acquire_system, atom_cas_acquire_relaxed_system)

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

def torch_ag_gemm(
    input: torch.Tensor,  # [local_M, k]
    weight: torch.Tensor,  # [local_N, K]
    transed_weight: bool,
    bias: Optional[torch.Tensor],
    TP_GROUP,
):
    local_M, K = input.shape
    world_size = TP_GROUP.size()
    if transed_weight:
        assert K == weight.shape[0]
    else:
        assert K == weight.shape[1]
        weight = weight.T
    assert input.device == weight.device
    # AG
    full_input = torch.empty((local_M * world_size, K), dtype=input.dtype, device=input.device)
    torch.distributed.all_gather_into_tensor(full_input, input, group=TP_GROUP)
    # Gemm
    output = torch.matmul(full_input, weight)

    if bias:
        assert False # not supported yet
        output = output + bias

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


@triton.jit
def kernel_set_signal(
    ptr,
    val
):
    tl.atomic_cas(ptr, 0, 1, sem="release", scope="gpu")

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


def cp_engine_producer_all_gather_full_mesh_push_multi_stream(
    rank,
    num_ranks,
    local_tensor: torch.Tensor,
    remote_tensor_buffers: List[torch.Tensor],
    one: torch.Tensor,
    M_PER_CHUNK: int,
    ag_stream_pool: List[torch.cuda.Stream],
    barrier_buffers: List[torch.Tensor],
):
    M_per_rank, N = local_tensor.shape
    chunk_num_per_rank = M_per_rank // M_PER_CHUNK
    num_stream = len(ag_stream_pool)
    rank_orders = [(rank + i) % num_ranks for i in range(num_ranks)]

    stream_offset = rank
    stream_offset = 0
    data_elem_size = local_tensor.element_size()
    barrier_elem_size = one.element_size()

    for idx, remote_rank in enumerate(rank_orders):
        if remote_rank == rank:
            stream_offset += chunk_num_per_rank
            continue
        for chunk_idx_intra_rank in range(chunk_num_per_rank):
            stream_offset += 1
            chunk_pos = rank * chunk_num_per_rank + chunk_idx_intra_rank
            stream_pos = idx % num_stream
            ag_stream = ag_stream_pool[stream_pos]
            M_dst_start_pos = rank * M_per_rank + chunk_idx_intra_rank * M_PER_CHUNK
            M_src_start_pos = chunk_idx_intra_rank * M_PER_CHUNK
            src_ptr = local_tensor.data_ptr() + M_src_start_pos * N * data_elem_size
            dst_ptr = remote_tensor_buffers[remote_rank].data_ptr() + M_dst_start_pos * N * data_elem_size
            nbytes = M_PER_CHUNK * N * data_elem_size
            cp_res = hip.hipMemcpyAsync(
                dst_ptr,
                src_ptr,
                nbytes,
                hip.hipMemcpyKind.hipMemcpyDeviceToDevice,
                ag_stream.cuda_stream,
            )
            HIP_CHECK(cp_res)
            # optim: use p2p_kernel to limit CUs that used for Copy
            # NOTE: this seems cause benchmark hang, to be debug later
            # num_ctas_rs = 16
            # num_warps_rs = 16
            # with torch.cuda.stream(ag_stream):
            #     copyp2p_kernel[(num_ctas_rs, )](
            #             remote_tensor_buffers[remote_rank][M_dst_start_pos:],
            #             local_tensor[M_src_start_pos:],
            #             M_PER_CHUNK * N,
            #             num_warps=num_warps_rs,
            #             BLOCK_SIZE=num_warps_rs * 64 * 16)

            # set_signal(barrier_buffers[remote_rank][chunk_pos].data_ptr(), 1, ag_stream)
            # cp_res = hip.hipMemcpyAsync(
            #     barrier_buffers[remote_rank].data_ptr() + chunk_pos * barrier_elem_size,
            #     one.data_ptr(),
            #     barrier_elem_size,
            #     hip.hipMemcpyKind.hipMemcpyDeviceToDevice,
            #     ag_stream.cuda_stream,
            # )
            # HIP_CHECK(cp_res)
            # optim: use set_signal kernel seems faster than Memcpy
            with torch.cuda.stream(ag_stream):
                kernel_set_signal[(1, )](
                    barrier_buffers[remote_rank][chunk_pos], 1)


@triton.jit
def wait_eq_sys(barrier_ptr, value):
    tid = thread_idx(axis=0)
    if tid == 0:
        while load_acquire_system(barrier_ptr) != value:
            pass

    tl.debug_barrier()

@triton.jit
def wait_cas_sys(barrier_ptr, cmp_ptr, value):
    tid = thread_idx(axis=0)
    if tid == 0:
        while atom_cas_acquire_relaxed_system(barrier_ptr, cmp_ptr, value) == value:
            pass

    tl.debug_barrier()

def matmul_get_configs():
    configs = []
    mma_instr_sizes = [16, None]
    kpack_options = [1, None]
    for BM in [128, 256]:
        for BN in [128, 256]:
            for BK in [64, 128]:
                for WAVES in [0, 2]:
                    for s in [2, 3, 4]:
                        for w in [4, 8]:
                            for mma_size in mma_instr_sizes:
                                for kpack in kpack_options:
                                    kwargs = {'num_stages': s, 'num_warps': w}
                                    first_arg = {
                                        'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, "BLOCK_SIZE_K": BK, "GROUP_SIZE_M": 1,
                                        'waves_per_eu': WAVES
                                    }
                                    if kpack is not None:
                                        first_arg['kpack'] = kpack
                                    if mma_size is not None:
                                        first_arg['matrix_instr_nonkdim'] = mma_size
                                    configs.append(triton.Config(first_arg, **kwargs))
    return configs

@triton.autotune(
    configs=[
        # triton.Config(
        #     {
        #         'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2,
        #         'kpack': 1, 'matrix_instr_nonkdim': 16
        #     }, num_warps=8, num_stages=2),
        # triton.Config(
        #     {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'waves_per_eu': 0},
        #     num_warps=8, num_stages=2),
        # triton.Config(
        #     {
        #         'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2,
        #         'kpack': 1, 'matrix_instr_nonkdim': 16
        #     }, num_warps=8, num_stages=2),
        # triton.Config(
        #     {
        #         'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'waves_per_eu': 0,
        #         'kpack': 1
        #     }, num_warps=8, num_stages=2),
        # triton.Config(
        #     {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'waves_per_eu': 0},
        #     num_warps=8, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
            num_warps=8, num_stages=2),
        # triton.Config(
        #     {
        #         'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'waves_per_eu': 0,
        #         'kpack': 1
        #     }, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
    use_cuda_graph=True,
)
@triton.heuristics({'EVEN_K': lambda args: args['K'] % args['BLOCK_SIZE_K'] == 0})
# @triton.autotune(configs=matmul_get_configs(), key=["M", "N", "K"])
# @triton.heuristics({'EVEN_K': lambda args: args['K'] % args['BLOCK_SIZE_K'] == 0})
@triton.jit
def kernel_consumer_gemm_persistent(
    A,
    localA,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    one,
    rank,
    world_size,
    barrier_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    M_PER_CHUNK: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = (pid % NUM_XCDS) * (NUM_SMS // NUM_XCDS) + (pid // NUM_XCDS)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    M_per_rank = M // world_size
    pid_m_per_rank = tl.cdiv(M_per_rank, BLOCK_SIZE_M)

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32
    for tile_id in range(pid, total_tiles, NUM_SMS):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        ## swizzle
        num_pid_m_per_copy_chunk = M_PER_CHUNK // BLOCK_SIZE_M
        chunk_offset = pid_m // (num_pid_m_per_copy_chunk * world_size)
        rank_offset = pid_m % (num_pid_m_per_copy_chunk * world_size) // num_pid_m_per_copy_chunk
        block_offset = pid_m % num_pid_m_per_copy_chunk

        rank_offset = (rank_offset + rank) % world_size
        pid_m = (rank_offset * M_per_rank + chunk_offset * M_PER_CHUNK + block_offset * BLOCK_SIZE_M) // BLOCK_SIZE_M

        offs_am = pid_m * BLOCK_SIZE_M
        offs_sig = offs_am // M_PER_CHUNK
        offs_rank = pid_m // pid_m_per_rank

        if offs_rank != rank:
            # token = dl.wait(barrier_ptr + offs_sig, 1, "sys", "acquire", waitValue=1)
            # A = dl.consume_token(A, token)
            wait_eq_sys(barrier_ptr + offs_sig, 1)
            # wait_cas_sys(barrier_ptr + offs_sig, one, 0)

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rk = tl.arange(0, BLOCK_SIZE_K)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        if offs_rank == rank:
            rm = rm % M_per_rank
            A_BASE = localA + rm[:, None] * stride_am + rk[None, :] * stride_ak
        else:
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

        c = acc.to(C.type.element_ty)
        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        c_mask = (rm[:, None] < M) & (rn[None, :] < N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        tl.store(C_, c, c_mask)


def triton_ag_gemm(
    a, b, transe_b,
    workspace_bufs,
    barrier_bufs,
    sync_bufs_ptr,
    one,
    rank, num_ranks, M, N, K,
    num_sms,
    current_stream,
    ag_streams,
    m_per_chunk,
):
    m_per_rank, K = a.shape
    n_per_rank = b.shape[0] if not transe_b else b.shape[1]

    for ag_stream in ag_streams:
        ag_stream.wait_stream(current_stream)

    output = torch.empty([WORLD_SIZE * m_per_rank, n_per_rank], dtype=a.dtype, device=a.device)

    def call_ag():
        cp_engine_producer_all_gather_full_mesh_push_multi_stream(
            rank, num_ranks, a, workspace_bufs, one,
            m_per_chunk, ag_streams, barrier_bufs)

    call_ag()

    # serial mode (for test)
    # for ag_stream in ag_streams:
    #     current_stream.wait_stream(ag_stream)
    # torch.cuda.synchronize()
    # torch.distributed.barrier()
    
    num_xcds = 1
    with torch.cuda.stream(current_stream):
        grid = lambda META: (min(
                num_sms,
                triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(n_per_rank, META["BLOCK_SIZE_N"]),
            ), )
        kernel_consumer_gemm_persistent[grid](
                workspace_bufs[rank][:M],
                a,
                b,
                output,  #
                M,
                n_per_rank,
                K,  #
                a.stride(0),
                a.stride(1),  #
                b.stride(1),  # layout of b is ((N, K), (K, 1)), stride_bk = b.stride(1)
                b.stride(0),  # stride_bn = b.stride(0)
                output.stride(0),
                output.stride(1),  #
                one,
                rank,
                num_ranks,
                barrier_bufs[rank],
                M_PER_CHUNK=m_per_chunk,
                NUM_SMS=num_sms,  #
                NUM_XCDS=num_xcds,
            )
    
    # ranks sync and reset barrier buffer
    # torch.cuda.current_stream().synchronize()
    barrier_all_on_stream(rank, num_ranks, sync_bufs_ptr, current_stream)
    barrier_bufs[rank].fill_(0)
    torch.cuda.current_stream().synchronize()

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

configs = []
configs.append(
triton.testing.Benchmark(
    x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
    x_vals=[
        # GEMM RS
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
            local_M = M // TP_GROUP.size()
            local_N = N // TP_GROUP.size()
            quantiles = [0.5, 0.2, 0.8]
            scale = TP_GROUP.rank() + 10
            input, weight, bias = _make_data(local_M, local_N, K, scale, dtype, device)

            M_PER_CHUNK = M // WORLD_SIZE
            m_chunk_num = (M + M_PER_CHUNK - 1) // M_PER_CHUNK

            workspaces = pyrocshmem.rocshmem_create_tensor_list_intra_node((M, K), dtype=dtype)
            barrier_bufs = pyrocshmem.rocshmem_create_tensor_list_intra_node((m_chunk_num,), dtype=torch.int)
            barrier_bufs[RANK].fill_(0)
            
            # sync bufs
            sync_bufs = pyrocshmem.rocshmem_create_tensor_list_intra_node((WORLD_SIZE,), dtype=torch.int)
            sync_bufs[RANK].fill_(0)
            sync_bufs_ptr = torch.tensor([t.data_ptr() for t in sync_bufs], device=torch.cuda.current_device(),
                                            requires_grad=False)

            current_stream = torch.cuda.current_stream()
            ag_streams = [torch.cuda.Stream(priority=-1) for i in range(WORLD_SIZE)]

            one = torch.ones((1024, ), dtype=torch.int32, device=torch.cuda.current_device())

            torch.cuda.synchronize()
            torch.distributed.barrier()
            torch.cuda.empty_cache()

            if provider == "torch":
                ms, min_ms, max_ms = \
                    triton.testing.do_bench_multiprocess(
                        lambda: torch_ag_gemm(
                            input, weight, transed_weight=False, bias=bias, TP_GROUP=TP_GROUP), quantiles=quantiles)
                self.results_torch[0] = {'ms': ms}

            if provider == 'triton':
                ms, min_ms, max_ms = triton.testing.do_bench_multiprocess(
                    lambda: triton_ag_gemm(
                        input, weight, False,
                        workspaces,
                        barrier_bufs,
                        sync_bufs_ptr,
                        one,
                        RANK, WORLD_SIZE, M, N, K,
                        num_sms,
                        current_stream,
                        ag_streams, M_PER_CHUNK), quantiles=quantiles)
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
    # M, N, K = 1024, 4096, 12288
    M, N, K = 2048, 3584, 14336
    
    local_M = M // WORLD_SIZE
    local_N = N // WORLD_SIZE
    
    scale = 10
    dtype = torch.float16
    a, b, bias = _make_data(local_M, local_N, K, scale, dtype, device)
    assert bias is None

    # check data
    atol, rtol = THRESHOLD_MAP[dtype], THRESHOLD_MAP[dtype]

    m_per_rank = M // WORLD_SIZE
    M_PER_CHUNK = M // WORLD_SIZE
    m_chunk_num = (M + M_PER_CHUNK - 1) // M_PER_CHUNK
    
    workspaces = pyrocshmem.rocshmem_create_tensor_list_intra_node((M, K), dtype=dtype)

    barrier_bufs = pyrocshmem.rocshmem_create_tensor_list_intra_node((m_chunk_num,), dtype=torch.int)
    barrier_bufs[RANK].fill_(0)
    
    # sync bufs
    sync_bufs = pyrocshmem.rocshmem_create_tensor_list_intra_node((WORLD_SIZE,), dtype=torch.int)
    sync_bufs[RANK].fill_(0)
    sync_bufs_ptr = torch.tensor([t.data_ptr() for t in sync_bufs], device=torch.cuda.current_device(),
                                    requires_grad=False)
    
    # to be removed
    one = torch.ones((1024, ), dtype=torch.int32, device=torch.cuda.current_device())

    current_stream = torch.cuda.current_stream()
    scatter_stream = torch.cuda.Stream(priority=-1)
    
    ag_streams = [torch.cuda.Stream(priority=-1) for i in range(WORLD_SIZE)]

    torch.cuda.synchronize()
    torch.distributed.barrier()
    
    if args.check:
        torch.cuda.empty_cache()
        input_list = [_make_data(local_M, local_N, K, scale, dtype, device) for _ in range(args.iters)]
        dist_out_list, torch_out_list = [], []

        # torch impl
        for input, weight, bias in input_list:
            torch_out = torch_ag_gemm(
                input, weight, transed_weight=False, bias=bias, TP_GROUP=TP_GROUP)
            torch_out_list.append(torch_out)

        # barrier_all_on_stream(RANK, WORLD_SIZE, sync_bufs_ptr, current_stream)
        pyrocshmem.rocshmem_barrier_all()
        torch.cuda.synchronize()
        torch.distributed.barrier()

        # dist triton impl
        for idx, (input, weight, bias) in enumerate(input_list):
            dist_out = triton_ag_gemm(
                    input, weight, False,
                    workspaces,
                    barrier_bufs,
                    sync_bufs_ptr,
                    one,
                    RANK, WORLD_SIZE, M, N, K,
                    num_sms,
                    current_stream,
                    ag_streams,
                    M_PER_CHUNK
                )

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
    
    ctx = get_torch_prof_ctx(False)
    with ctx:
        # torch.cuda.empty_cache()
        torch_out, torch_perf = perf_func(
            partial(
                torch_ag_gemm, a, b, False, bias, TP_GROUP),
                iters = perf_iters,
                warmup_iters = perf_warmups)

        pyrocshmem.rocshmem_barrier_all()
        torch.cuda.synchronize()
        torch.distributed.barrier()
        triton_out, triton_perf = perf_func(
            partial(triton_ag_gemm,
                        a, b, False,
                        workspaces,
                        barrier_bufs,
                        sync_bufs_ptr,
                        one,
                        RANK, WORLD_SIZE, M, N, K,
                        num_sms,
                        current_stream,
                        ag_streams, M_PER_CHUNK),
            iters = perf_iters,
            warmup_iters = perf_warmups)
    
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

    # best_config = kernel_consumer_gemm_persistent.best_config
    # print("Best Config:", best_config)