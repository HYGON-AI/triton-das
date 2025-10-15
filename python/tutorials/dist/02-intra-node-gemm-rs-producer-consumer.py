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

from utils import NVSHMEM_SIGNAL_DTYPE

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

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        # target_m = ((pid_m * BLOCK_SIZE_M % M_per_rank) + M_per_rank * rank)
        # offs_cm = target_m + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        # # Modif: get scatter buffer ptr
        dtype = a_ptr.dtype.element_ty
        
        # c_ptr = tl.load(scatter_buf_ptr + rank_offset).to(tl.pointer_type(dtype))
        # c_ptr = libshmem_device.remote_ptr(scatter_buf_ptr, rank_offset).to(tl.pointer_type(dtype))
        
        # store to local scatter buffer
        c_ptr = scatter_buf_ptr
        
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        c = accumulator.to(dtype)

        tl.store(c_ptrs, c, mask=c_mask)

        counter_start = start_m // M_per_rank
        counter_end = (start_m + BLOCK_SIZE_M - 1) // M_per_rank
        counter_end = min(counter_end, num_ranks - 1)

        # print(f"pid_m: {pid_m} pid: {start_pid} start: {counter_start} end: {counter_end}")
        for counter_id in range(counter_start, counter_end + 1):
            m_start = M_per_rank * counter_id
            m_end = M_per_rank * (counter_id + 1) - 1
            tiled_m_start = m_start // BLOCK_SIZE_M
            tiled_m_end = m_end // BLOCK_SIZE_M
            tiled_m_size = tiled_m_end - tiled_m_start + 1
            tiled_n = tl.cdiv(N, BLOCK_SIZE_N)
            # `tiled_m_size * tiled_n` represents the total number of tiles within the rank.
            val = tl.atomic_add(counter_ptr + counter_id, 1, sem="release", scope="gpu")
            # If `val` is equal to `tiled_m_size * tiled_n - 1`, it means the current tile
            # is the last one to be completed for this rank.
            if val == tiled_m_size * tiled_n - 1:
                tl.store(barrier_ptr + counter_id, 1)


@triton.jit
def kernel_ring_reduce(
    c_ptr,  # [M, N]
    out_ptr,  # [M_per_rank, N]
    barrier_ptr,
    # shape of matrix
    M_per_rank,
    N,
    rank,
    num_ranks: tl.constexpr,
    # tile size
    BLOCK_SIZE: tl.constexpr,
    NUM_SMS: tl.constexpr,  #
):
    start_pid = tl.program_id(axis=0)
    
    num_tiles = tl.cdiv(M_per_rank * N, BLOCK_SIZE)
    
    for pid in tl.range(start_pid, num_tiles, NUM_SMS, flatten=False):
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
    tl.atomic_cas(ptr, 0, 1, sem="release", scope="gpu")

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
    # pynvshmem.nvshmemx_signal_wait_until_on_stream(
    #         ptr,
    #         libshmem_device.NVSHMEM_CMP_EQ,
    #         1,
    #         stream.cuda_stream)
    
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
    rank, num_ranks, M, N, K,
    num_sms,
    current_stream,
    scatter_stream
):
    scatter_stream.wait_stream(current_stream)
    counter_buf = torch.zeros((num_ranks, ), dtype=torch.int32, device=device)
    
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

    grid = lambda META: (min(
                num_sms_gemm,
                triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
            ), )
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

    '''
    # optim 1: memcpy to scatter and reduce
    # NOTE: copy_ affects gemm kernel perf since it uses CUs (blit kernel)
    #       so here we impl copyp2p_kernel to limit copy CUs
    with torch.cuda.stream(scatter_stream):
        to_rank = (rank - 1 + num_ranks) % num_ranks
        for stage in range(num_ranks):
            segment = (rank + stage + 1) % num_ranks
            m_start = segment * m_per_rank
            m_end = m_start + m_per_rank
            src = gemm_bufs[rank][m_start:m_end]
            dst = scatter_bufs[to_rank][m_start:m_end]
            # wait gemm buf ready (local buf)
            _wait_eq_cuda(barrier_bufs[rank][segment].data_ptr(), scatter_stream)
            if stage != 0:
                # wait scatter buf ready (local buf)
                _wait_eq_cuda(scatter_signal_bufs[rank][segment].data_ptr(), scatter_stream)
                buffer = scatter_bufs[rank][m_start:m_end]
                cur_out = output if output is not None and stage == num_ranks - 1 else buffer
                # directly reduce to output
                add_continuous(src, buffer, cur_out, scatter_stream, num_ctas=num_sms)
            if stage == num_ranks - 1:
                break
            # push remote scatter buffer
            if stage == 0:
                # dst.copy_(src)
                # NOTE: 
                # num_warps: 32, warp_size: 64 n_max_threads: 1024
                copyp2p_kernel[(num_ctas_rs, )](
                    dst, src, num_elems_per_rank,
                    num_warps=num_warps_rs,
                    BLOCK_SIZE=num_warps_rs * 64 * 16) # per thread loads 128-bit
            else:
                # dst.copy_(buffer)
                copyp2p_kernel[(num_ctas_rs, )](
                    dst, buffer, num_elems_per_rank,
                    num_warps=num_warps_rs,
                    BLOCK_SIZE=num_warps_rs * 64 * 16) # per thread loads 128-bit
            # set scatter buf ready (symm buf)
            pynvshmem.nvshmemx_signal_op_on_stream(
                scatter_signal_bufs[rank][segment].data_ptr(),
                1,
                libshmem_device.NVSHMEM_SIGNAL_SET,
                to_rank,
                scatter_stream.cuda_stream)
        # barrier (sync all PEs data)
        # pynvshmem.nvshmem_barrier_all()
    '''

    # '''
    # optim 2: with copyp2p kernel (require stores to remote)
    # NOTE: gemm kernel not affected (because we limit CU for copyp2p_kernel)
    with torch.cuda.stream(scatter_stream):
        to_rank = (rank - 1 + num_ranks) % num_ranks
        for stage in range(num_ranks):
            segment = (rank + stage + 1) % num_ranks
            m_start = segment * m_per_rank
            m_end = m_start + m_per_rank
            src = gemm_bufs[rank][m_start:m_end]
            dst = scatter_bufs[to_rank][m_start:m_end]
            # wait gemm buf ready (local buf)
            # _wait_eq_cuda(barrier_bufs[rank][segment].data_ptr(), scatter_stream)
            # print(f"RANK {rank}:{segment} wait barrier_bufs: {barrier_bufs[rank][segment]}")
            _wait_eq_cuda(barrier_bufs[rank][segment], scatter_stream)
            # print(f"RANK {rank}:{segment} wait barrier_bufs done")
            cur_out = output if output is not None and stage == num_ranks - 1 else dst
            # P2P_KERNEL_BLOCK_SIZE = 1024
            if stage == 0:
                # only copy (gemm -> scatter)
                copyp2p_kernel[(num_ctas_rs, )](
                    cur_out, src, num_elems_per_rank,
                    num_warps=num_warps_rs,
                    BLOCK_SIZE=num_warps_rs * 64 * 16)
            else:
                # _wait_eq_cuda(scatter_signal_bufs[rank][segment].data_ptr(), scatter_stream)
                # print(f"RANK {rank}:{segment} wait scatter_signal_bufs: {scatter_signal_bufs[rank][segment]}")
                _wait_eq_cuda(scatter_signal_bufs[rank][segment], scatter_stream)
                # print(f"RANK {rank}:{segment} wait scatter_signal_bufs done")
                # add and write to remote ranks
                add_continuous_kernel[(num_ctas_rs, )](
                    src, scatter_bufs[rank][m_start:m_end],
                    cur_out, num_elems_per_rank,
                    num_warps=num_warps_rs,
                    BLOCK_SIZE=num_warps_rs * 64 * 16)
            if stage == num_ranks - 1:
                break
            # set signal
            kernel_set_signal[(1, )](
                scatter_signal_bufs[to_rank][segment],
                1,
            )
            # pynvshmem.nvshmemx_signal_op_on_stream(
            #     scatter_signal_bufs[rank][segment].data_ptr(),
            #     1,
            #     libshmem_device.NVSHMEM_SIGNAL_SET,
            #     to_rank,
            #     scatter_stream.cuda_stream)

    # barrier (sync all PEs data)
    current_stream.wait_stream(scatter_stream)
    barrier_all_on_stream(rank, num_ranks, sync_bufs_ptr, current_stream)
    # '''

    '''
    # optim 3: direct reduce on gemm buffer (no copy but requires load remote buffer)
    # NOTE: load from peer seems slower than store to peer
    with torch.cuda.stream(scatter_stream):
        # to_rank = (rank - 1 + num_ranks) % num_ranks
        from_rank = (rank + 1 + num_ranks) % num_ranks
        for stage in range(num_ranks):
            segment = (rank + stage + 1) % num_ranks
            m_start = segment * m_per_rank
            m_end = m_start + m_per_rank
            local_buf = gemm_bufs[rank][m_start:m_end]
            remote_buf = gemm_bufs[from_rank][m_start:m_end]
            # wait local gemm buf ready
            _wait_eq_cuda(barrier_bufs[rank][segment].data_ptr(), scatter_stream)
            cur_out = output if stage == num_ranks - 1 else local_buf
            if stage == 0:
                pass
            else:
                remote_barrier = scatter_signal_bufs[from_rank][segment]
                _wait_eq_cuda(remote_barrier.data_ptr(), scatter_stream)
                # local[seg] += remote[seg]
                add_continuous_kernel[(num_ctas_rs, )](
                    local_buf, remote_buf,
                    cur_out, num_elems_per_rank,
                    num_warps=num_warps_rs,
                    BLOCK_SIZE=num_warps_rs * 64 * 16)
            if stage == num_ranks - 1:
                break
            # set signal
            pynvshmem.nvshmemx_signal_op_on_stream(
                scatter_signal_bufs[rank][segment].data_ptr(),
                1,
                libshmem_device.NVSHMEM_SIGNAL_SET,
                rank,
                scatter_stream.cuda_stream)
    '''

    # reset signal bufs
    # torch.cuda.synchronize()

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

            gemm_bufs = pynvshmem.nvshmem_create_tensor_list_intra_node((M, N), dtype=dtype)
            # barrier_bufs = pynvshmem.nvshmem_create_tensor_list_intra_node((m_chunk_num,), dtype=NVSHMEM_SIGNAL_DTYPE)
            barrier_bufs = pynvshmem.nvshmem_create_tensor_list_intra_node((m_chunk_num,), dtype=torch.int)
            barrier_bufs[RANK].fill_(0)

            # scatter bufs
            scatter_bufs = pynvshmem.nvshmem_create_tensor_list_intra_node((M, N), dtype=dtype)
            # scatter_signal_bufs = pynvshmem.nvshmem_create_tensor_list_intra_node((WORLD_SIZE,), dtype=NVSHMEM_SIGNAL_DTYPE)
            scatter_signal_bufs = pynvshmem.nvshmem_create_tensor_list_intra_node((WORLD_SIZE,), dtype=torch.int)
            scatter_signal_bufs[RANK].fill_(0)

            # sync bufs
            sync_bufs = pynvshmem.nvshmem_create_tensor_list_intra_node((WORLD_SIZE,), dtype=torch.int)
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
                            input, weight, bias, TP_GROUP, transpose_weight=False, all_reduce=False), quantiles=quantiles)
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
    pynvshmem.init_nvshmem_by_uniqueid(TP_GROUP)

    # test_nvshmem_signal(RANK)
    # torch.distributed.destroy_process_group()
    # print("exit")
    # exit(0)

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
    
    gemm_bufs = pynvshmem.nvshmem_create_tensor_list_intra_node((M, N), dtype=dtype)
    # barrier_bufs = pynvshmem.nvshmem_create_tensor_list_intra_node((m_chunk_num,), dtype=NVSHMEM_SIGNAL_DTYPE)
    barrier_bufs = pynvshmem.nvshmem_create_tensor_list_intra_node((m_chunk_num,), dtype=torch.int)
    barrier_bufs[RANK].fill_(0)
    
    # scatter bufs
    scatter_bufs = pynvshmem.nvshmem_create_tensor_list_intra_node((M, N), dtype=dtype)
    # scatter_signal_bufs = pynvshmem.nvshmem_create_tensor_list_intra_node((WORLD_SIZE,), dtype=NVSHMEM_SIGNAL_DTYPE)
    scatter_signal_bufs = pynvshmem.nvshmem_create_tensor_list_intra_node((WORLD_SIZE,), dtype=torch.int)
    scatter_signal_bufs[RANK].fill_(0)
    
    # sync bufs
    sync_bufs = pynvshmem.nvshmem_create_tensor_list_intra_node((WORLD_SIZE,), dtype=torch.int)
    sync_bufs[RANK].fill_(0)
    sync_bufs_ptr = torch.tensor([t.data_ptr() for t in sync_bufs], device=torch.cuda.current_device(),
                                    requires_grad=False)

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
    
    ctx = get_torch_prof_ctx(False)
    with ctx:
        # torch.cuda.empty_cache()
        torch_out, torch_perf = perf_func(
            partial(
                torch_gemm_rs, a, b, bias, TP_GROUP, transpose_weight=False, all_reduce=False),
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
                        RANK, WORLD_SIZE, M, N, local_K,
                        num_sms,
                        current_stream,
                        scatter_stream),
            iters = perf_iters,
            warmup_iters = perf_warmups)
    
    try:
        torch.testing.assert_close(triton_out, torch_out, atol=atol, rtol=rtol)
    except Exception as e:
        # print(f"rank {RANK} torch: {torch_out}\n")
        # print(f"rank {RANK} triton: {triton_out}\n")
        raise e

    dist_print(f"triton #{RANK}", triton_perf, need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))
    dist_print(f"torch #{RANK}", torch_perf, need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))

    torch.distributed.destroy_process_group()

    # best_config = kernel_gemm_rs_producer_persistent.best_config
    # print("Best Config:", best_config)