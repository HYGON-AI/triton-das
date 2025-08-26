import pytest
import torch

import time
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.ops
from triton.runtime.driver import driver
import math
# from triton.testing import get_dram_gbps, get_max_tensorcore_tflops, nvsmi

global abort_if_check_failed
abort_if_check_failed = True

#######################
# Utilities
#######################

ARCH_NAME = driver.active.get_current_target().arch

def check_perf(cur_ms, cur_perf, ref_perf, atol=None, args='', err_msg=''):
    if abort_if_check_failed and cur_perf < ref_perf - atol:
        raise AssertionError(f'{err_msg} {cur_perf} is not greater than {ref_perf} (atol={atol})')
    print(args)
    print(f'{cur_ms:.3f} ms \t cur: {cur_perf:.3f} \t ref: {ref_perf:.3f} \t dif={cur_perf - ref_perf:.3f}')

def compare_tensors(tensor1, tensor2):
    # 检查两个张量的形状是否相同
    if tensor1.shape != tensor2.shape:
        raise ValueError("The shapes of the two tensors do not match.")

    # 计算误差
    error = tensor1 - tensor2

    # 计算平均误差
    mean_error = torch.mean(torch.abs(error))

    return mean_error


#######################
# Matrix Multiplication
#######################

matmul_data = {
    'gfx928': {
        # square
        (512, 512, 512): {'float16': 7.820, 'float32': 2.799},
        (1024, 1024, 1024): {'float16': 11.980, 'float32': 4.105},
    }
}

@pytest.mark.parametrize('M, N, K, dtype_str', [(M, N, K, dtype_str)
                                                for M, N, K in matmul_data[ARCH_NAME].keys()
                                                for dtype_str in ['float16', 'float32']])
def test_matmul(M, N, K, dtype_str):
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    dtype = {'float16': torch.float16, 'float32': torch.float32, 'int8': torch.int8}[dtype_str]
    torch.manual_seed(0)
    ref_gpu_perf = matmul_data[ARCH_NAME][(M, N, K)][dtype_str]
    if dtype == torch.int8:
        a = torch.randint(-128, 127, (M, K), dtype=dtype, device='cuda')
        b = torch.randint(-128, 127, (N, K), dtype=dtype, device='cuda')
        b = b.t()  # only test row-col layout
    else:
        a = torch.randn((M, K), dtype=dtype, device='cuda')
        b = torch.randn((K, N), dtype=dtype, device='cuda')
    fn = lambda: triton.ops.matmul(a, b)
    ms = triton.testing.do_bench_cudagraph(fn)
    cur_gpu_perf = 2. * M * N * K / ms * 1e-9
    check_perf(ms, cur_gpu_perf, ref_gpu_perf, atol=2,
               args=f"\nArguments: ({M}, {N}, {K}, {dtype_str})")


#######################
# Element-Wise
#######################

@triton.jit
def _add(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

elementwise_data = {
    'gfx928': {
        1024 * 16: {'float16': 25.826, 'float32': 47.033},
        1024 * 64: {'float16': 98.007, 'float32': 174.487},
        1024 * 256: {'float16': 282.918, 'float32': 561.145},
        1024 * 1024: {'float16': 609.696, 'float32': 193.790},
        1024 * 16384: {'float16': 642.313, 'float32': 627.150},
        1024 * 65536: {'float16': 661.674, 'float32': 635.334},
        # Non pow 2
        1020 * 100: {'float16': 138.052, 'float32': 261.483},
        10003 * 7007: {'float16': 468.616, 'float32': 448.157},
    }
}

@pytest.mark.parametrize('N', elementwise_data[ARCH_NAME].keys())
@pytest.mark.parametrize("dtype_str", ['float16', 'bfloat16', 'float32'])
def test_elementwise(N, dtype_str):
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    torch.manual_seed(0)
    dtype = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}[dtype_str]
    ref_dtype_str = 'float16' if dtype_str == 'bfloat16' else dtype_str
    ref_gpu_perf = elementwise_data[ARCH_NAME][N][ref_dtype_str]
    z = torch.empty((N, ), dtype=dtype, device='cuda')
    x = torch.randn_like(z)
    y = torch.randn_like(z)
    grid = lambda args: (triton.cdiv(N, args['BLOCK_SIZE']), )
    fn = lambda: _add[grid](x, y, z, N, BLOCK_SIZE=1024)
    ms = triton.testing.do_bench_cudagraph(fn)
    cur_gpu_perf = 3. * N * z.element_size() / ms * 1e-6
    check_perf(ms, cur_gpu_perf, ref_gpu_perf, atol=20,
               args=f"\nArguments: ({N}, {dtype_str})")


#######################
# Flash-Attention
#######################

flash_attention_data = {
    'gfx928': {
        (4, 48, 4096, 64, True, True, 'forward', 'float16'): 485.774,
        # (4, 48, 4096, 64, True, True, 'forward', 'bfloat16'): 0.471,
        # (4, 48, 1024, 16, True, True, 'forward', 'float32'): 0.155,
        # (4, 48, 4096, 64, True, True, 'backward', 'float16'): 0.232,
        # (4, 48, 4096, 64, True, True, 'backward', 'bfloat16'): 0.231,
        # (4, 48, 1024, 16, True, True, 'backward', 'float32'): 0.138,
        # (4, 48, 4096, 64, True, False, 'forward', 'float16'): 0.306,
        # (4, 48, 4096, 64, True, False, 'forward', 'bfloat16'): 0.266,
        # (4, 48, 1024, 16, True, False, 'forward', 'float32'): 0.098,
        # (4, 48, 4096, 64, True, False, 'backward', 'float16'): 0.134,
        # (4, 48, 4096, 64, True, False, 'backward', 'bfloat16'): 0.135,
        # (4, 48, 1024, 16, True, False, 'backward', 'float32'): 0.092,
        # (4, 48, 4096, 64, False, True, 'forward', 'float16'): 0.541,
        # (4, 48, 4096, 64, False, True, 'forward', 'bfloat16'): 0.471,
        # (4, 48, 1024, 16, False, True, 'forward', 'float32'): 0.150,
        # (4, 48, 4096, 64, False, True, 'backward', 'float16'): 0.291,
        # (4, 48, 4096, 64, False, True, 'backward', 'bfloat16'): 0.255,
        # (4, 48, 1024, 16, False, True, 'backward', 'float32'): 0.144,
        # (4, 48, 4096, 64, False, False, 'forward', 'float16'): 0.306,
        # (4, 48, 4096, 64, False, False, 'forward', 'bfloat16'): 0.266,
        # (4, 48, 1024, 16, False, False, 'forward', 'float32'): 0.098,
        # (4, 48, 4096, 64, False, False, 'backward', 'float16'): 0.159,
        # (4, 48, 4096, 64, False, False, 'backward', 'bfloat16'): 0.159,
        # (4, 48, 1024, 16, False, False, 'backward', 'float32'): 0.088,
    }
}

@pytest.mark.parametrize("Z, H, N_CTX, D_HEAD, seq_par, causal, mode, dtype_str",
                         [(Z, H, N_CTX, D_HEAD, seq_par, causal, mode, dtype_str)
                          for Z, H, N_CTX, D_HEAD, seq_par, causal, mode, dtype_str
                          in flash_attention_data[ARCH_NAME].keys()])
def test_flash_attention(Z, H, N_CTX, D_HEAD, seq_par, causal, mode, dtype_str):
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    is_backward = mode == 'backward'
    torch.manual_seed(20)
    dtype = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}[dtype_str]
    q = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2).requires_grad_()
    k = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.4, std=0.2).requires_grad_()
    v = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2).requires_grad_()
    sm_scale = D_HEAD ** -0.5
    # benchmark
    fn = lambda: triton.ops.attention(q, k, v, causal, sm_scale, seq_par)
    if is_backward:
        o = fn()
        do = torch.randn_like(o)
        fn = lambda: o.backward(do, retain_graph=True)
    ms = triton.testing.do_bench_cudagraph(fn)
    flops_per_matmul = 2.0 * Z * H * N_CTX * N_CTX * D_HEAD
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if is_backward:
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    cur_gpu_perf = total_flops / ms * 1e-9
    ref_gpu_perf = flash_attention_data[ARCH_NAME][(Z, H, N_CTX, D_HEAD, seq_par, causal, mode, dtype_str)]
    check_perf(ms, cur_gpu_perf, ref_gpu_perf, atol=20,
               args=f"\nArguments: ({Z}, {H}, {N_CTX}, {D_HEAD}, {seq_par}, {causal}, {mode}, {dtype_str})")


#######################
# Reduction
#######################

@triton.jit
def _sum(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    # run in a loop to only to make it compute bound.
    for i in range(100):
        x = tl.sum(x, axis=0) + y

    tl.store(output_ptr + offsets, x, mask=mask)

reduction_data = {
    'gfx928': {
        1024 * 16384: {'float16': 2.154, 'float32': 2.178, 'int32': 2.166},
        1024 * 65536: {'float16': 2.201, 'float32': 2.230, 'int32': 2.225},
    }
}

@pytest.mark.parametrize('N', reduction_data[ARCH_NAME].keys())
@pytest.mark.parametrize("dtype_str", ['float16', 'float32', 'int32'])
def test_reductions(N, dtype_str):
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    torch.manual_seed(0)
    dtype = {'float16': torch.float16, 'float32': torch.float32, 'int32': torch.int32}[dtype_str]
    ref_gpu_perf = reduction_data[ARCH_NAME][N][dtype_str]
    z = torch.empty((N, ), dtype=dtype, device='cuda')
    if dtype == torch.float16 or dtype == torch.float32:
        x = torch.randn_like(z)
        y = torch.randn_like(z)
    else:
        info = torch.iinfo(dtype)
        x = torch.randint(info.min, info.max, (N, ), dtype=dtype, device='cuda')
        y = torch.randint(info.min, info.max, (N, ), dtype=dtype, device='cuda')
    grid = lambda args: (triton.cdiv(N, args['BLOCK_SIZE']), )
    fn = lambda: _sum[grid](x, y, z, N, BLOCK_SIZE=1024)
    ms = triton.testing.do_bench_cudagraph(fn)
    cur_gpu_perf = 100. * 2. * N / ms * 1e-9
    check_perf(ms, cur_gpu_perf, ref_gpu_perf, atol=1,
               args=f"\nArguments: ({N}, {dtype_str})")


#######################
# MHA
#######################

# def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
#                     return_attn_probs=False):
#     # k, v = (padding_bshd(t) for t in (k, v))
#     # q, k, v = (t.transpose(1, 2) for t in (q, k, v))
#     # if softmax_scale is None:
#     #     D_HEAD = q.shape[-1]
#     #     softmax_scale = D_HEAD**-0.5
#     # input_metadata = MetaData(sm_scale=softmax_scale, causal=causal, dropout_p=dropout_p, return_encoded_softmax=return_attn_probs)
#     # input_metadata.max_seqlens_q = q.shape[2]
#     # input_metadata.max_seqlens_k = k.shape[2]
#     return FlashAttnFunc.apply(q, k, v, None, input_metadata).transpose(1, 2)

@triton.autotune(
   configs=[
       triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16}, num_stages=1, num_warps=1),
       triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16}, num_stages=1, num_warps=1),
       triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=1, num_warps=1),
       triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=1, num_warps=2),
   ],
   key=['CACHE_KEY_SEQLEN_Q', 'CACHE_KEY_SEQLEN_K', 'BIAS_TYPE', 'IS_CAUSAL', 'BLOCK_HEADDIM']
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    Bias,
    Out,
    Lse,
    TMP,  # NOTE: TMP is a scratchpad buffer to workaround a compiler bug
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_bb,
    stride_bh,
    stride_bm,
    stride_ob,
    stride_oh,
    stride_om,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # off_b = tl.program_id(1)
    # off_h = tl.program_id(2)
    # off_hb = off_b * nheads + off_h
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    # Initialize pointers to Q, K, V
    # Adding parenthesis around indexing might use int32 math instead of int64 math?
    # https://github.com/openai/triton/issues/741
    # I'm seeing a tiny bit of difference (5-7us)
    q_ptrs = (
        Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    )
    k_ptrs = (
        K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :])
    )
    v_ptrs = (
        V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])
    )
    if BIAS_TYPE == "vector":
        b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + offs_n
    elif BIAS_TYPE == "matrix":
        b_ptrs = (
            Bias
            + off_b * stride_bb
            + off_h * stride_bh
            + (offs_m[:, None] * stride_bm + offs_n[None, :])
        )
    # initialize pointer to m and l
    t_ptrs = TMP + off_hb * seqlen_q_rounded + offs_m
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    # [2022-10-30] TD: Triton bug - in the case of EVEN_M=True and EVEN_N=False, if we just call
    # tl.load(q_ptrs), we get the wrong output!
    if EVEN_M & EVEN_N:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
        else:
            q = tl.load(
                q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0
            )
    # loop over k, v and update accumulator
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k)) # , trans_b=True
        # Trying to combine the two masks seem to make the result wrong
        if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))
        if IS_CAUSAL:
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf"))
        if BIAS_TYPE != "none":
            if BIAS_TYPE == "vector":
                if EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs + start_n, mask=(start_n + offs_n) < seqlen_k, other=0.0
                    ).to(tl.float32)
                bias = bias[None, :]
            elif BIAS_TYPE == "matrix":
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs + start_n,
                        mask=(offs_m[:, None] < seqlen_q)
                        & ((start_n + offs_n)[None, :] < seqlen_k),
                        other=0.0,
                    ).to(tl.float32)
            # Slightly faster to multiply the softmax_scale in the tl.exp below since the compiler
            # can then fuse the mult and add into an fma instruction. But if we have bias we need to
            # to multiply with softmax_scale here.
            qk = qk * softmax_scale + bias
            m_ij = tl.maximum(tl.max(qk, 1), lse_i)
            p = tl.exp(qk - m_ij[:, None])
        else:
            m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
            p = tl.exp(qk * softmax_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        # scale acc_o
        acc_o_scale = tl.exp(m_i - m_ij)

        # # -- update output accumulator --
        # BUG: have to store and immediately load
        tl.store(t_ptrs, acc_o_scale)
        acc_o_scale = tl.load(t_ptrs)
        acc_o = acc_o * acc_o_scale[:, None]
        # update acc_o
        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_vn)
            else:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)

        # -- update statistics
        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)

    o_scale = tl.exp(m_i - lse_i)
    # BUG: have to store and immediately load
    tl.store(t_ptrs, o_scale)
    o_scale = tl.load(t_ptrs)
    acc_o = acc_o * o_scale[:, None]
    # rematerialize offsets to save registers
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back l and m
    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m
    tl.store(lse_ptrs, lse_i)
    # initialize pointers to output
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = (
        Out
        + off_b * stride_ob
        + off_h * stride_oh
        + (offs_m[:, None] * stride_om + offs_d[None, :])
    )
    if EVEN_M:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o)
        else:
            tl.store(out_ptrs, acc_o, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen_q)
        else:
            tl.store(
                out_ptrs, acc_o, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim)
            )


@triton.jit
def _bwd_preprocess_do_o_dot(
    Out,
    DO,
    Delta,
    stride_ob,
    stride_oh,
    stride_om,
    stride_dob,
    stride_doh,
    stride_dom,
    nheads,
    seqlen_q,
    seqlen_q_rounded,
    headdim,
    BLOCK_M: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    # load
    o = tl.load(
        Out + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :],
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
        other=0.0,
    ).to(tl.float32)
    do = tl.load(
        DO
        + off_b * stride_dob
        + off_h * stride_doh
        + offs_m[:, None] * stride_dom
        + offs_d[None, :],
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
        other=0.0,
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hb * seqlen_q_rounded + offs_m, delta)


@triton.jit
def _bwd_store_dk_dv(
    dk_ptrs,
    dv_ptrs,
    dk,
    dv,
    offs_n,
    offs_d,
    seqlen_k,
    headdim,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
):
    # [2022-11-01] TD: Same bug. In the case of EVEN_N=True and EVEN_M=False,
    # if we just call tl.store(dv_ptrs), there's a race condition
    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            tl.store(dv_ptrs, dv)
            tl.store(dk_ptrs, dk)
        else:
            tl.store(dv_ptrs, dv, mask=offs_d[None, :] < headdim)
            tl.store(dk_ptrs, dk, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(dv_ptrs, dv, mask=offs_n[:, None] < seqlen_k)
            tl.store(dk_ptrs, dk, mask=offs_n[:, None] < seqlen_k)
        else:
            tl.store(dv_ptrs, dv, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))
            tl.store(dk_ptrs, dk, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))


@triton.jit
def _bwd_kernel_one_col_block(
    start_n,
    Q,
    K,
    V,
    Bias,
    DO,
    DQ,
    DK,
    DV,
    LSE,
    D,
    softmax_scale,
    stride_qm,
    stride_kn,
    stride_vn,
    stride_bm,
    stride_dom,
    stride_dqm,
    stride_dkn,
    stride_dvn,
    seqlen_q,
    seqlen_k,
    headdim,
    ATOMIC_ADD: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # We need to make sure begin_m is a multiple of BLOCK_M (not BLOCK_N)
    begin_m = 0 if not IS_CAUSAL else ((start_n * BLOCK_N) // BLOCK_M) * BLOCK_M
    # initialize row/col offsets
    offs_qm = begin_m + tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    # initialize pointers to value-like data
    q_ptrs = Q + (offs_qm[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_d[None, :])
    do_ptrs = DO + (offs_qm[:, None] * stride_dom + offs_d[None, :])
    dq_ptrs = DQ + (offs_qm[:, None] * stride_dqm + offs_d[None, :])
    if BIAS_TYPE == "vector":
        b_ptrs = Bias + offs_n
    elif BIAS_TYPE == "matrix":
        b_ptrs = Bias + (offs_qm[:, None] * stride_bm + offs_n[None, :])
    # initialize dv and dk
    dv = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    # There seems to be some problem with Triton pipelining that makes results wrong for
    # headdim=64, seqlen=(113, 255), bias_type='matrix'. In this case the for loop
    # may have zero step, and pipelining with the bias matrix could screw it up.
    # So we just exit early.
    if begin_m >= seqlen_q:
        dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :])
        dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])
        _bwd_store_dk_dv(
            dk_ptrs,
            dv_ptrs,
            dk,
            dv,
            offs_n,
            offs_d,
            seqlen_k,
            headdim,
            EVEN_M=EVEN_M,
            EVEN_N=EVEN_N,
            EVEN_HEADDIM=EVEN_HEADDIM,
        )
        return
    # k and v stay in SRAM throughout
    # [2022-10-30] TD: Same bug as the fwd. In the case of EVEN_N=True and EVEN_M=False,
    # if we just call tl.load(k_ptrs), we get the wrong output!
    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
        else:
            k = tl.load(k_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
            v = tl.load(v_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
            v = tl.load(v_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
        else:
            k = tl.load(
                k_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0
            )
            v = tl.load(
                v_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0
            )
    # loop over rows
    num_block_m = tl.cdiv(seqlen_q, BLOCK_M)
    for start_m in range(begin_m, num_block_m * BLOCK_M, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m_curr = start_m + offs_m
        # load q, k, v, do on-chip
        # Same bug as below. Otherwise gives wrong result for headdim=40, seqlen=(128, 117)
        if EVEN_M & EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            if EVEN_HEADDIM:
                q = tl.load(q_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0)
            else:
                q = tl.load(
                    q_ptrs,
                    mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
        # recompute p = softmax(qk, dim=-1).T
        qk = tl.dot(q, tl.trans(k)) #, trans_b=True
        # Trying to combine the two masks seem to make the result wrong
        if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
            qk = tl.where(offs_n[None, :] < seqlen_k, qk, float("-inf"))
        if IS_CAUSAL:
            qk = tl.where(offs_m_curr[:, None] >= (offs_n[None, :]), qk, float("-inf"))
        if BIAS_TYPE != "none":
            tl.debug_barrier()  # Race condition otherwise
            if BIAS_TYPE == "vector":
                if EVEN_N:
                    bias = tl.load(b_ptrs).to(tl.float32)
                else:
                    bias = tl.load(b_ptrs, mask=offs_n < seqlen_k, other=0.0).to(tl.float32)
                bias = bias[None, :]
            elif BIAS_TYPE == "matrix":
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k),
                        other=0.0,
                    ).to(tl.float32)
            qk = qk * softmax_scale + bias
        # There seems to be a race condition when headdim=48/96, and dq, dk, dv are wrong.
        # Also wrong for headdim=64.
        if not (EVEN_M & EVEN_HEADDIM):
            tl.debug_barrier()
        lse_i = tl.load(LSE + offs_m_curr)
        if BIAS_TYPE == "none":
            p = tl.exp(qk * softmax_scale - lse_i[:, None])
        else:
            p = tl.exp(qk - lse_i[:, None])
        # compute dv
        # [2022-10-30] TD: A Triton bug: if EVEN_M=True and EVEN_HEADDIM=False, if we call
        # do = tl.load(do_ptrs, mask=offs_d[None, :] < headdim, other=0.0), we get wrong outputs
        # in the case of headdim=48/96, seqlen_q & seqlen_k >= 512. If headdim=40 or seqlen < 512,
        # the output is correct.
        if EVEN_M & EVEN_HEADDIM:
            do = tl.load(do_ptrs)
        else:
            # [2022-11-01] TD: Triton bug, there's a race condition if we just use m_mask and not d_mask.
            do = tl.load(
                do_ptrs,
                mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                other=0.0,
            )
        # if EVEN_M:
        #     if EVEN_HEADDIM:
        #         do = tl.load(do_ptrs)
        #     else:
        #         do = tl.load(do_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
        # else:
        #     if EVEN_HEADDIM:
        #         do = tl.load(do_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0)
        #     else:
        #         do = tl.load(do_ptrs, mask=(offs_m_curr[:, None] < seqlen_q)
        #                                    & (offs_d[None, :] < headdim), other=0.0)
        dv += tl.dot(tl.trans(p.to(do.dtype)), do) # , trans_a=True
        # compute dp = dot(v, do)
        # There seems to be a race condition when headdim=48/96, and dq, dk are wrong.
        # Also wrong for headdim=128, seqlen=(108, 256), and ATOMIC_ADD=True
        # Also wrong for headdim=64, seqlen=(1023, 1024), and ATOMIC_ADD=False
        if not (EVEN_M & EVEN_HEADDIM):
            tl.debug_barrier()
        dp = tl.dot(do, tl.trans(v))
        # There's a race condition for headdim=48
        if not EVEN_HEADDIM:
            tl.debug_barrier()
        # compute ds = p * (dp - delta[:, None])
        # Putting the subtraction after the dp matmul (instead of before) is slightly faster
        Di = tl.load(D + offs_m_curr)
        # Converting ds to q.dtype here reduces register pressure and makes it much faster
        # for BLOCK_HEADDIM=128
        ds = (p * (dp - Di[:, None]) * softmax_scale).to(q.dtype)
        # compute dk = dot(ds.T, q)
        dk += tl.dot(tl.trans(ds), q) #, trans_a=True
        # compute dq
        if not (
            EVEN_M & EVEN_HEADDIM
        ):  # Otherewise there's a race condition when BIAS_TYPE='matrix'
            tl.debug_barrier()
        if not ATOMIC_ADD:
            if EVEN_M & EVEN_HEADDIM:  # Race condition if we just do EVEN_M
                dq = tl.load(dq_ptrs, eviction_policy="evict_last")
                dq += tl.dot(ds, k)
                tl.store(dq_ptrs, dq, eviction_policy="evict_last")
            else:
                if EVEN_HEADDIM:
                    dq = tl.load(
                        dq_ptrs,
                        mask=offs_m_curr[:, None] < seqlen_q,
                        other=0.0,
                        eviction_policy="evict_last",
                    )
                    dq += tl.dot(ds, k)
                    tl.store(
                        dq_ptrs,
                        dq,
                        mask=offs_m_curr[:, None] < seqlen_q,
                        eviction_policy="evict_last",
                    )
                else:
                    dq = tl.load(
                        dq_ptrs,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                        other=0.0,
                        eviction_policy="evict_last",
                    )
                    dq += tl.dot(ds, k)
                    tl.store(
                        dq_ptrs,
                        dq,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                        eviction_policy="evict_last",
                    )
        else:  # If we're parallelizing across the seqlen_k dimension
            dq = tl.dot(ds, k)
            if EVEN_M & EVEN_HEADDIM:  # Race condition if we just do EVEN_M
                tl.atomic_add(dq_ptrs, dq)
            else:
                if EVEN_HEADDIM:
                    tl.atomic_add(dq_ptrs, dq, mask=offs_m_curr[:, None] < seqlen_q)
                else:
                    tl.atomic_add(
                        dq_ptrs,
                        dq,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                    )
        # increment pointers
        dq_ptrs += BLOCK_M * stride_dqm
        q_ptrs += BLOCK_M * stride_qm
        do_ptrs += BLOCK_M * stride_dom
        if BIAS_TYPE == "matrix":
            b_ptrs += BLOCK_M * stride_bm
    # write-back
    dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :])
    dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])
    _bwd_store_dk_dv(
        dk_ptrs,
        dv_ptrs,
        dk,
        dv,
        offs_n,
        offs_d,
        seqlen_k,
        headdim,
        EVEN_M=EVEN_M,
        EVEN_N=EVEN_N,
        EVEN_HEADDIM=EVEN_HEADDIM,
    )


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


# @triton.autotune(
#     configs=[
#         # triton.Config(
#         #     {"BLOCK_M": 128, "BLOCK_N": 128, "SEQUENCE_PARALLEL": False},
#         #     num_warps=8,
#         #     num_stages=1,
#         #     pre_hook=init_to_zero("DQ"),
#         # ),
#         # triton.Config(
#         #     {"BLOCK_M": 128, "BLOCK_N": 128, "SEQUENCE_PARALLEL": True},
#         #     num_warps=8,
#         #     num_stages=1,
#         #     pre_hook=init_to_zero("DQ"),
#         # ),
#         # Other configs seem to give wrong results when seqlen_q % 128 != 0, disabling them for now
#         # # Kernel is buggy (give wrong result) if we set BLOCK_m=128, BLOCK_n=64, num_warps=*4*
#         # triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False}, num_warps=8, num_stages=1, pre_hook=init_to_zero('DQ')),
#         # triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True}, num_warps=8, num_stages=1, pre_hook=init_to_zero('DQ')),
#         # triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False}, num_warps=4, num_stages=1, pre_hook=init_to_zero('DQ')),
#         # triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True}, num_warps=4, num_stages=1, pre_hook=init_to_zero('DQ')),
#     ],
#     key=["CACHE_KEY_SEQLEN_Q", "CACHE_KEY_SEQLEN_K", "BIAS_TYPE", "IS_CAUSAL", "BLOCK_HEADDIM"],
# )

@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _bwd_kernel(
    Q,
    K,
    V,
    Bias,
    DO,
    DQ,
    DK,
    DV,
    LSE,
    D,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_bb,
    stride_bh,
    stride_bm,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    stride_dkb,
    stride_dkh,
    stride_dkn,
    stride_dvb,
    stride_dvh,
    stride_dvn,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    SEQUENCE_PARALLEL: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # offset pointers for batch/head
    Q += off_b * stride_qb + off_h * stride_qh
    K += off_b * stride_kb + off_h * stride_kh
    V += off_b * stride_vb + off_h * stride_vh
    DO += off_b * stride_dob + off_h * stride_doh
    DQ += off_b * stride_dqb + off_h * stride_dqh
    DK += off_b * stride_dkb + off_h * stride_dkh
    DV += off_b * stride_dvb + off_h * stride_dvh
    if BIAS_TYPE != "none":
        Bias += off_b * stride_bb + off_h * stride_bh
    # pointer to row-wise quantities in value-like data
    D += off_hb * seqlen_q_rounded
    LSE += off_hb * seqlen_q_rounded
    if not SEQUENCE_PARALLEL:
        num_block_n = tl.cdiv(seqlen_k, BLOCK_N)
        for start_n in range(0, num_block_n):
            _bwd_kernel_one_col_block(
                start_n,
                Q,
                K,
                V,
                Bias,
                DO,
                DQ,
                DK,
                DV,
                LSE,
                D,
                softmax_scale,
                stride_qm,
                stride_kn,
                stride_vn,
                stride_bm,
                stride_dom,
                stride_dqm,
                stride_dkn,
                stride_dvn,
                seqlen_q,
                seqlen_k,
                headdim,
                ATOMIC_ADD=False,
                BIAS_TYPE=BIAS_TYPE,
                IS_CAUSAL=IS_CAUSAL,
                BLOCK_HEADDIM=BLOCK_HEADDIM,
                EVEN_M=EVEN_M,
                EVEN_N=EVEN_N,
                EVEN_HEADDIM=EVEN_HEADDIM,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )
    else:
        start_n = tl.program_id(0)
        _bwd_kernel_one_col_block(
            start_n,
            Q,
            K,
            V,
            Bias,
            DO,
            DQ,
            DK,
            DV,
            LSE,
            D,
            softmax_scale,
            stride_qm,
            stride_kn,
            stride_vn,
            stride_bm,
            stride_dom,
            stride_dqm,
            stride_dkn,
            stride_dvn,
            seqlen_q,
            seqlen_k,
            headdim,
            ATOMIC_ADD=True,
            BIAS_TYPE=BIAS_TYPE,
            IS_CAUSAL=IS_CAUSAL,
            BLOCK_HEADDIM=BLOCK_HEADDIM,
            EVEN_M=EVEN_M,
            EVEN_N=EVEN_N,
            EVEN_HEADDIM=EVEN_HEADDIM,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )


def _flash_attn_forward(q, k, v, bias=None, causal=False, softmax_scale=None):
    # shape constraints
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, _ = k.shape
    assert k.shape == (batch, seqlen_k, nheads, d)
    assert v.shape == (batch, seqlen_k, nheads, d)
    assert d <= 128, "FlashAttention only support head dimensions up to 128"
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
    assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert q.is_cuda and k.is_cuda and v.is_cuda
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)

    has_bias = bias is not None
    bias_type = "none"
    if has_bias:
        assert bias.dtype in [q.dtype, torch.float]
        assert bias.is_cuda
        assert bias.dim() == 4
        if bias.stride(-1) != 1:
            bias = bias.contiguous()
        if bias.shape[2:] == (1, seqlen_k):
            bias_type = "vector"
        elif bias.shape[2:] == (seqlen_q, seqlen_k):
            bias_type = "matrix"
        else:
            raise RuntimeError(
                "Last 2 dimensions of bias must be (1, seqlen_k)" " or (seqlen_q, seqlen_k)"
            )
        bias = bias.expand(batch, nheads, seqlen_q, seqlen_k)
    bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2)) if has_bias else (0, 0, 0)

    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    lse = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    tmp = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    o = torch.empty_like(q)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    BLOCK = 128
    num_warps = 4 if d <= 64 else 8
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
    _fwd_kernel[grid](
        q,
        k,
        v,
        bias,
        o,
        lse,
        tmp,
        softmax_scale,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        *bias_strides,
        o.stride(0),
        o.stride(2),
        o.stride(1),
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        d,
        seqlen_q // 32,
        seqlen_k // 32,  # key for triton cache (limit number of compilations)
        # Can't use kwargs here because triton autotune expects key to be args, not kwargs
        # IS_CAUSAL=causal, BLOCK_HEADDIM=d,
        bias_type,
        causal,
        BLOCK_HEADDIM,
    )
    return o, lse, softmax_scale  # softmax_scale could have been updated

def padding_bshd(t):   # BSHD
    batch, seqlen, nheads, dim = t.shape
    # t = torch.nn.functional.pad(t.reshape(batch, seqlen, nheads*dim), (0, 32), 'constant', 0)[:,:,:-32].reshape(batch, seqlen, nheads, dim) # pad: nheads*dim+32
    t = torch.nn.functional.pad(t.reshape(batch, seqlen, nheads, dim), (0, 32), 'constant', 0)[:,:,:,:-32] # pad: dim+32
    return t

def _flash_attn_backward(
    do, q, k, v, o, lse, dq, dk, dv, bias=None, causal=False, softmax_scale=None
):
    # Make sure that the last dimension is contiguous
    if do.stride(-1) != 1:
        do = do.contiguous()
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, _ = k.shape
    # assert d in {16, 32, 64, 128}
    assert d <= 128
    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    assert lse.shape == (batch, nheads, seqlen_q_rounded)
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == o.stride(-1) == 1
    assert dq.stride(-1) == dk.stride(-1) == dv.stride(-1) == 1
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)
    # dq_accum = torch.zeros_like(q, dtype=torch.float32)
    dq_accum = torch.empty_like(q, dtype=torch.float32)
    delta = torch.empty_like(lse)
    # delta = torch.zeros_like(lse)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
    _bwd_preprocess_do_o_dot[grid](
        o,
        do,
        delta,
        o.stride(0),
        o.stride(2),
        o.stride(1),
        do.stride(0),
        do.stride(2),
        do.stride(1),
        nheads,
        seqlen_q,
        seqlen_q_rounded,
        d,
        BLOCK_M=128,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
    )

    has_bias = bias is not None
    bias_type = "none"
    if has_bias:
        assert bias.dtype in [q.dtype, torch.float]
        assert bias.is_cuda
        assert bias.dim() == 4
        assert bias.stride(-1) == 1
        if bias.shape[2:] == (1, seqlen_k):
            bias_type = "vector"
        elif bias.shape[2:] == (seqlen_q, seqlen_k):
            bias_type = "matrix"
        else:
            raise RuntimeError(
                "Last 2 dimensions of bias must be (1, seqlen_k)" " or (seqlen_q, seqlen_k)"
            )
        bias = bias.expand(batch, nheads, seqlen_q, seqlen_k)
    bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2)) if has_bias else (0, 0, 0)

    BLOCK_M = 128
    BLOCK_N = 64
    # num_warps = 4
    grid = lambda META: (
        triton.cdiv(seqlen_k, META["BLOCK_N"]) if META["SEQUENCE_PARALLEL"] else 1,
        batch * nheads,
    )
    _bwd_kernel[grid](
        q,
        k,
        v,
        bias,
        do,
        dq_accum,
        dk,
        dv,
        lse,
        delta,
        softmax_scale,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        *bias_strides,
        do.stride(0),
        do.stride(2),
        do.stride(1),
        dq_accum.stride(0),
        dq_accum.stride(2),
        dq_accum.stride(1),
        dk.stride(0),
        dk.stride(2),
        dk.stride(1),
        dv.stride(0),
        dv.stride(2),
        dv.stride(1),
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        d,
        seqlen_q // 32,
        seqlen_k // 32,  # key for triton cache (limit number of compilations)
        # Can't use kwargs here because triton autotune expects key to be args, not kwargs
        # IS_CAUSAL=causal, BLOCK_HEADDIM=d,
        bias_type,
        causal,
        BLOCK_HEADDIM,
        SEQUENCE_PARALLEL=False,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        # num_warps=num_warps,
        # num_stages=1,
    )
    dq.copy_(dq_accum)

class MetaData():
    cu_seqlens_q = None
    cu_seqlens_k = None
    max_seqlens_q = 0
    max_seqlens_k = 0
    bias = None
    alibi_slopes = None
    causal = False
    num_contexts = 0
    varlen = False
    dropout_p, return_encoded_softmax = 0.0, False

    def __init__(self, sm_scale=1.0, causal=False, dropout_p=0.0, return_encoded_softmax=False):
        self.sm_scale = sm_scale
        self.causal = causal
        self.dropout_p = dropout_p
        self.return_encoded_softmax = return_encoded_softmax

    def set_varlen_params(self, cu_seqlens_q, cu_seqlens_k):
        self.varlen = True
        self.cu_seqlens_q = cu_seqlens_q
        self.cu_seqlens_k = cu_seqlens_k
        # Without "varlen", there should still be one sequence.
        assert len(cu_seqlens_q) >= 2
        assert len(cu_seqlens_q) == len(cu_seqlens_k)
        self.num_contexts = len(cu_seqlens_q) - 1
        for i in range (0, self.num_contexts):
            self.max_seqlens_q = max(cu_seqlens_q[i+1].item() - cu_seqlens_q[i].item(), self.max_seqlens_q)
            self.max_seqlens_k = max(cu_seqlens_k[i+1].item() - cu_seqlens_k[i].item(), self.max_seqlens_k)

    def need_bias(self, bias, batch, nheads, seqlen_q, seqlen_k):
        assert bias.is_cuda
        assert bias.dim() == 4
        assert bias.shape[0] == 1
        assert bias.shape[2:] == (seqlen_q, seqlen_k)
        self.bias = bias

    def need_alibi(self, alibi_slopes, batch, nheads):
        assert alibi_slopes.is_cuda
        assert alibi_slopes.dim() == 2
        assert alibi_slopes.shape[0] == batch
        assert alibi_slopes.shape[1] == nheads
        self.alibi_slopes = alibi_slopes

    def need_causal(self):
        self.causal = True

    def need_dropout(dropout_p, return_encoded_softmax):
        self.dropout_p = dropout_p
        self.return_encoded_softmax = return_encoded_softmax

    def check_args(self, q, k, v, o):
        assert q.dim() == k.dim() and q.dim() == v.dim()
        if self.varlen:
            assert q.dim() == 3
            total_q, nheads_q, head_size = q.shape
            total_k, nheads_k, _ = k.shape
            assert self.cu_seqlens_q is not None
            assert self.cu_seqlens_k is not None
            assert len(self.cu_seqlens_q) == len(self.cu_seqlens_k)
            # TODO: Remove once bias is supported with varlen
            assert self.bias == None
            # TODO:Remove once dropout is supported with varlen
            assert self.dropout_p == 0.0
            assert not self.return_encoded_softmax
        else:
            assert q.dim() == 4
            batch, nheads_q, seqlen_q, head_size = q.shape
            _, nheads_k, seqlen_k, _ = k.shape
            assert self.max_seqlens_q > 0 and self.max_seqlens_k > 0
            assert self.cu_seqlens_q is None and self.cu_seqlens_k is None
        assert k.shape == v.shape
        assert q.shape[-1] == k.shape[-1] and q.shape[-1] == v.shape[-1]
        # TODO: Change assert if we support qkl f8 and v f16
        assert q.dtype == k.dtype and q.dtype == v.dtype
        assert head_size <= 256
        assert o.shape == q.shape
        assert (nheads_q % nheads_k) == 0

class FlashAttnQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv, bias=None, causal=False, softmax_scale=None):
        """
        qkv: (batch, seqlen, 3, nheads, headdim)
        bias: optional, shape broadcastible to (batch, nheads, seqlen, seqlen).
            For example, ALiBi mask for causal would have shape (1, nheads, 1, seqlen).
            ALiBi mask for non-causal would have shape (1, nheads, seqlen, seqlen)
        """
        # Make sure that the last dimension is contiguous
        if qkv.stride(-1) != 1:
            qkv = qkv.contiguous()
        o, lse, ctx.softmax_scale = _flash_attn_forward(
            qkv[:, :, 0],
            qkv[:, :, 1],
            qkv[:, :, 2],
            bias=bias,
            causal=causal,
            softmax_scale=softmax_scale,
        )
        ctx.save_for_backward(qkv, o, lse, bias)
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        qkv, o, lse, bias = ctx.saved_tensors
        assert not ctx.needs_input_grad[1], "FlashAttention does not support bias gradient yet"
        # Triton's autotune causes the Tensor._version to change, and so Pytorch autograd
        # does a memcpy. To avoid this we run in inference_mode, which doesn't track the version.
        with torch.inference_mode():
            dqkv = torch.empty_like(qkv)
            _flash_attn_backward(
                do,
                qkv[:, :, 0],
                qkv[:, :, 1],
                qkv[:, :, 2],
                o,
                lse,
                dqkv[:, :, 0],
                dqkv[:, :, 1],
                dqkv[:, :, 2],
                bias=bias,
                causal=ctx.causal,
                softmax_scale=ctx.softmax_scale,
            )
        return dqkv, None, None, None


flash_attn_qkvpacked_func = FlashAttnQKVPackedFunc.apply


class FlashAttnKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, bias=None, causal=False, softmax_scale=None):
        """
        q: (batch, seqlen_q, nheads, headdim)
        kv: (batch, seqlen_k, 2, nheads, headdim)
        bias: optional, shape broadcastible to (batch, nheads, seqlen_q, seqlen_k).
            For example, ALiBi mask for causal would have shape (1, nheads, 1, seqlen_k).
            ALiBi mask for non-causal would have shape (1, nheads, seqlen_q, seqlen_k)
        """
        # Make sure that the last dimension is contiguous
        q, kv = [x if x.stride(-1) == 1 else x.contiguous() for x in [q, kv]]
        o, lse, ctx.softmax_scale = _flash_attn_forward(
            q, kv[:, :, 0], kv[:, :, 1], bias=bias, causal=causal, softmax_scale=softmax_scale
        )
        ctx.save_for_backward(q, kv, o, lse, bias)
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, kv, o, lse, bias = ctx.saved_tensors
        if len(ctx.needs_input_grad) >= 3:
            assert not ctx.needs_input_grad[2], "FlashAttention does not support bias gradient yet"
        # Triton's autotune causes the Tensor._version to change, and so Pytorch autograd
        # does a memcpy. To avoid this we run in inference_mode, which doesn't track the version.
        with torch.inference_mode():
            dq = torch.empty_like(q)
            dkv = torch.empty_like(kv)
            _flash_attn_backward(
                do,
                q,
                kv[:, :, 0],
                kv[:, :, 1],
                o,
                lse,
                dq,
                dkv[:, :, 0],
                dkv[:, :, 1],
                bias=bias,
                causal=ctx.causal,
                softmax_scale=ctx.softmax_scale,
            )
        return dq, dkv, None, None, None


flash_attn_kvpacked_func = FlashAttnKVPackedFunc.apply


# class FlashAttnFunc(torch.autograd.Function):
    # @staticmethod
def mha_forward(q, k, v, bias=None, causal=False, softmax_scale=None):
    """
    q: (batch_size, seqlen_q, nheads, headdim)
    k, v: (batch_size, seqlen_k, nheads, headdim)
    bias: optional, shape broadcastible to (batch, nheads, seqlen_q, seqlen_k).
        For example, ALiBi mask for causal would have shape (1, nheads, 1, seqlen_k).
        ALiBi mask for non-causal would have shape (1, nheads, seqlen_q, seqlen_k)
    """
    # Make sure that the last dimension is contiguous
    q, k, v = [x if x.stride(-1) == 1 else x.contiguous() for x in [q, k, v]]
    o, lse, softmax_scale = _flash_attn_forward(
        q, k, v, bias=bias, causal=causal, softmax_scale=softmax_scale
    )
    # ctx.save_for_backward(q, k, v, o, lse, bias)
    # ctx.causal = causal
    return o, lse, softmax_scale

    # @staticmethod
def mha_backward(do, q, k, v, o, lse, causal, softmax_scale, bias=None):
    # q, k, v, o, lse, bias = ctx.saved_tensors
    # assert not ctx.needs_input_grad[3], "FlashAttention does not support bias gradient yet"
    # Triton's autotune causes the Tensor._version to change, and so Pytorch autograd
    # does a memcpy. To avoid this we run in inference_mode, which doesn't track the version.
    # with torch.inference_mode():
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    _flash_attn_backward(
        do,
        q,
        k,
        v,
        o,
        lse,
        dq,
        dk,
        dv,
        bias=bias,
        causal=causal,
        softmax_scale=softmax_scale,
    )
    return dq, dk, dv
# class FlashAttnFunc(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, q, k, v, bias=None, causal=False, softmax_scale=None):
#         """
#         q: (batch_size, seqlen_q, nheads, headdim)
#         k, v: (batch_size, seqlen_k, nheads, headdim)
#         bias: optional, shape broadcastible to (batch, nheads, seqlen_q, seqlen_k).
#             For example, ALiBi mask for causal would have shape (1, nheads, 1, seqlen_k).
#             ALiBi mask for non-causal would have shape (1, nheads, seqlen_q, seqlen_k)
#         """
#         # Make sure that the last dimension is contiguous
#         q, k, v = [x if x.stride(-1) == 1 else x.contiguous() for x in [q, k, v]]
#         o, lse, ctx.softmax_scale = _flash_attn_forward(
#             q, k, v, bias=bias, causal=causal, softmax_scale=softmax_scale
#         )
#         ctx.save_for_backward(q, k, v, o, lse, bias)
#         ctx.causal = causal
#         return o

#     @staticmethod
#     def backward(ctx, do):
#         q, k, v, o, lse, bias = ctx.saved_tensors
#         assert not ctx.needs_input_grad[3], "FlashAttention does not support bias gradient yet"
#         # Triton's autotune causes the Tensor._version to change, and so Pytorch autograd
#         # does a memcpy. To avoid this we run in inference_mode, which doesn't track the version.
#         with torch.inference_mode():
#             dq = torch.empty_like(q)
#             dk = torch.empty_like(k)
#             dv = torch.empty_like(v)
#             _flash_attn_backward(
#                 do,
#                 q,
#                 k,
#                 v,
#                 o,
#                 lse,
#                 dq,
#                 dk,
#                 dv,
#                 bias=bias,
#                 causal=ctx.causal,
#                 softmax_scale=ctx.softmax_scale,
#             )
#         return dq, dk, dv, None, None, None


# triton_flash_attn_func = FlashAttnFunc.apply

try:
    from flash_attn.flash_attn_interface import flash_attn_func as _C_flashattention
    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

def native_multi_head_attention_2(q, k, v, mask=None, mask_type=None):
    k1 = k.transpose(-2, -1)
    qkt = torch.matmul(q, k1)
    if mask_type == 0 and mask is not None:
        qkt.masked_fill_(mask, -float('inf'))  # Apply the mask
    qkt_1 = F.softmax(qkt, dim=-1)
    return torch.matmul(qkt_1, v), qkt, qkt_1

mha_data = {
    # arugment:
    # key(tuple):
    # batch_size, nheads, seq_len, head_size, causal, dtype, n_warm, softmax_scale
    # value[list]:
    # [fw_tflops, bw_tflops]
    'gfx928': {
        (1, 16, 8192, 128, True, 'float16', 1, 1): [31.339, 0.432],
        (1, 32, 8192, 128, True, 'float16', 1, 1): [10.267, 0.706],
        (1, 8, 8192, 128, True, 'float16', 1, 1): [30.159, 0.235],
        (1, 13, 8192, 128, True, 'float16', 1, 1): [40.987, 0.373],
        (1, 26, 8192, 128, True, 'float16', 1, 1): [47.708, 0.635],
        (1, 52, 8192, 128, True, 'float16', 1, 1): [50.962, 0.866],
    }
}
@pytest.mark.parametrize("batch_size, nheads, seq_len, head_size, causal, dtype, n_warm, softmax_scale",
                         [(batch_size, nheads, seq_len, head_size, causal, dtype, n_warm, softmax_scale)
                          for batch_size, nheads, seq_len, head_size, causal, dtype, n_warm, softmax_scale
                          in mha_data[ARCH_NAME].keys()])
def test_mha(batch_size, nheads, seq_len, head_size, causal, dtype, n_warm, softmax_scale):

    bias = None
    device = 'cuda'
    if dtype == "float16":
        data_type = torch.float16
    else:
        data_type = torch.bfloat16

    q_gold = torch.randn(batch_size, seq_len, nheads, head_size, device=device, dtype=data_type, requires_grad=True)
    k_gold = torch.randn(batch_size, seq_len, nheads, head_size, device=device, dtype=data_type, requires_grad=True)
    v_gold = torch.randn(batch_size, seq_len, nheads, head_size, device=device, dtype=data_type, requires_grad=True)
    dout_gold = torch.rand_like(q_gold)

    q_triton = q_gold.clone().contiguous()
    k_triton = k_gold.clone().contiguous()
    v_triton = v_gold.clone().contiguous()
    dout_triton = dout_gold.clone()

    # # 定义数据
    # input_data = [
    #     ["batchsize", "qhead", "kvhead", "seqlen", "headdim", "causal_mask"],
    #     [batch_size, nheads, nheads, seq_len, head_size, causal]
    # ]

    # # 使用 tabulate 格式化输出
    # print(tabulate(input_data, headers="firstrow", tablefmt="grid"))

    # print("===================== Triton  FWD & BWD ====================")
    # Triton FWD warm up
    for i in range(n_warm):
        out_triton = mha_forward(q_triton, k_triton, v_triton, bias, causal, softmax_scale)
    # FWD time perf
    torch.cuda.synchronize()
    start_time = time.time()
    out_triton, triton_lse, triton_softmax_scale = mha_forward(q_triton, k_triton, v_triton, bias, causal, softmax_scale)
    torch.cuda.synchronize()
    end_time = time.time()
    triton_fwd_time = (end_time - start_time)*1000

    # print(f"Triton Forward Time: {(end_time - start_time)*1000:.2f} ms")
    # Triton BWD warm up
    for i in range(n_warm):
        gradients = mha_backward(dout_triton, q_triton, k_triton, v_triton, out_triton, triton_lse, causal, triton_softmax_scale, bias=None)
    # BWD time perf
    torch.cuda.synchronize()
    start_time = time.time()
    gradients = mha_backward(dout_triton, q_triton, k_triton, v_triton, out_triton, triton_lse, causal, triton_softmax_scale, bias=None)
    torch.cuda.synchronize()
    end_time = time.time()
    triton_bwd_time = (end_time - start_time)*1000
    # print(f"Triton Backward Time: {(end_time - start_time)*1000:.2f} ms")
    dq_triton = gradients[0]
    dk_triton = gradients[1]
    dv_triton = gradients[2]
    # print("backward triton q: ", dk_triton[0][0][0][0])

    # if HAS_FLASH:
    # print("===================== Flash FWD & BWD ====================")
    # q_flash = q_gold.clone().permute(0, 2, 1, 3).contiguous()
    # k_flash = k_gold.clone().permute(0, 2, 1, 3).contiguous()
    # v_flash = v_gold.clone().permute(0, 2, 1, 3).contiguous()

    # # dq_flash = torch.empty_like(dout_triton)
    # # dk_flash = torch.empty_like(dout_triton)
    # # dv_flash = torch.empty_like(dout_triton)
    # dout_triton = dout_gold.clone().permute(0, 2, 1, 3).contiguous()

    # for _ in range(n_warm):
    #     out_flash_all = _C_flashattention.fwd(q_flash, k_flash, v_flash, None,  dropout_p, softmax_scale,
    #                                       causal, -1, -1, False,  None, )
    # torch.cuda.synchronize()
    # start_time = time.time()
    # out_flash_all = _C_flashattention.fwd(q_flash, k_flash, v_flash, None, dropout_p, softmax_scale,
    #                                     causal, -1, -1, False,  None, )
    # torch.cuda.synchronize()
    # end_time = time.time()

    # # out_flash=out_flash.permute(0, 2, 1, 3).contiguous()
    # print(f"Flash Forward Time: {(end_time - start_time)*1000:.2f} ms")
    # out_flash, flash_softmax_lse = out_flash_all[0], out_flash_all[5]

    # dq_flash, dk_flash, dv_flash, _ = _C_flashattention.bwd(
    #     dout_triton, q_flash, k_flash, v_flash, out_flash, flash_softmax_lse, None, None, None, 0.0,
    #     softmax_scale, causal, -1,-1, None, None
    # )
    # (arg0: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor, arg3: torch.Tensor, arg4: torch.Tensor,
    #  arg5: torch.Tensor, arg6: Optional[torch.Tensor], arg7: Optional[torch.Tensor], arg8: Optional[torch.Tensor], arg9: float,
    #  arg10: float, arg11: bool, arg12: int, arg13: int, arg14: Optional[torch.Generator], arg15: Optional[torch.Tensor])
    # print("backward flash q: ", dk_flash[0][0][0][0])

    # out_flash=out_flash.permute(0, 2, 1, 3).contiguous()
    # print("===================== Torch  FWD & BWD ====================")
    q_ref = q_gold.clone().permute(0, 2, 1, 3).contiguous()
    k_ref = k_gold.clone().permute(0, 2, 1, 3).contiguous()
    v_ref = v_gold.clone().permute(0, 2, 1, 3).contiguous()

    # 获取下三角
    mask = torch.ones(q_gold.size(-3), k_gold.size(-3), dtype=torch.bool, device=q_gold.device).tril().logical_not() if causal else None
    mask_type = 0 if causal else None

    dout_ref = dout_gold.clone().permute(0, 2, 1, 3).contiguous()

    torch.cuda.synchronize()
    start_time = time.time()
    out_ref = native_multi_head_attention_2(q_ref, k_ref, v_ref, mask, mask_type)
    torch.cuda.synchronize()
    end_time = time.time()
    torch_fwd_time = (end_time - start_time)*1000
    # 计算梯度
    torch.cuda.synchronize()
    start_time = time.time()
    gradients_ref = torch.autograd.grad(outputs=[out_ref[0]], inputs=[q_ref, k_ref, v_ref], grad_outputs=[dout_ref])
    torch.cuda.synchronize()
    end_time = time.time()
    torch_bwd_time = (end_time - start_time)*1000

    dq_ref = gradients_ref[0].permute(0, 2, 1, 3).contiguous()
    dk_ref = gradients_ref[1].permute(0, 2, 1, 3).contiguous()
    dv_ref = gradients_ref[2].permute(0, 2, 1, 3).contiguous()
    # print("backward ref q: ", dk_ref[0][0][0][0])
    error_fwd = compare_tensors(out_triton,out_ref[0].transpose(2,1))
    error_bwd= compare_tensors(dk_triton,dk_ref)

    # visualized output
    headers = ["triton", "", "torch", "", "Error", ""]
    sub_headers = ["fwd_time(ms)", "bwd_time(ms)", "fwd_time(ms)", "bwd_time(ms)", "fwd_error", "bwd_error"]

    flops_per_matmul = 2.0 * batch_size * nheads * seq_len * seq_len * head_size
    total_flops_fw = 2 * flops_per_matmul
    if causal:
        total_flops_fw *= 0.5
    total_flops_bw = total_flops_fw * 2.5 # 2.0(bwd) + 0.5(recompute)
    cur_gpu_perf_fw = total_flops_fw / triton_fwd_time * 1e-9
    cur_gpu_perf_bw = total_flops_bw / triton_bwd_time * 1e-9

    ref_gpu_perf = mha_data[ARCH_NAME][(batch_size, nheads, seq_len, head_size,
                                        causal, dtype, n_warm, softmax_scale)]
    check_perf(triton_fwd_time, cur_gpu_perf_fw, ref_gpu_perf[0], atol=1,
               args=f"\nArguments: ({batch_size}, {nheads}, {seq_len}, {head_size}, {causal}, {dtype}, {n_warm}, {softmax_scale})")
    check_perf(triton_bwd_time, cur_gpu_perf_bw, ref_gpu_perf[1], atol=1,
               args=f"\nArguments: ({batch_size}, {nheads}, {seq_len}, {head_size}, {causal}, {dtype}, {n_warm}, {softmax_scale})")

