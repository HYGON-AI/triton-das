"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)
Credits: OpenAI kernel team

Extra Credits:
- Original flash attention paper (https://arxiv.org/abs/2205.14135)
- Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

"""

import pytest
import torch
import os

import triton
import triton.language as tl

class InterleaveManager:
    KEY = "CHAINED_DOT_SHORTCUT"

    @classmethod
    def enable(cls):
        os.environ[cls.KEY] = 'ON'

    @classmethod
    def disable(cls):
        if cls.KEY in os.environ:
            del os.environ[cls.KEY]

    @classmethod
    def is_enabled(cls):
        return os.getenv(cls.KEY) == 'ON'

    @classmethod
    def reset(cls, enable):
        if enable:
            cls.enable()
        else:
            cls.disable()


name_to_torch_types = {
    'fp16': torch.float16,
}

ORIGIN_INTERLEAVE_ENABLED = InterleaveManager.is_enabled()

# Turn it on in need. ONLY work for fwd. Manually disabled in bwd.
# FIXME: Interleave is NOT supported for now.
ENABLE_FWD_INTERLEAVE = False

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,
                    K_block_ptr, V_block_ptr,
                    start_m,
                    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,
                    N_CTX,
                    pre_load_v: tl.constexpr):
    # range of values handled by this stage
    #STAGE==1是为了当casual=true时非mask部分
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    #STAGE==2是与STAGE==1相对应的，当casual=true时，计算非mask部分。
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
        K_block_ptr = tl.advance(K_block_ptr, (0, lo))
        V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # causal = False
    else:
        lo, hi = 0, N_CTX
    #研究这段代码我们可以直接先按照causal=false的情况分析，
    #Q已经拿到数据块了，是[BLOCK_M,BLOCK_DMODEL]大小的矩阵，然后循环处理K，V，
    #从0到N_CTX每次去BLOCK_N宽度，既k每次取[BLOCK_DMODEL,BLOCK_N],V每次取[BLOCK_N,BLOCK_DMODEL]
    #这个地方就是我们公式中所描述的分块累加既f(1),f(2),L(1),L(2)的计算。
    #目前要循环N=4096/BLOCK_N次，既我们会有N个局部最大值。
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)
        if pre_load_v:
            v = tl.load(V_block_ptr)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n)
            qk = tl.where(mask, qk, float("-inf"))
        qk += tl.dot(q, k)
        #找出当前块以及之前的所有块中最大的值。
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        #求出公式中的Xi-Mn
        qk = qk - m_ij[:, None]
        #求局部的P
        p = tl.math.exp2(qk)
        # -- update output accumulator --
        #这都是公式中的变换e(m1-m2)
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        if not pre_load_v:
            v = tl.load(V_block_ptr)
        acc += tl.dot(p.to(v.dtype), v)
        # -- update m_i and l_i
        l_ij = tl.sum(p, 1)
        l_i = l_i * alpha + l_ij
        # update m_i and l_i
        m_i = m_ij
        #循环都最后就可以求出整行的最大值m_i和分母l_i了
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i


# We don't run auto-tuning everytime to keep the tutorial fast. Uncommenting
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
@triton.autotune(
   configs=[
       triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 0, 'pre_load_v': False}, num_stages=1, num_warps=4),
       triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 0, 'pre_load_v': True}, num_stages=1, num_warps=4),
       triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'waves_per_eu': 1, 'pre_load_v': False}, num_stages=1, num_warps=2),
   ],
   key=['Z', 'H', 'N_CTX', 'STAGE', 'BLOCK_DMODEL'],
)


@triton.jit
def _attn_fwd(Q, K, V, VT, sm_scale, M, Out,
              stride_qz, stride_qh, stride_qm, stride_qk,
              stride_kz, stride_kh, stride_kn, stride_kk,
              stride_vz, stride_vh, stride_vk, stride_vn,
              stride_vtz, stride_vth, stride_vtn, stride_vtk,
              stride_oz, stride_oh, stride_om, stride_on,
              Z, H, k_head, q_num_per_group,
              N_CTX,
              BLOCK_DMODEL: tl.constexpr,
              STAGE: tl.constexpr,
              BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr,
              pre_load_v: tl.constexpr,
              GRID_AXIS_HZ: tl.constexpr,
              INTERLEAVE: tl.constexpr
              ):
    if GRID_AXIS_HZ == 0:
        start_m = tl.program_id(1)
        off_hz = tl.program_id(0)
    else:
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)

    # Reverse the ordering of blocks, issue heavy-load blocks first, issue light-load blocks last
    # So that to reduce the idle state of simd wave slots
    start_m = tl.cdiv(N_CTX, BLOCK_M) - 1 - start_m

    off_kh = off_hz // q_num_per_group
    #x方向的线程块能处理的所有数据大小，既N_CTX*BLOCK_DMODEL,(4096,128),
    q_offset = off_hz * stride_qh
    kv_offset = off_kh * stride_kh
    #从代码中分析得出，x方向上的所有线程块负责处理N_CTX*BLOCK_DMODEL,y方向对应[batch*n_head],故就有y*x=batch*n_head*x个线程块。
    # block pointers
    #分割数据块Q，每个线程块分配的数据块大小为[BLOCK_M,BLOCK_DMODEL]
    #该线程块中的每个线程都能拿到该数据块的首地址。
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    #同理分割数据块V，每个线程块分配的数据块大小为[BLOCK_N,BLOCK_DMODEL]
    #该线程块中的每个线程都能拿到该数据块的首地址。
    # V_block_ptr = tl.make_block_ptr(
    #     base=V + kv_offset,
    #     shape=(N_CTX, BLOCK_DMODEL),
    #     strides=(stride_vk, stride_vn),
    #     offsets=(0, 0),
    #     block_shape=(BLOCK_N, BLOCK_DMODEL),
    #     order=(1, 0),
    # )

    # Note: bmz add to increase lds coalesce
    V_block_ptr = tl.make_block_ptr(
        base=VT + kv_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vtk, stride_vtn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(0, 1),
    )

    #同理分割数据块K
    #该线程块中的每个线程都能拿到该数据块的首地址。
    K_block_ptr = tl.make_block_ptr(
        base=K + kv_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    #同理分割数据块O
    #该线程块中的每个线程都能拿到该数据块的首地址。
    O_block_ptr = tl.make_block_ptr(
        base=Out + q_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    # initialize offsets
    #Q为外部循环处理，对应并行计算处理中，一个线程块负责BLOCK_M行的数据处理。所以每个线程块内部
    #要循环处理kv的数据。
    #KV为内部循环处理，
    #根据线程块计算每个线程的偏移量，triton编程中只是指定了grid，并没有指定block,所以线程块内部的线程布局有待分析。是每个线程负责一行的数据处理还是多个线程负责一行数据的处理，这个要等后面分析triton编程后给出定论。
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    #为每行分配临时变量Mi(每行的临时最大值),Li(每行的权重分母),acc是累加和
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    #这代码搞让人头晕，这个常数1.44269504分析了半天才明白它代表e的值，是用2的1.44269504次方代表e的值。
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout on NV GPUs but in VGPRs on AMD GPUs
    #tl.load()导入的数据在AMD的GPU上竟然是放在VGPR上的，有些意外。
    q = tl.load(Q_block_ptr)
    #直接先把scale先融入到Q中，这样后面的计算直接不用写那么繁琐了。
    q = (q * qk_scale).to(q.dtype)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    #当casual=True时，stage=3,这时STAGE&1和STAGW&2都成立，然后分两步进行计算
    #第一步（STAGE&1)计算不用考虑mask的数据块，第二步(STAGE&2)计算mask有影响的数据块。
    ###############################################################

    offs_n_expand = offs_n[None, :]
    offs_n_broad = tl.broadcast_to(offs_n_expand, (BLOCK_M, BLOCK_N))
    if INTERLEAVE:
        offs_n = (offs_n_broad % 4) * 4 + (offs_n_broad % 16) // 4 + (offs_n_broad // 16) * 16
    else:
        offs_n = offs_n_broad

    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,
                                        start_m,
                                        BLOCK_M, BLOCK_DMODEL, BLOCK_N,
                                        4 - STAGE, offs_m, offs_n, N_CTX,
                                        pre_load_v,
                                        )
    # stage 2: on-band
    #计算受到mask影响的数据块。
    if STAGE & 2:
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        tl.debug_barrier()
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,
                                        start_m,
                                        BLOCK_M, BLOCK_DMODEL, BLOCK_N,
                                        2, offs_m, offs_n, N_CTX,
                                        pre_load_v,
                                        )
    # epilogue
    # write back m
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i + tl.math.log2(l_i))
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))

@triton.jit
def _attn_bwd_preprocess(O, DO,
                         Delta,
                         Z, H, N_CTX,
                         BLOCK_M: tl.constexpr, D_HEAD: tl.constexpr
                         ):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, D_HEAD)
    o = tl.load(O + off_hz * D_HEAD * N_CTX + off_m[:, None] * D_HEAD + off_n[None, :])
    do = tl.load(DO + off_hz * D_HEAD * N_CTX + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + off_hz * N_CTX + off_m, delta)


# The main inner-loop logic for computing dK and dV.
@triton.jit
def _attn_bwd_dkdv(dk, dv,
                   Q, QT, k, v, sm_scale,
                   DO,
                   M, D,
                   # shared by Q/K/V/DO.
                   stride_tok, stride_d,
                   # shared by QT/KT/VT/DOT.
                   stride_td, stride_ttok,
                   H, N_CTX, BLOCK_M1: tl.constexpr,
                   BLOCK_N1: tl.constexpr,
                   BLOCK_DMODEL: tl.constexpr,
                   # Filled in by the wrapper.
                   start_n, start_m, num_steps,
                   MASK: tl.constexpr):
    #也可以用来求取M最大值对应的下标。
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, BLOCK_DMODEL)

    # Note: bmz add to increase load/store coalesce and increase lds coalesce, not fit for zde
    Q_block_ptr = tl.make_block_ptr(
        base=QT,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_ttok, stride_td),
        offsets=(start_m, 0),
        block_shape=(BLOCK_M1, BLOCK_DMODEL),
        order=(0,1)
    )

    QT_block_ptr = tl.make_block_ptr(
        base=Q,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_d, stride_tok),
        offsets=(0, start_m),
        block_shape=(BLOCK_DMODEL, BLOCK_M1),
        order=(0,1)
    )
    DO_block_ptr = tl.make_block_ptr(
        base=DO,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_tok, stride_d),
        offsets=(start_m, 0),
        block_shape=(BLOCK_M1, BLOCK_DMODEL),
        order=(1,0)
    )
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    #这里的BLOCK_M1与设置的BLOCK_M1意义不一样，这是形参的名字,实际是MASK_BLOCK_M1
    step_m = BLOCK_M1
    for blk_idx in range(num_steps):
        qT = tl.load(QT_block_ptr)
        # Load m before computing qk to reduce pipeline stall.
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)
        qkT = tl.dot(k, qT)
        pT = tl.math.exp2(qkT - m[None, :])
        # Autoregressive masking.
        if MASK:
            mask = (offs_m[None, :] >= offs_n[:, None])
            pT = tl.where(mask, pT, 0.0)
        do = tl.load(DO_block_ptr)
        # Compute dV.
        ppT = pT
        ppT = ppT.to(tl.float16)
        #因为最后是做累加运算，所以才可以循环分块处理
        dv += tl.dot(ppT, do)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m)

        q = tl.load(Q_block_ptr)

        # Compute dP and dS.
        #公式中都是在计算dp和ds，在求dk时用的是dsT所以直接将dp和ds进行转置后就可以直接
        #求dk了。
        dpT = tl.dot(v, tl.trans(do))
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(tl.float16)

        # dk += tl.dot(dsT, tl.trans(qT))
        dk += tl.dot(dsT, q)
        # Increment pointers.
        curr_m += step_m
        QT_block_ptr = tl.advance(QT_block_ptr, (0, step_m))
        DO_block_ptr = tl.advance(DO_block_ptr, (step_m, 0))
        Q_block_ptr = tl.advance(Q_block_ptr, (step_m, 0))
    return dk, dv


# the main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_dq(dq, q, K, KT, V,
                 do, m, D,
                 # shared by Q/K/V/DO.
                 stride_tok, stride_d,
                 # shared by QT/KT/VT/DOT
                 stride_td, stride_ttok,
                 H, N_CTX,
                 BLOCK_M2: tl.constexpr,
                 BLOCK_N2: tl.constexpr,
                 BLOCK_DMODEL: tl.constexpr,
                 # Filled in by the wrapper.
                 start_m, start_n, num_steps,
                 MASK: tl.constexpr,
                 GQA_FLAG: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, BLOCK_DMODEL)
    KT_block_ptr = tl.make_block_ptr(
        base=K,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_d, stride_tok),
        offsets=(0, start_n),
        block_shape=(BLOCK_DMODEL, BLOCK_N2),
        order=(0, 1)
    )

    if GQA_FLAG:
        # Note: bmz add to increase load/store coalesce and increase lds coalesce, not fit for zde
        K_block_ptr = tl.make_block_ptr(
            base=KT,
            shape=(N_CTX, BLOCK_DMODEL),
            strides=(stride_ttok, stride_td),
            offsets=(start_n, 0),
            block_shape=(BLOCK_N2, BLOCK_DMODEL),
            order=(0,1)
        )

    VT_block_ptr = tl.make_block_ptr(
        base=V,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_d, stride_tok),
        offsets=(0, start_n),
        block_shape=(BLOCK_DMODEL, BLOCK_N2),
        order=(0, 1)
    )
    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_m)
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    for blk_idx in range(num_steps):
        kT = tl.load(KT_block_ptr)
        qk = tl.dot(q, kT)
        p = tl.math.exp2(qk - m)
        # Autoregressive masking.
        if MASK:
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            mask = (offs_m[:, None] >= offs_n[None, :])
            p = tl.where(mask, p, 0.0)
        # Compute dP and dS.
        vT = tl.load(VT_block_ptr)
        dp = tl.dot(do, vT).to(tl.float32)

        ds = p * (dp - Di[:, None])
        ds = ds.to(tl.float16)
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        if GQA_FLAG:
            k = tl.load(K_block_ptr)
            dq += tl.dot(ds, k)
        else:
            dq += tl.dot(ds, tl.trans(kT))

        # Increment pointers.
        curr_n += step_n
        KT_block_ptr = tl.advance(KT_block_ptr, (0, step_n))
        VT_block_ptr = tl.advance(VT_block_ptr, (0, step_n))

        if GQA_FLAG:
            K_block_ptr = tl.advance(K_block_ptr, (step_n, 0))
    return dq


@triton.autotune(
   configs=[
       triton.Config({'BLOCK_M1': 32, 'BLOCK_N1': 64, 'BLOCK_M2': 64, 'BLOCK_N2': 32, 'BLK_SLICE_FACTOR': 1},
                      num_stages=1, num_warps=4),
       triton.Config({'BLOCK_M1': 32, 'BLOCK_N1': 64, 'BLOCK_M2': 64, 'BLOCK_N2': 32, 'BLK_SLICE_FACTOR': 2},
                      num_stages=1, num_warps=4),
   ],
   key=['H', 'N_CTX', 'K_HEAD', 'BLOCK_DMODEL'],
)

@triton.jit
def _attn_bwd(Q, QT, K, KT, V, sm_scale,
              DO,
              DQ, DK, DV,
              M, D,
              # shared by Q/K/V/DO.
              stride_z, stride_h, stride_tok, stride_d,
              stride_kz,
              # shared by QT/KT/VT/DOT.
              stride_td, stride_ttok,
              # H = 16, N_CTX = 1024
              H, N_CTX, K_HEAD, q_num_per_group,
              GQA_FLAG: tl.constexpr,
              BLOCK_DMODEL: tl.constexpr,
              BLOCK_M1: tl.constexpr,
              BLOCK_N1: tl.constexpr,
              BLOCK_M2: tl.constexpr,
              BLOCK_N2: tl.constexpr,
              BLK_SLICE_FACTOR: tl.constexpr):
    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)

    #N_CTX*H*BATCH的偏移，其实就是head的在行上的偏移
    #z与BATCH，H与HEAD是对应的。
    bid  = tl.program_id(2)
    hid  = tl.program_id(1)
    bhid = bid*H + hid
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * hid + stride_z * bid).to(tl.int64)
    k_adj = (bid * stride_kz + (hid//q_num_per_group) * stride_h).to(tl.int64)
    pid = tl.program_id(0)
    # offset pointers for batch/head
    # offset pointers for batch/head
    #所有数据指针都移动到当前N_CTX*BLOCK_DMODEL页

    Q += adj
    QT+= adj
    K += k_adj
    KT+= k_adj
    V += k_adj
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz

    # THIS BLOCK DOES DQ:
    start_m = pid * BLOCK_M2
    end_n = start_m + BLOCK_M2

    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
    offs_m = start_m + tl.arange(0, BLOCK_M2)

    Q_block_ptr = tl.make_block_ptr(
        base=Q,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_tok, stride_d),
        offsets=(start_m, 0),
        block_shape=(BLOCK_M2, BLOCK_DMODEL),
        order=(1, 0)
    )

    DO_block_ptr = tl.make_block_ptr(
        base=DO,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_tok, stride_d),
        offsets=(start_m, 0),
        block_shape=(BLOCK_M2, BLOCK_DMODEL),
        order=(1, 0)
    )
    q = tl.load(Q_block_ptr)
    do = tl.load(DO_block_ptr)
    dq = tl.zeros([BLOCK_M2, BLOCK_DMODEL], dtype=tl.float32)

    m = tl.load(M + offs_m)
    m = m[:, None]

    # Compute dQ for masked (diagonal) blocks.
    # NOTE: This code scans each row of QK^T backward (from right to left,
    # but inside each call to _attn_bwd_dq, from left to right), but that's
    # not due to anything important.  I just wanted to reuse the loop
    # structure for dK & dV above as much as possible.
    num_steps = BLOCK_M2 // MASK_BLOCK_N2
    dq = _attn_bwd_dq(dq, q, K, KT, V,
                      do, m, D,
                      stride_tok, stride_d,
                      stride_td, stride_ttok,
                      H, N_CTX,
                      BLOCK_M2, MASK_BLOCK_N2, BLOCK_DMODEL,
                      start_m, end_n - num_steps * MASK_BLOCK_N2, num_steps,
                      MASK=True,
                      GQA_FLAG=GQA_FLAG
                      )
    end_n -= num_steps * MASK_BLOCK_N2
    # stage 2
    #与求dk,dv是类似的，只不过dk,dv是按照i方向从上到下，dq是按照j方向，从左到右。
    #都是累加下三角的数据，只不过dk，dv是下三角对列上的值进行累加，dq是对行上的所有值进行累加。

    num_steps = end_n // BLOCK_N2
    dq = _attn_bwd_dq(dq, q, K, KT, V,
                      do, m, D,
                      stride_tok, stride_d,
                      stride_td, stride_ttok,
                      H, N_CTX,
                      BLOCK_M2, BLOCK_N2, BLOCK_DMODEL,
                      start_m, end_n - num_steps * BLOCK_N2, num_steps,
                      MASK=False,
                      GQA_FLAG=GQA_FLAG
                      )
    # Write back dQ.
    DQ_block_ptr = tl.make_block_ptr(
        base=DQ,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_tok, stride_d),
        offsets=(start_m, 0),
        block_shape=(BLOCK_M2, BLOCK_DMODEL),
        order=(1, 0)
    )
    dq *= LN2
    tl.store(DQ_block_ptr, dq.to(tl.float16))



    # THIS BLOCK DOES DK & DV:
    offs_k = tl.arange(0, BLOCK_DMODEL)
    #每个线程块要处理K,V的位置
    start_n = pid * BLOCK_N1
    # This assignment is important. It is what allows us to pick the diagonal
    # blocks. Later, when we want to do the lower triangular, we update start_m
    # after the first dkdv call.
    #start_m=start_n这样就代表将mask的操作范围限制在了N_CTX*BLOCK_DMODEL内。
    #而且大小为BLOCK_N1*BLCOK_N1
    start_m = start_n
    #BLK_SLICE_FACTOR代表BLOCK_M1要分几步处理
    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    #对应论文中的中伪代码，初始化dk,dv

    dv = tl.zeros([BLOCK_N1, BLOCK_DMODEL], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, BLOCK_DMODEL], dtype=tl.float32)
    #对应论文中的伪代码，导入Kj，Vj,每个线程块获得自己需要处理的K和V的数据

    K_block_ptr = tl.make_block_ptr(
        base=K,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_tok, stride_d),
        offsets=(start_n, 0),
        block_shape=(BLOCK_N1, BLOCK_DMODEL),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_tok, stride_d),
        offsets=(start_n, 0),
        block_shape=(BLOCK_N1, BLOCK_DMODEL),
        order=(1, 0),
    )

    # load K and V: they stay in SRAM throughout the inner loop for dkdv.
    k = tl.load(K_block_ptr)
    v = tl.load(V_block_ptr)

    num_steps = BLOCK_N1 // MASK_BLOCK_M1

    dk, dv = _attn_bwd_dkdv(dk, dv,
                            Q, QT, k, v, sm_scale,
                            DO,
                            M, D,
                            stride_tok, stride_d,
                            stride_td, stride_ttok,
                            H, N_CTX,
                            MASK_BLOCK_M1, BLOCK_N1, BLOCK_DMODEL,
                            start_n, start_m, num_steps,
                            MASK=True
                            )
    #这个地方代码设计的非常巧妙,这个地方才会改变start_m，mask部分已经累加结束，现在是求没有mask部分。
    start_m += num_steps * MASK_BLOCK_M1
    #为啥是(N_CTX-start_m),因为在整体的P矩阵中根据mask对角线，只需要计算对角线下方的数据，这样
    #(N_CTX - start_m)之前的行数据全是0，也就不需要计算了。可以画出P（N，N）矩阵推演一下。
    num_steps = (N_CTX - start_m) // BLOCK_M1
    # Compute dK and dV for non-masked blocks.
    dk, dv = _attn_bwd_dkdv(
        dk, dv,
        Q, QT, k, v, sm_scale,
        DO,
        M, D,
        stride_tok, stride_d,
        stride_td, stride_ttok,
        H, N_CTX,
        BLOCK_M1, BLOCK_N1, BLOCK_DMODEL,
        start_n, start_m, num_steps,
        MASK=False
    )

    DV_block_ptrs = tl.make_block_ptr(
        base=DV,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_tok, stride_d),
        offsets=(start_n, 0),
        block_shape=(BLOCK_N1, BLOCK_DMODEL),
        order=(1,0)
    )
    tl.store(DV_block_ptrs, dv.to(tl.float16))

    # Write back dK.
    dk *= sm_scale
    DK_block_ptrs = tl.make_block_ptr(
        base=DK,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_tok, stride_d),
        offsets=(start_n, 0),
        block_shape=(BLOCK_N1, BLOCK_DMODEL),
        order=(1,0)
    )
    tl.store(DK_block_ptrs, dk.to(tl.float16))




class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, interleave):
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        q_head = q.shape[1]
        k_head = k.shape[1]
        q_num_per_group = q_head // k_head
        assert q_head % k_head == 0
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}

        # Note: bmz add to increase lds coalesce
        vt = v.transpose(-1, -2).contiguous()

        o = torch.empty_like(q, dtype=v.dtype)
        if torch.version.hip is None:
            BLOCK_M = 128
            BLOCK_N = 64 if Lk <= 64 else 32
            num_stages = 4 if Lk <= 64 else 3
            num_warps = 4 if Lk <= 64 else 8
            # Tuning for H100
            if torch.cuda.get_device_capability()[0] == 9:
                num_warps = 8
                num_stages = 7 if Lk >= 64 else 3
        #casual=true stage=3为什么设置成3，_attn_fwd函数内部有解释
        #casual=false stage=1
        stage = 3 if causal else 1
        #分配线程块


        if k_head < q_head:
            grid = lambda META: (
                q.shape[0] * q.shape[1],
                triton.cdiv(q.shape[2], META['BLOCK_M']),
                1
            )
            grid_axis_hz = 0
        else:
            # for MHA cases, reverse the ordering of the blocks would deteriotate performance,
            # switch to the grid layout of GQA would solve this issue.
            # TODO: why? Need to look into it
            grid = lambda META: (
                triton.cdiv(q.shape[2], META['BLOCK_M']),
                q.shape[0] * q.shape[1],
                1
            )
            grid_axis_hz = 1


        #M用于保留每行的最大值。所以M的维度就是[BATCH*N_HEAD,N_CTX]
        M = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        _attn_fwd[grid](
            q, k, v, vt, sm_scale, M, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            vt.stride(0), vt.stride(1), vt.stride(2), vt.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q.shape[0], q.shape[1], k_head, q_num_per_group,
            N_CTX=q.shape[2],
            BLOCK_DMODEL=Lk,
            STAGE=stage,
            GRID_AXIS_HZ=grid_axis_hz,
            INTERLEAVE=interleave
        )

        ## restore the grid for bwd kernel
        best_config = _attn_fwd.best_config
        block_m = int(best_config.__str__().split(",")[0].split("BLOCK_M:")[1])
        grid = (triton.cdiv(q.shape[2], block_m), q.shape[0] * q.shape[1], 1)

        ctx.save_for_backward(q, k, v, o, M)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        if torch.version.hip is not None:
            BLOCK = 64
        else:
            BLOCK = 128
        q, k, v, o, M = ctx.saved_tensors
        assert do.is_contiguous()
        #assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
        dq = torch.empty_like(q)
        dk = torch.empty_like(q)
        dv = torch.empty_like(q)

        BATCH, N_HEAD, N_CTX = q.shape[:3]
        K_HEAD = k.shape[1]
        q_num_per_group = N_HEAD // K_HEAD

        # Note: bmz add to increase load/store coalesce and increase lds coalesce, not fit for zde
        gqa_flag = N_HEAD != K_HEAD
        qt = q.transpose(-1, -2).contiguous()
        kt = k.transpose(-1, -2).contiguous() if gqa_flag else k

        PRE_BLOCK = 128
        NUM_WARPS, NUM_STAGES = 4, 1
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 64, 64, 32
        BLK_SLICE_FACTOR = 2
        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        arg_k = k
        arg_k = arg_k * (ctx.sm_scale * RCP_LN2)

        arg_kt = kt
        arg_kt = arg_kt * (ctx.sm_scale * RCP_LN2)

        assert N_CTX % PRE_BLOCK == 0
        pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
        delta = torch.empty_like(M)
        ##################################
        #提前计算rowsum(doi o oi),见公式，与公式对应。公式中求dsi的时候用到了这个。提前求出来。
        #块的大小为[128,128],既[PRE_BLOCK,BLOCK_DMODEL],这与计算dv，dk,dq的kernel不同。
        #################################
        _attn_bwd_preprocess[pre_grid](
            o, do,
            delta,
            BATCH, N_HEAD, N_CTX,
            BLOCK_M=PRE_BLOCK, D_HEAD=ctx.BLOCK_DMODEL
        )
        #K,V作为外部循环，每个线程块负责一块，Q作为内部循环，每个线程块都要循环遍历一遍。
        grid = lambda META: (
            triton.cdiv(N_CTX, META['BLOCK_N1']),
            N_HEAD,
            BATCH
        )
        _attn_bwd[grid](
            q, qt, arg_k, arg_kt, v, ctx.sm_scale, do, dq, dk, dv,
            M, delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),k.stride(0),
            qt.stride(2), qt.stride(3),
            N_HEAD, N_CTX, K_HEAD, q_num_per_group,
            GQA_FLAG=gqa_flag,
            BLOCK_DMODEL=ctx.BLOCK_DMODEL
        )

        if N_HEAD != K_HEAD:
            re_dk = dk.reshape(BATCH,K_HEAD,N_HEAD//K_HEAD,N_CTX,ctx.BLOCK_DMODEL)
            dk = re_dk.sum(dim=2,keepdim=False)
            re_dv = dv.reshape(BATCH,K_HEAD,N_HEAD//K_HEAD,N_CTX,ctx.BLOCK_DMODEL)
            dv = re_dv.sum(dim=2,keepdim=False)

        return dq, dk, dv, None, None, None

attention = _attention.apply

attention_perf_model_cases_list = [
    # 'Z, Q_H, K_H, N_CTX, D_HEAD'
    # Ali case
    (1, 16, 16, 8192, 128),
    (1, 32, 32, 8192, 128),
    (1, 32,  4, 8192, 128),
    (1, 52,  4, 8192, 128),
    (1, 16,  2, 8192, 128),
    (1, 26,  2, 8192, 128),
    (1,  8,  1, 8192, 128),
    (1, 13,  1, 8192, 128),
    # llama2-7b
    (1, 32, 32, 4096, 128),
    (1, 16, 16, 4096, 128),
    (1, 8,  8,  4096, 128),
    (1, 4,  4,  4096, 128),
    # llama2-13b
    (1, 40, 40, 4096, 128),
    (1, 20, 20, 4096, 128),
    (1, 10, 10, 4096, 128),
    (1, 5,  5,  4096, 128),
    # llama3-8b
    (1, 32, 8,  8192, 128),
    (1, 16, 4,  8192, 128),
    (1, 8,  2,  8192, 128),
    (1, 4,  1,  8192, 128),
    # qwen2-7b
    (1, 28, 4,  4096, 128),
    (1, 14, 2,  4096, 128),
    (1, 7,  1,  4096, 128),
]

@pytest.mark.parametrize('Z, Q_H, K_H, N_CTX, D_HEAD',
                         attention_perf_model_cases_list)
@pytest.mark.parametrize('dtype', ['fp16'])
@pytest.mark.parametrize('causal', [False, True])
def test_op_fwd(Z, Q_H, K_H, N_CTX, D_HEAD, causal, dtype):
    InterleaveManager.reset(ENABLE_FWD_INTERLEAVE)
    torch.manual_seed(20)
    q = torch.empty((Z, Q_H, N_CTX, D_HEAD), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    k = torch.empty((Z, K_H, N_CTX, D_HEAD), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    v = torch.empty((Z, K_H, N_CTX, D_HEAD), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    q_num_per_group = Q_H // K_H
    qq=q
    kk_re=[]
    vv_re=[]
    for i in range(K_H):
        k_c = k[:,i:i+1,:,:]
        m_k_c = k_c.repeat(1, q_num_per_group,1,1)
        kk_re.append(m_k_c)
        v_c = v[:,i:i+1,:,:]
        m_v_c = v_c.repeat(1, q_num_per_group,1,1)
        vv_re.append(m_v_c)
    kk=torch.cat(kk_re,dim=1)
    vv=torch.cat(vv_re,dim=1)

    qq = qq.to(name_to_torch_types[dtype])
    kk = kk.to(name_to_torch_types[dtype])
    sm_scale = D_HEAD**-0.5
    dout = torch.randn_like(qq, dtype=torch.float16)
    # reference implementation
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    p = torch.matmul(qq.half(), kk.transpose(2, 3).half()) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    ref_out = torch.matmul(p, vv)
    # triton implementation
    tk = torch.empty_like(k)
    for i in range(N_CTX):
        j = (i % 4) * 4 + int((i % 16) / 4) + int(i / 16) * 16
        tk.transpose(2, 3)[:, :, :, j] = k.transpose(2, 3)[:, :, :, i]
    k = tk if InterleaveManager.is_enabled() else k
    tri_out = attention(q, k, v, causal, sm_scale, InterleaveManager.is_enabled())
    # compare
    atol = 1.4e-1 if dtype == 'fp8' else 1e-2
    rtol = 1e-2 if dtype == 'fp8' else 0
    #pytest识别的是代码是否执行结束，如果是跳转或者中断就表示执行失败。
    torch.testing.assert_close(ref_out, tri_out, atol=atol, rtol=rtol)
    InterleaveManager.reset(ORIGIN_INTERLEAVE_ENABLED)


@pytest.mark.parametrize('Z, Q_H, K_H, N_CTX, D_HEAD',
                         attention_perf_model_cases_list)
def test_op_bwd(Z, Q_H, K_H, N_CTX, D_HEAD, dtype=torch.float16):
    InterleaveManager.disable()
    torch.manual_seed(20)
    causal = True
    q = (torch.empty((Z, Q_H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, K_H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, K_H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())

    sm_scale = 0.5
    dout = torch.randn_like(q)
    q_num_per_group = Q_H // K_H
    qq=q
    kk_re=[]
    vv_re=[]
    for i in range(K_H):
        k_c = k[:,i:i+1,:,:]
        m_k_c = k_c.repeat(1, q_num_per_group,1,1)
        kk_re.append(m_k_c)
        v_c = v[:,i:i+1,:,:]
        m_v_c = v_c.repeat(1, q_num_per_group,1,1)
        vv_re.append(m_v_c)
    kk=torch.cat(kk_re,dim=1)
    vv=torch.cat(vv_re,dim=1)
    ## reference implementation
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    p = torch.matmul(qq, kk.transpose(2, 3)) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    ref_out = torch.matmul(p, vv)
    ref_out.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None
    # # triton implementation
    tri_out = attention(q, k, v, causal, sm_scale, False)
    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None
    # compare
    torch.testing.assert_close(ref_out, tri_out, atol=1e-2, rtol=0)
    if torch.version.hip is None:
        torch.testing.assert_close(ref_dv, tri_dv, atol=1e-2, rtol=0)
    # The current block size for MI200 series is 64x64. This results in
    # larger differences in float results due to rounding.
    else:
        torch.testing.assert_close(ref_dv, tri_dv, atol=5e-2, rtol=0)
    torch.testing.assert_close(ref_dk, tri_dk, atol=5e-2, rtol=1e-2)
    torch.testing.assert_close(ref_dq, tri_dq, atol=5e-2, rtol=1e-2)
    InterleaveManager.reset(ORIGIN_INTERLEAVE_ENABLED)

HAS_FLASH = False

configs = []
for mode in ['fwd', 'bwd']:
    for causal in [True]:
        if mode == 'bwd' and causal == False:
            continue
        configs.append(triton.testing.Benchmark(
            x_names=['BATCH', 'H', 'K_H', 'N_CTX', 'D_HEAD'],
            x_vals= attention_perf_model_cases_list,
            line_arg='provider',
            line_vals=['triton'] + (['flash'] if HAS_FLASH else []),
            line_names=['Triton'] + ([f'Flash'] if HAS_FLASH else []),
            styles=[('red', '-'), ('blue', '-')],
            ylabel='ms',
            xlabel='hhh',
            plot_name=f'fused-attention-{mode}-causal={causal}',
            args={
                'dtype': torch.float16,
                'mode': mode,
                'causal': causal,
            },
        ))

@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, K_H, N_CTX, D_HEAD, causal, mode, provider, dtype=torch.float16, device="cuda"):
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 10
    # Bwd pass only supports causal=True right now
    if mode == 'bwd':
        causal = True
        InterleaveManager.disable()
    else:
        InterleaveManager.reset(ENABLE_FWD_INTERLEAVE)
    if provider == "triton":
        q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        k = torch.randn((BATCH, K_H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        v = torch.randn((BATCH, K_H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        sm_scale = D_HEAD ** -0.5
        if InterleaveManager.is_enabled():
            tk = torch.empty_like(k)
            for i in range(N_CTX):
                j = (i % 4) * 4 + int((i % 16) / 4) + int(i / 16) * 16
                tk.transpose(2, 3)[:, :, :, j] = k.transpose(2, 3)[:, :, :, i]
            fn = lambda: attention(q, tk, v, causal, sm_scale, True)
        else:
            fn = lambda: attention(q, k, v, causal, sm_scale, False)
        if mode == 'bwd':
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        InterleaveManager.reset(ORIGIN_INTERLEAVE_ENABLED)
    if provider == "flash":
        qkv = torch.randn((BATCH, N_CTX, 3, H, D_HEAD), dtype=dtype, device=device, requires_grad=True)
        fn = lambda: flash_attn_func(qkv, causal=causal)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * D_HEAD
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    return total_flops / ms * 1e-9

# only works on post-Ampere GPUs right now
bench_flash_attention.run(print_data=True)
