# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Read several MLS tiles per MFMA tile without changing logical elements."""

import pytest
import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.experimental import gluon as g
from triton.experimental.gluon._runtime import GluonASTSource
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna4 import async_copy
from triton.experimental.gluon.language.amd.hcu import (
    MlsSharedLayout,
    MmacInterleave2Layout,
    MmacLayout,
    matrix_load,
)


@g.jit
def _copy(X, Y, OP: gl.constexpr, T: gl.constexpr, W: gl.constexpr, R: gl.constexpr, K: gl.constexpr, P: gl.constexpr,
          OFFSET: gl.constexpr, K_CONTIGUOUS: gl.constexpr):
    bits: gl.constexpr = X.dtype.element_ty.primitive_bitwidth
    tile_k: gl.constexpr = 128 if bits == 8 else 64
    kw: gl.constexpr = 8 if bits == 8 else 4
    wm: gl.constexpr = W if OP == 0 else 4 // W
    wn: gl.constexpr = 4 // W if OP == 0 else W
    f: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, kw * 4], mmac_layout=MmacLayout.LTS,
                                           warps_per_cta=[wm, wn], tiles_per_warp=[T, 4] if OP == 0 else [4, T])
    if OP == 0:
        shared = gl.allocate_shared_memory(
            X.dtype.element_ty, (P, K),
            MlsSharedLayout(0, (64, tile_k), bits=bits, order=(1, 0) if K_CONTIGUOUS else (0, 1)))
        matrix_load(shared, X, (P, K), (K, 1) if K_CONTIGUOUS else (1, P), (0, 0), warps=(wm, wn))
    else:
        shared = gl.allocate_shared_memory(
            X.dtype.element_ty, (K, P),
            MlsSharedLayout(1, (tile_k, 64), bits=bits, alt=1, order=(0, 1) if K_CONTIGUOUS else (1, 0)))
        matrix_load(shared, X, (K, P), (1, K) if K_CONTIGUOUS else (P, 1), (0, 0), warps=(wm, wn))
    async_copy.commit_group()
    async_copy.wait_group(0)
    if OP == 0:
        view = shared.slice(OFFSET, R, dim=0)
        value = async_copy.load_shared_relaxed(view, gl.DotOperandLayout(0, f, kw))
        rows: gl.constexpr = R
        cols: gl.constexpr = K
    else:
        view = shared.slice(OFFSET, R, dim=1)
        value = async_copy.load_shared_relaxed(view, MmacInterleave2Layout(gl.DotOperandLayout(1, f, kw)))
        rows: gl.constexpr = K
        cols: gl.constexpr = R
    plain: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [2, 2], [1, 0])
    value = gl.convert_layout(value, plain).to(Y.dtype.element_ty, bitcast=True)
    r = gl.arange(0, rows, layout=gl.SliceLayout(1, plain))
    c = gl.arange(0, cols, layout=gl.SliceLayout(0, plain))
    # An ordinary store makes the reference independent of matrix-store mapping.
    gl.store(Y + r[:, None] * cols + c[None, :], value)


# T, non-K warps, view extent, K repetitions, parent extent, view origin.
# Cover half/one/two/four MLS tiles per warp, outer repetitions in both
# dimensions, and both halves of a larger parent allocation.
CASES = [
    (2, 2, 64, 1, 64, 0),
    (4, 2, 128, 1, 128, 0),
    (8, 2, 256, 1, 256, 0),
    (8, 1, 128, 1, 128, 0),
    (8, 4, 512, 1, 512, 0),
    (8, 2, 512, 2, 512, 0),
    (16, 2, 512, 1, 512, 0),
    (8, 2, 256, 1, 512, 0),
    (8, 2, 256, 1, 512, 256),
]


@pytest.mark.parametrize('dtype,output_dtype,tile_k', [
    ('fp16', 'i16', 64),
    ('bf16', 'i16', 64),
    ('fp8e4nv', 'u8', 128),
])
@pytest.mark.parametrize('op', [0, 1])
@pytest.mark.parametrize('t,r,k_reps', [(8, 256, 1), (8, 512, 2), (16, 512, 1)])
def test_mls_local_load_tiles_compile(dtype, output_dtype, tile_k, op, t, r, k_reps):
    # The old lowering fails its element-count assertion even without a GPU.
    source = GluonASTSource(_copy, signature={'X': '*' + dtype, 'Y': '*' + output_dtype},
                            constexprs=dict(OP=op, T=t, W=2, R=r, K=tile_k * k_reps, P=r, OFFSET=0,
                                            K_CONTIGUOUS=True), attrs={(i, ): [['tt.divisibility', 16]]
                                                                       for i in range(2)})
    kernel = triton.compile(source, target=GPUTarget('hip', 'gfx946', 64), options={'num_warps': 4})
    assert 'ds_read_matrix' in kernel.asm['amdgcn']


@pytest.mark.parametrize('dtype,k_contiguous', [
    (torch.float16, True),
    (torch.float16, False),
    (torch.bfloat16, True),
    (torch.bfloat16, False),
    (torch.float8_e4m3fn, True),
])
@pytest.mark.parametrize('op', [0, 1])
@pytest.mark.parametrize('t,w,r,k_reps,p,offset', CASES)
def test_mls_local_load_tiles(dtype, op, k_contiguous, t, w, r, k_reps, p, offset):
    if not torch.cuda.is_available():
        pytest.skip('requires HCU')
    if torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] != 'gfx946':
        pytest.skip('requires gfx946 MLS')
    bits = torch.empty((), dtype=dtype).element_size() * 8
    k = (128 if bits == 8 else 64) * k_reps
    shape = (p, k) if op == 0 else (k, p)
    gen = torch.Generator().manual_seed(562)
    storage_dtype = torch.uint8 if bits == 8 else torch.int16
    # Include all bit patterns; compare bits so NaNs and signed zeros survive.
    low, high = (0, 256) if bits == 8 else (-32768, 32768)
    raw = torch.randint(low, high, shape, generator=gen, dtype=storage_dtype)
    expected = raw[offset:offset + r, :] if op == 0 else raw[:, offset:offset + r]
    physical = raw.T.contiguous() if (op == 0) != k_contiguous else raw.contiguous()
    source = physical.view(dtype).cuda()
    output = torch.full((r * k + 64, ), 173, dtype=storage_dtype, device='cuda')
    kernel = _copy[(1, )](source, output[32:-32], OP=op, T=t, W=w, R=r, K=k, P=p, OFFSET=offset,
                          K_CONTIGUOUS=k_contiguous, num_warps=4)
    got = output.cpu()
    torch.testing.assert_close(got[32:-32].reshape(expected.shape), expected, rtol=0, atol=0)
    assert (got[:32] == 173).all() and (got[-32:] == 173).all()
    assert 'ds_read_matrix' in kernel.asm['amdgcn']


@g.jit
def _dot(A, B, C, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr, TM: gl.constexpr, TN: gl.constexpr,
         K_CONTIGUOUS: gl.constexpr):
    f: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 16], mmac_layout=MmacLayout.LTS,
                                           warps_per_cta=[2, 2], tiles_per_warp=[TM, TN])
    result: gl.constexpr = MmacInterleave2Layout(f)
    sa = gl.allocate_shared_memory(A.dtype.element_ty, (M, K), MlsSharedLayout(0, (64, 64), bits=16))
    sb = gl.allocate_shared_memory(
        B.dtype.element_ty, (K, N),
        MlsSharedLayout(1, (64, 64), bits=16, alt=1, order=(0, 1) if K_CONTIGUOUS else (1, 0)))
    matrix_load(sa, A, (M, K), (K, 1), (0, 0), warps=(2, 2))
    matrix_load(sb, B, (K, N), (1, K) if K_CONTIGUOUS else (N, 1), (0, 0), warps=(2, 2))
    async_copy.commit_group()
    async_copy.wait_group(0)
    a = async_copy.load_shared_relaxed(sa, gl.DotOperandLayout(0, f, 4))
    b = async_copy.load_shared_relaxed(sb, MmacInterleave2Layout(gl.DotOperandLayout(1, f, 4)))
    acc = gl.amd.cdna4.mfma(a, b, gl.full((M, N), 0, gl.float32, result))
    plain: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [2, 2], [1, 0])
    acc = gl.convert_layout(acc, plain)
    m = gl.arange(0, M, layout=gl.SliceLayout(1, plain))
    n = gl.arange(0, N, layout=gl.SliceLayout(0, plain))
    gl.store(C + m[:, None] * N + n[None, :], acc)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('k_contiguous', [True, False])
@pytest.mark.parametrize('tm,tn,m,n,k', [
    (4, 8, 128, 256, 64),
    # Exercise outer N and K repetitions separately to keep accumulators and
    # operands within the native register budget of the PMD test target.
    (2, 8, 64, 512, 64),
    (2, 8, 64, 256, 128),
    (8, 2, 256, 64, 64),
    (8, 2, 512, 64, 64),
])
def test_mls_local_load_dot(dtype, k_contiguous, tm, tn, m, n, k):
    if not torch.cuda.is_available():
        pytest.skip('requires HCU')
    if torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] != 'gfx946':
        pytest.skip('requires gfx946 MLS')
    gen = torch.Generator().manual_seed(563)
    a = (torch.randint(-32, 32, (m, k), generator=gen).float() / 32).to(dtype)
    b = (torch.randint(-32, 32, (k, n), generator=gen).float() / 32).to(dtype)
    physical_b = b.T.contiguous() if k_contiguous else b
    out = torch.full((m, n), float('nan'), dtype=torch.float32, device='cuda')
    _dot[(1, )](a.cuda(), physical_b.cuda(), out, M=m, N=n, K=k, TM=tm, TN=tn, K_CONTIGUOUS=k_contiguous, num_warps=4)
    # These products and sums are exact binary fractions in FP32.
    torch.testing.assert_close(out.cpu(), a.float() @ b.float(), rtol=0, atol=0)
