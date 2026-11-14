# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Round-trip MLS fragments when waves share a global MLS tile."""

import pytest
import torch
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna4 import async_copy
from triton.experimental.gluon.language.amd.hcu import (
    MmacLayout, MlsSharedLayout, MmacInterleave2Layout, matrix_load, get_wave_id,
)


@g.jit
def _copy_mls_fragments(A, B, OA, OB, M: gl.constexpr, N: gl.constexpr,
                        W: gl.constexpr, WM: gl.constexpr, SINGLE_LOADER: gl.constexpr = False):
    f: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], mmac_layout=MmacLayout.LTS,
        warps_per_cta=[WM, W // WM],
        tiles_per_warp=[M // (16 * WM), N // (16 * (W // WM))],
    )
    sa = gl.allocate_shared_memory(
        gl.float8e4nv, (M, 128), MlsSharedLayout(0, (64, 128)))
    sb = gl.allocate_shared_memory(
        gl.float8e4nv, (128, N),
        MlsSharedLayout(1, (128, 64), alt=1, order=(0, 1)))
    if SINGLE_LOADER:
        pred = get_wave_id() == 0
        matrix_load(sa, A, (M, 128), (128, 1), (0, 0),
                    warps=(WM, W // WM), pred=pred)
        matrix_load(sb, B, (128, N), (1, 128), (0, 0),
                    warps=(WM, W // WM), pred=pred)
    else:
        matrix_load(sa, A, (M, 128), (128, 1), (0, 0), warps=(WM, W // WM))
        matrix_load(sb, B, (128, N), (1, 128), (0, 0), warps=(WM, W // WM))
    async_copy.commit_group()
    async_copy.wait_group(0)
    a = async_copy.load_shared_relaxed(sa, gl.DotOperandLayout(0, f, 8))
    b = async_copy.load_shared_relaxed(
        sb, MmacInterleave2Layout(gl.DotOperandLayout(1, f, 8)))
    la: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [WM, W // WM], [1, 0])
    lb: gl.constexpr = gl.BlockedLayout([8, 1], [16, 4], [WM, W // WM], [0, 1])
    av = gl.convert_layout(a, la).to(gl.uint8, bitcast=True)
    bv = gl.convert_layout(b, lb).to(gl.uint8, bitcast=True)
    m = gl.arange(0, M, layout=gl.SliceLayout(1, la))
    ka = gl.arange(0, 128, layout=gl.SliceLayout(0, la))
    kb = gl.arange(0, 128, layout=gl.SliceLayout(1, lb))
    n = gl.arange(0, N, layout=gl.SliceLayout(0, lb))
    gl.store(OA + m[:, None] * 128 + ka[None, :], av)
    gl.store(OB + kb[:, None] * N + n[None, :], bv)


@pytest.mark.parametrize("m,n,waves,wm,single_loader", [
    (64, 64, 1, 1, False), (64, 64, 2, 2, False), (64, 64, 4, 4, False),
    (64, 64, 2, 1, False), (64, 64, 4, 2, False),
    (128, 64, 4, 4, False), (128, 128, 4, 2, False), (64, 64, 4, 4, True),
])
def test_mls_fragments_shared_by_waves(m, n, waves, wm, single_loader):
    if not torch.cuda.is_available():
        pytest.skip("requires an HCU device")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch != "gfx946":
        pytest.skip("requires gfx946 version-3 MLS encodings")
    gen = torch.Generator().manual_seed(317)
    a = torch.randn((m, 128), generator=gen).to(torch.float8_e4m3fn)
    b = torch.randn((n, 128), generator=gen).to(torch.float8_e4m3fn)
    actual_a = torch.full((m, 128), 255, dtype=torch.uint8, device="cuda")
    actual_b = torch.full((128, n), 255, dtype=torch.uint8, device="cuda")
    _copy_mls_fragments[(1,)](
        a.cuda(), b.cuda(), actual_a, actual_b,
        M=m, N=n, W=waves, WM=wm, SINGLE_LOADER=single_loader,
        num_warps=waves, num_stages=1)
    torch.testing.assert_close(actual_a.cpu(), a.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(
        actual_b.cpu(), b.T.contiguous().view(torch.uint8), rtol=0, atol=0)


@g.jit
def _mls_dot_loop(A, B, C, W: gl.constexpr, WM: gl.constexpr):
    f: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], mmac_layout=MmacLayout.LTS,
        warps_per_cta=[WM, W // WM],
        tiles_per_warp=[4 // WM, 4 // (W // WM)],
    )
    result_layout: gl.constexpr = MmacInterleave2Layout(f)
    sa = gl.allocate_shared_memory(
        gl.float8e4nv, (64, 128), MlsSharedLayout(0, (64, 128)))
    sb = gl.allocate_shared_memory(
        gl.float8e4nv, (128, 64),
        MlsSharedLayout(1, (128, 64), alt=1, order=(0, 1)))
    acc = gl.full((64, 64), 0, gl.float32, result_layout)
    for k in range(0, 512, 128):
        matrix_load(sa, A, (64, 512), (512, 1), (0, k), warps=(WM, W // WM))
        matrix_load(sb, B, (512, 64), (1, 512), (k, 0), warps=(WM, W // WM))
        async_copy.commit_group()
        async_copy.wait_group(0)
        a = async_copy.load_shared_relaxed(sa, gl.DotOperandLayout(0, f, 8))
        b = async_copy.load_shared_relaxed(
            sb, MmacInterleave2Layout(gl.DotOperandLayout(1, f, 8)))
        acc = gl.amd.cdna4.mfma(a, b, acc)
        gl.thread_barrier()
    m = gl.arange(0, 64, layout=gl.SliceLayout(1, result_layout))
    n = gl.arange(0, 64, layout=gl.SliceLayout(0, result_layout))
    gl.store(C + m[:, None] * 64 + n[None, :], acc)


@pytest.mark.parametrize("waves,wm", [(2, 2), (4, 4), (4, 2)])
def test_mls_cooperative_wait_in_loop(waves, wm):
    if not torch.cuda.is_available():
        pytest.skip("requires an HCU device")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx946":
        pytest.skip("requires gfx946 version-3 MLS encodings")
    gen = torch.Generator().manual_seed(123)
    a = torch.randn((64, 512), generator=gen).to(torch.float8_e4m3fn)
    b = torch.randn((64, 512), generator=gen).to(torch.float8_e4m3fn)
    reference = a.float() @ b.float().T
    aa, bb = a.cuda(), b.cuda()
    out = torch.empty((64, 64), dtype=torch.float32, device="cuda")
    for _ in range(2):
        _mls_dot_loop[(1,)](aa, bb, out, W=waves, WM=wm,
                            num_warps=waves, num_stages=1)
        torch.testing.assert_close(out.cpu(), reference, rtol=1e-4, atol=1e-3)


@g.jit
def _copy_mls_subview(X, Y, P: gl.constexpr, V: gl.constexpr,
                      OFFSET: gl.constexpr, W: gl.constexpr, OP: gl.constexpr):
    if OP == 0:
        f: gl.constexpr = gl.amd.AMDMFMALayout(
            version=4, instr_shape=[16, 16, 32], mmac_layout=MmacLayout.LTS,
            warps_per_cta=[W, 1], tiles_per_warp=[V // (16 * W), 4])
        shared = gl.allocate_shared_memory(
            gl.float8e4nv, (P, 128), MlsSharedLayout(0, (64, 128)))
        matrix_load(shared, X, (P, 128), (128, 1), (0, 0), warps=(W, 1))
    else:
        f: gl.constexpr = gl.amd.AMDMFMALayout(
            version=4, instr_shape=[16, 16, 32], mmac_layout=MmacLayout.LTS,
            warps_per_cta=[1, W], tiles_per_warp=[4, V // (16 * W)])
        shared = gl.allocate_shared_memory(
            gl.float8e4nv, (128, P),
            MlsSharedLayout(1, (128, 64), alt=1, order=(0, 1)))
        matrix_load(shared, X, (128, P), (1, 128), (0, 0), warps=(1, W))
    async_copy.commit_group()
    async_copy.wait_group(0)
    if OP == 0:
        value = async_copy.load_shared_relaxed(
            shared.slice(OFFSET, V, dim=0), gl.DotOperandLayout(0, f, 8))
        layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [W, 1], [1, 0])
        value = gl.convert_layout(value, layout).to(gl.uint8, bitcast=True)
        r = gl.arange(0, V, layout=gl.SliceLayout(1, layout))
        k = gl.arange(0, 128, layout=gl.SliceLayout(0, layout))
        gl.store(Y + r[:, None] * 128 + k[None, :], value)
    else:
        value = async_copy.load_shared_relaxed(
            shared.slice(OFFSET, V, dim=1),
            MmacInterleave2Layout(gl.DotOperandLayout(1, f, 8)))
        layout: gl.constexpr = gl.BlockedLayout([8, 1], [16, 4], [1, W], [0, 1])
        value = gl.convert_layout(value, layout).to(gl.uint8, bitcast=True)
        k = gl.arange(0, 128, layout=gl.SliceLayout(1, layout))
        r = gl.arange(0, V, layout=gl.SliceLayout(0, layout))
        gl.store(Y + k[:, None] * V + r[None, :], value)


@pytest.mark.parametrize('op,p,v,offset,waves', [
    (0, 64, 64, 0, 2), (0, 128, 64, 0, 2), (0, 128, 64, 64, 2),
    (0, 128, 64, 0, 4), (0, 128, 64, 64, 4),
    (0, 128, 32, 32, 2),
    (0, 256, 64, 64, 2), (0, 256, 64, 192, 2),
    (1, 128, 64, 0, 2), (1, 128, 64, 64, 2),
    (1, 256, 64, 64, 2), (1, 256, 64, 192, 2),
])
def test_mls_subview_fragments(op, p, v, offset, waves):
    if not torch.cuda.is_available():
        pytest.skip('requires an HCU device')
    if torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] != 'gfx946':
        pytest.skip('requires gfx946 version-3 MLS encodings')
    gen = torch.Generator().manual_seed(559)
    raw = torch.randint(0, 256, (p, 128), generator=gen, dtype=torch.uint8)
    source = raw.view(torch.float8_e4m3fn).cuda()
    shape = (v, 128) if op == 0 else (128, v)
    output = torch.full((v * 128 + 64,), 173, dtype=torch.uint8, device='cuda')
    _copy_mls_subview[(1,)](source, output[32:-32], P=p, V=v, OFFSET=offset,
                           W=waves, OP=op, num_warps=waves, num_stages=1)
    got = output.cpu()
    expected = raw[offset:offset + v]
    if op == 1:
        expected = expected.T.contiguous()
    torch.testing.assert_close(got[32:-32].reshape(shape), expected, rtol=0, atol=0)
    assert (got[:32] == 173).all() and (got[-32:] == 173).all()
