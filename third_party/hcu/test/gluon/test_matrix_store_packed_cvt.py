# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Packed downcasts must preserve matrix-store and other users' values."""
import pytest
import torch
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.hcu import (
    MmacLayout, MmacInterleave2Layout, matrix_store,
)


@g.jit
def _downcast(X, Y, Other, N: gl.constexpr, STORE: gl.constexpr,
              RTZ: gl.constexpr, TILE: gl.constexpr):
    blocked: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
    m = gl.arange(0, 128, layout=gl.SliceLayout(1, blocked))
    n = gl.arange(0, N, layout=gl.SliceLayout(0, blocked))
    x = gl.load(X + m[:, None] * N + n[None, :])
    native: gl.constexpr = MmacInterleave2Layout(gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], mmac_layout=MmacLayout.LTS,
        warps_per_cta=[4, 1], tiles_per_warp=[2, N // 16], element_bitwidth=32))
    x = gl.convert_layout(x, native)
    if RTZ:
        y = x.to(Y.dtype.element_ty, fp_downcast_rounding="rtz")
    else:
        y = x.to(Y.dtype.element_ty)
    if STORE:
        matrix_store(Y, y, (128, N), (N, 1), (0, 0), tile=TILE)
    else:
        plain = gl.convert_layout(y, blocked)
        gl.store(Y + m[:, None] * N + n[None, :], plain)
    # A second user of the same truncf result must retain logical element order.
    other = gl.convert_layout(y, blocked)
    gl.store(Other + m[:, None] * N + n[None, :], other)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('n', [64, 128])
@pytest.mark.parametrize('store,rtz,tile', [
    (False, False, (32, 32)),
    (True, False, (16, 32)), (True, False, (32, 32)),
    (True, True, (16, 32)), (True, True, (32, 32)),
])
def test_matrix_store_packed_cvt(dtype, n, store, rtz, tile, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip('requires HCU')
    if torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] != 'gfx946':
        pytest.skip('requires gfx946 native matrix store')
    gen = torch.Generator().manual_seed(561)
    x = torch.randn((128, n), generator=gen, dtype=torch.float32) * 100
    # Signed zero, ties on both sides of an even mantissa, finite overflow,
    # FP16 subnormals, infinities and NaNs. Rotate across all register positions.
    edge = torch.tensor([0., -0., 1., -1., 1.00048828125, 1.00146484375,
                         1.00390625, 1.01171875, 65504., 65520., -65520.,
                         2.**-24, -2.**-24, 2.**-25, 2.**-14,
                         float('inf'), -float('inf'), float('nan')])
    for i in range(0, x.numel(), 97):
        length = min(edge.numel(), x.numel() - i)
        x.view(-1)[i:i + length] = edge[:length]
    expected = x.to(dtype)
    if rtz:
        # Correct RTNE results that rounded away from zero by one destination
        # ULP; this also turns finite overflow from infinity into max-finite.
        toward_zero = torch.nextafter(expected, torch.zeros_like(expected))
        expected = torch.where(expected.float().abs() > x.abs(),
                               toward_zero, expected)
    y = torch.empty((128, n), dtype=dtype, device='cuda')
    other = torch.empty_like(y)
    compiled = _downcast[(1,)](x.cuda(), y, other, N=n, STORE=store, RTZ=rtz, TILE=tile,
                               num_warps=4, num_stages=1)
    if rtz and dtype == torch.float16:
        # Check that matrix-store fusion leaves explicit RTZ on the ordinary
        # conversion path. This is a preservation check, not a test of that
        # path's rounding implementation (which differs from CPU RTZ on PMD).
        expected = other.cpu()
    for actual in [y.cpu(), other.cpu()]:
        torch.testing.assert_close(torch.isnan(actual), torch.isnan(expected))
        valid = ~torch.isnan(expected)
        # Bitwise comparison also checks signed zero and exact rounding.
        torch.testing.assert_close(actual.view(torch.int16)[valid],
                                   expected.view(torch.int16)[valid], rtol=0, atol=0)
    asm = compiled.asm['amdgcn']
    if store:
        instruction = '32x16' if tile == (16, 32) else '32x32'
        assert 'matrix_store_' + instruction + '_b16' in asm
    if store and not rtz:
        assert ('v_cvt_pkrtz_f16_f32' in asm or 'v_cvt_pk_f16_f32' in asm
                or 'v_cvt_pk_bf16_f32' in asm)
    (tmp_path / 'kernel.amdgcn').write_text(asm)
