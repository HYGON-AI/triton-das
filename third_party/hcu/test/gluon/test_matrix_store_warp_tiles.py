# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Matrix-store addresses must follow the explicit accumulator layout."""

import pytest
import torch
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language._core import builtin, _unwrap_if_constexpr
from triton.experimental.gluon.language.amd.hcu import MmacInterleave2Layout, MmacLayout


@builtin
def _store_native(base, value, shape, stride, tile, boundary, _semantic=None):
    # Keep the tested tiles_per_warp: public matrix_store normalizes it to the
    # whole per-wave tensor extent, hiding outer MFMA tile repetitions.
    def constants(values, dtype):
        return [_semantic.cast(_semantic.to_tensor(v), dtype).handle for v in values]

    _semantic.builder.create_hcu_mls_store_from_registers(base.handle, value.handle,
                                                          constants(_unwrap_if_constexpr(shape), gl.int32),
                                                          constants((_unwrap_if_constexpr(stride), 1), gl.int64),
                                                          constants((1, 8), gl.int32), list(_unwrap_if_constexpr(tile)),
                                                          1, 3, [1, 0], list(_unwrap_if_constexpr(boundary)))


@g.jit
def _copy(X, Y, M: gl.constexpr, N: gl.constexpr, TM: gl.constexpr, TN: gl.constexpr, WN: gl.constexpr,
          TILE: gl.constexpr, BOUNDARY: gl.constexpr, VALID_M: gl.constexpr, VALID_N: gl.constexpr):
    native: gl.constexpr = MmacInterleave2Layout(
        gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], mmac_layout=MmacLayout.LTS,
                             warps_per_cta=[4 // WN, WN], tiles_per_warp=[TM, TN]))
    m = gl.arange(0, M, layout=gl.SliceLayout(1, native))
    n = gl.arange(0, N, layout=gl.SliceLayout(0, native))
    value = gl.load(X + m[:, None] * N + n[None, :])
    _store_native(Y, value, (1 + VALID_M, 8 + VALID_N), N + 16, TILE, BOUNDARY)


# TM, TN, WN, M, N. N spans 32/64/128/256 columns per warp; several
# cases have additional register repetitions outside the warp's MFMA tile.
CASES = [
    (4, 2, 4, 64, 128),
    (4, 4, 2, 128, 128),
    (4, 8, 2, 128, 256),
    (4, 8, 4, 64, 512),
    (4, 8, 1, 256, 128),
    (4, 8, 2, 128, 512),
    (4, 4, 2, 256, 256),
    (2, 8, 2, 128, 512),
    (4, 16, 2, 128, 512),
]


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('tile', [(16, 32), (32, 32)])
@pytest.mark.parametrize('boundary', [(), (0, ), (1, ), (0, 1)])
@pytest.mark.parametrize('tm,tn,wn,m,n', CASES)
def test_matrix_store_warp_tiles(dtype, tile, boundary, tm, tn, wn, m, n):
    if not torch.cuda.is_available():
        pytest.skip('requires HCU')
    if torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] != 'gfx946':
        pytest.skip('requires gfx946 native matrix store')
    gen = torch.Generator().manual_seed(560)
    # Exactly representable in both types; distinguish every tile boundary
    # without introducing a downcast or a matrix-load on the producing side.
    x = (torch.randint(-128, 128, (m, n), generator=gen).float() / 8).to(dtype)
    expected = torch.full((m + 2, n + 16), 512, dtype=dtype)
    output = expected.cuda()
    valid_m = m - (7 if 0 in boundary else 0)
    valid_n = n - (11 if 1 in boundary else 0)
    expected[1:1 + valid_m, 8:8 + valid_n] = x[:valid_m, :valid_n]
    kernel = _copy[(1, )](x.cuda(), output, M=m, N=n, TM=tm, TN=tn, WN=wn, TILE=tile, BOUNDARY=boundary,
                          VALID_M=valid_m, VALID_N=valid_n, num_warps=4)
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    assert f'tilesPerWarp = [{tm}, {tn}]' in kernel.asm['ttgir']
    assert 'matrix_store_' in kernel.asm['amdgcn']
