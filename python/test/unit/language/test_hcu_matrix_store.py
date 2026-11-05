"""Native matrix-store API coverage: layouts, instruction variants and bounds."""
import pytest
import torch
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.hcu import MmacInterleave2Layout, MmacLayout, matrix_store
import triton


@g.jit
def store_kernel(X, Y, M: gl.constexpr, N: gl.constexpr,
                 WM: gl.constexpr, WN: gl.constexpr, SOURCE: gl.constexpr,
                 MODE: gl.constexpr, TILE: gl.constexpr, VERSION: gl.constexpr,
                 RM: gl.constexpr, RN: gl.constexpr, STRIDE: gl.constexpr,
                 OM: gl.constexpr, ON: gl.constexpr, BC: gl.constexpr,
                 GRID_N: gl.constexpr):
    if SOURCE == 2:
        layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [WM, WN], [1, 0])
    else:
        f: gl.constexpr = gl.amd.AMDMFMALayout(
            version=4, instr_shape=[16, 16, 32], mmac_layout=MODE,
            warps_per_cta=[WM, WN], tiles_per_warp=[4, 4])
        if SOURCE == 1:
            layout: gl.constexpr = MmacInterleave2Layout(f)
        else:
            layout: gl.constexpr = f
    m = gl.arange(0, M, layout=gl.SliceLayout(1, layout))
    n = gl.arange(0, N, layout=gl.SliceLayout(0, layout))
    row = gl.program_id(0) * M
    col = gl.program_id(1) * N
    value = gl.load(X + (row + m[:, None]) * (N * GRID_N) + col + n[None, :])
    matrix_store(Y, value, (RM, RN), (STRIDE, 1), (row + OM, col + ON),
                 tile=TILE, version=VERSION, boundary_check=BC)


def cases():
    result = []
    # Capability table: both B16 tiles, v2/v3, and M/N wave distributions.
    for dtype in ('float16', 'bfloat16'):
        for tile in ((16, 32), (32, 32)):
            for version in (2, 3):
                for m, n, wm, wn in ((64, 256, 1, 4), (128, 128, 2, 2), (256, 64, 4, 1)):
                    result.append((dtype, m, n, wm, wn, 1, 3, tile, version, (), (1, 1), False))
    # Multiple register groups, all boundary combinations, row pitch/origin.
    for bc in ((0,), (1,), (0, 1)):
        for tile in ((16, 32), (32, 32)):
            result.append(('bfloat16', 256, 256, 2, 2, 1, 3, tile, 3, bc, (1, 1), False))
    # Ordinary MMAC modes and blocked values normalize through convert_layout.
    for mode in (1, 2, 3, 4):
        result.append(('float16', 128, 128, 2, 2, 0, mode, (32, 32), 3, (0, 1), (1, 1), False))
    result.append(('bfloat16', 64, 128, 2, 2, 2, 3, (32, 32), 3, (0, 1), (1, 1), False))
    # Integer B16 destinations with distinct regions for every wave.
    for dtype in ('int16', 'uint16'):
        result.append((dtype, 64, 128, 2, 2, 2, 3, (32, 32), 3, (0, 1), (1, 1), False))
    result.append(('float16', 64, 128, 2, 2, 2, 3, (32, 32), 3, (0, 1), (2, 2), False))
    result.append(('bfloat16', 64, 128, 1, 4, 1, 3, (32, 32), 3, (0, 1), (1, 1), True))
    return result


@pytest.mark.parametrize('case', cases())
def test_native_matrix_store(case):
    if not torch.cuda.is_available() or triton.runtime.driver.active.get_current_target().arch != 'gfx946':
        pytest.skip('requires gfx946')
    dtype, m, n, wm, wn, source, mode, tile, version, bc, grid, tiny = case
    td = getattr(torch, dtype)
    rows, cols = m * grid[0], n * grid[1]
    x = torch.arange(rows * cols).reshape(rows, cols).remainder(253).float()
    if td in (torch.float16, torch.bfloat16):
        x = (x - 126) / 8
    om, on = 1, 3
    rm = rows + om - (5 if 0 in bc else 0)
    rn = cols + on - (7 if 1 in bc else 0)
    if tiny:
        rm, rn = 17, 23
    stride = rn + 16
    initial = torch.full((stride * (rows + om + 2) + 128,), 512, dtype=td)
    expected = initial.clone()
    for row in range(max(0, min(rows, rm - om))):
        width = max(0, min(cols, rn - on))
        start = (row + om) * stride + on
        expected[start:start + width] = x[row, :width].to(td)
    out = initial.cuda()
    compiled = store_kernel[grid](
        x.contiguous().cuda(), out, M=m, N=n, WM=wm, WN=wn, SOURCE=source,
        MODE=mode, TILE=tile, VERSION=version, RM=rm, RN=rn, STRIDE=stride,
        OM=om, ON=on, BC=bc, GRID_N=grid[1], num_warps=wm * wn)
    got = out.cpu()
    assert torch.equal(got.view(torch.uint8), expected.view(torch.uint8)), str(case)
    assert 'matrix_store_' in compiled.asm['amdgcn']


@g.jit
def invalid_store_kernel(X, Y, TILE: gl.constexpr, VERSION: gl.constexpr,
                         ORDER: gl.constexpr, INTERLEAVE: gl.constexpr,
                         COLUMN_STRIDE: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [1, 4], [1, 0])
    m = gl.arange(0, 32, layout=gl.SliceLayout(1, layout))
    n = gl.arange(0, 32, layout=gl.SliceLayout(0, layout))
    value = gl.load(X + m[:, None] * 32 + n[None, :])
    matrix_store(Y, value, (32, 32), (32, COLUMN_STRIDE), (0, 0),
                 tile=TILE, version=VERSION, order=ORDER, interleave=INTERLEAVE)


@pytest.mark.parametrize('overrides,message', [
    ({'TILE': (8, 32)}, 'unsupported HCU matrix_store'),
    ({'VERSION': 1}, 'unsupported HCU matrix_store'),
    ({'ORDER': (0, 1)}, 'unsupported HCU matrix_store'),
    ({'INTERLEAVE': 0}, 'unsupported HCU matrix_store'),
    ({'COLUMN_STRIDE': 2}, 'unit column stride'),
    ({'OUTPUT_DTYPE': 'float32'}, 'unsupported HCU matrix_store'),
])
def test_matrix_store_invalid_configuration(overrides, message):
    if not torch.cuda.is_available() or triton.runtime.driver.active.get_current_target().arch != 'gfx946':
        pytest.skip('requires gfx946')
    args = dict(TILE=(32, 32), VERSION=3, ORDER=(1, 0), INTERLEAVE=1, COLUMN_STRIDE=1)
    args.update(overrides)
    dtype = getattr(torch, args.pop('OUTPUT_DTYPE', 'bfloat16'))
    x = torch.zeros(1024).cuda()
    y = torch.zeros(1024, dtype=dtype).cuda()
    with pytest.raises(triton.CompilationError, match=message):
        invalid_store_kernel.warmup(x, y, **args, grid=(1,), num_warps=4)


@g.jit
def dynamic_bounds_kernel(X, Y, rows, cols, stride):
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [2, 2], [1, 0])
    m = gl.arange(0, 64, layout=gl.SliceLayout(1, layout))
    n = gl.arange(0, 128, layout=gl.SliceLayout(0, layout))
    value = gl.load(X + m[:, None] * 128 + n[None, :])
    matrix_store(Y, value, (rows, cols), (stride, 1), (1, 3), boundary_check=(0, 1))


@pytest.mark.parametrize('rows,cols', [(59, 119), (17, 23)])
def test_matrix_store_dynamic_bounds(rows, cols):
    if not torch.cuda.is_available() or triton.runtime.driver.active.get_current_target().arch != 'gfx946':
        pytest.skip('requires gfx946')
    x = torch.arange(64 * 128).remainder(253).float().reshape(64, 128)
    initial = torch.full((68, 160), 512, dtype=torch.bfloat16)
    expected = initial.clone()
    expected[1:rows, 3:cols] = x[:rows - 1, :cols - 3].bfloat16()
    y = initial.cuda()
    dynamic_bounds_kernel[(1,)](x.cuda(), y, rows, cols, 160, num_warps=4)
    assert torch.equal(y.cpu().view(torch.uint8), expected.view(torch.uint8))
