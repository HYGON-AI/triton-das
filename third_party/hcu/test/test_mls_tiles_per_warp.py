"""tilesPerWarp must not repeat an MLS tile that warps already cover.

A reduction pins every warp on M. gfx946/gfx948 then fold the whole 64-row
FP8 MLS tile into tilesPerWarp, so each warp recomputes every row. The M
factor is shrunk when that product exceeds the dot's M extent.
"""

import re

import pytest
import triton
import triton.language as tl
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.backends.hcu.compiler import HIPBackend
from triton.compiler import ASTSource


@triton.jit
def _mls_k_dot(k_ptr, q_ptr, out_ptr, REDUCE: tl.constexpr, M: tl.constexpr,
               N: tl.constexpr, KD: tl.constexpr):
    k = tl.matrix_load(
        k_ptr, shape=[M, KD], strides=[KD, 1],
        block_shape=[M, KD], offsets=[0, 0],
    )
    offs_k = tl.arange(0, KD)
    offs_n = tl.arange(0, N)
    q = tl.load(q_ptr + offs_k[:, None] + offs_n[None, :] * KD)
    acc = tl.dot(k, q)
    if REDUCE:
        reduced = tl.sum(acc, axis=1)
        tl.store(out_ptr + tl.arange(0, M), reduced)
    else:
        tl.store(out_ptr + tl.arange(0, M)[:, None] * N + offs_n[None, :], acc)


def _compile_ttgir(arch, reduce, m, n=64, kd=128, num_warps=4):
    backend = HIPBackend(GPUTarget("hip", arch, 64))
    options = backend.parse_options({"num_warps": num_warps, "num_stages": 1})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    source = ASTSource(
        _mls_k_dot,
        signature={"k_ptr": "*fp8e4nv", "q_ptr": "*fp8e4nv", "out_ptr": "*fp32"},
        constexprs={"REDUCE": reduce, "M": m, "N": n, "KD": kd},
        attrs={(i,): [["tt.divisibility", 16]] for i in range(3)},
    )
    module = source.make_ir(
        options, backend.get_codegen_implementation(options),
        backend.get_module_map(), context)
    metadata = {}
    module = backend.make_ttir(module, metadata, options)
    return str(backend.make_ttgir(module, metadata, options))


def _mfma_layouts(ttgir):
    # The printer drops tilesPerWarp when it is the default [1, 1].
    layouts = []
    for match in re.finditer(r"#ttg\.\w+_mfma<\{([^}]*)\}>", ttgir):
        body = match.group(1)
        warps = re.search(r"warpsPerCTA = \[(\d+), (\d+)\]", body)
        if not warps:
            continue
        tiles = re.search(r"tilesPerWarp = \[(\d+), (\d+)\]", body)
        tile = (int(tiles.group(1)), int(tiles.group(2))) if tiles else (1, 1)
        layouts.append((int(warps.group(1)), int(warps.group(2)), *tile))
    return layouts


@pytest.mark.parametrize("arch,m,reduce,warps,tiles", [
    # 64 rows, 4 warps, 64-row MLS tile: shrink 4 -> 1.
    # gfx948 shares this MLS version-3 tile table; gfx946 is the arch that
    # lowers the dot in this compile-only check.
    ("gfx946", 64, True, (4, 1), (1, 1)),
    # 256 rows: 4 * 4 * 16 does not exceed M, so the tile factor stays.
    ("gfx946", 256, True, (4, 1), (4, 1)),
    # No reduction: warps split N, and the M tile factor is left alone.
    ("gfx946", 64, False, (1, 4), (4, 1)),
    # gfx938's FP8 MLS tile is already 16 rows, so tilesPerWarp[M] is 1.
    ("gfx938", 64, True, (4, 1), (1, 1)),
])
def test_mls_tiles_per_warp_when_warps_cover_m(arch, m, reduce, warps, tiles):
    ttgir = _compile_ttgir(arch, reduce, m)
    layouts = set(_mfma_layouts(ttgir))
    assert (*warps, *tiles) in layouts
    if reduce and m == 64 and arch == "gfx946":
        assert (4, 1, 4, 1) not in layouts


@triton.jit
def _fp16_col_a_gemm(a_ptr, b_ptr, c_ptr, BLOCK_N: tl.constexpr):
    # M is the contiguous dim. gfx938 fp16 MLS tiles are 32 or 64 rows, so a
    # 32-row load is one instruction and tilesPerWarp[M] stays 2.
    a = tl.matrix_load(
        a_ptr, shape=[32, 64], strides=[1, 32],
        block_shape=[32, 64], offsets=[0, 0],
    )
    offs_k = tl.arange(0, 64)
    offs_n = tl.arange(0, BLOCK_N)
    b = tl.load(b_ptr + offs_k[:, None] * BLOCK_N + offs_n[None, :])
    acc = tl.dot(a, b)
    tl.store(c_ptr + tl.arange(0, 32)[:, None] * BLOCK_N + offs_n[None, :], acc)


@pytest.mark.parametrize("mmac_layout_force,block_n", [(1, 32), (1, 64), (4, 32), (4, 64)])
def test_fp16_col_a_keeps_32row_mls_tile(mmac_layout_force, block_n):
    backend = HIPBackend(GPUTarget("hip", "gfx938", 64))
    options = backend.parse_options({
        "num_warps": 4,
        "num_stages": 1,
        "mmac_layout_force": mmac_layout_force,
    })
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    source = ASTSource(
        _fp16_col_a_gemm,
        signature={"a_ptr": "*fp16", "b_ptr": "*fp16", "c_ptr": "*fp16"},
        constexprs={"BLOCK_N": block_n},
        attrs={(i,): [["tt.divisibility", 16]] for i in range(3)},
    )
    module = source.make_ir(
        options, backend.get_codegen_implementation(options),
        backend.get_module_map(), context)
    metadata = {}
    module = backend.make_ttir(module, metadata, options)
    ttgir = str(backend.make_ttgir(module, metadata, options))
    layouts = set(_mfma_layouts(ttgir))
    assert any(layout[2:] == (2, 1) for layout in layouts), layouts
    assert not any(layout[2:] == (1, 1) and layout[0] > 1 for layout in layouts), layouts
