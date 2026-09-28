"""Explicit WASP partition tuples; run only in zx-perf-node60.

All runtime inputs and reference results are computed on the CPU.
"""
import re

import pytest
import torch
import triton
import triton.language as tl

from triton.backends.compiler import GPUTarget
libtriton = pytest.importorskip("triton._C.libtriton")
if not hasattr(libtriton, "hcu"):
    pytest.skip("HCU compiler support is not built", allow_module_level=True)

from triton.backends.hcu import wasp

from matmul_mls_wasp import matmul_mls_wasp_kernel
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon._runtime import GluonASTSource


@pytest.fixture(scope="module", autouse=True)
def require_gfx946():
    # Query the active device at test setup, never at module import.
    if not torch.cuda.is_available():
        pytest.skip("WASP runtime tests require a gfx946 GPU")
    target = triton.runtime.driver.active.get_current_target()
    if target.backend not in ("hcu", "hip") or target.arch != "gfx946":
        pytest.skip("WASP runtime tests require an HCU gfx946 target")


def gemm_data(m, n, k, *, transposed_b=False, dtype=torch.float16):
    generator = torch.Generator().manual_seed(20260927)
    a = torch.randn((m, k), generator=generator, dtype=torch.float16, device="cpu")
    b = torch.randn((n, k) if transposed_b else (k, n),
                    generator=generator, dtype=torch.float16, device="cpu")
    if transposed_b:
        b = b.T
    golden = a.float() @ b.float()
    c = torch.full((m, n), float("nan"), dtype=dtype, device="cpu")
    return a.cuda(), b.cuda(), c.cuda(), golden


@gluon.jit
def _add_part(X, Y, Z, GROUP: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    i = GROUP * 512 + gl.arange(0, 512, layout=layout)
    gl.store(Z + i, gl.load(X + i) + gl.load(Y + i))


@gluon.jit
def _empty():
    pass


@gluon.jit
def _partitioned_add(X, Y, Z, WAVES: gl.constexpr):
    if len(WAVES) == 2:
        gl.warp_specialize([(_empty, ()), (_add_part, (X, Y, Z, 0)),
                            (_add_part, (X, Y, Z, 1))], WAVES, [24, 24])
    elif len(WAVES) == 3:
        gl.warp_specialize([(_empty, ()), (_add_part, (X, Y, Z, 0)),
                            (_add_part, (X, Y, Z, 1)),
                            (_add_part, (X, Y, Z, 2))], WAVES, [24, 24, 24])
    else:
        gl.warp_specialize([(_empty, ()), (_add_part, (X, Y, Z, 0)),
                            (_add_part, (X, Y, Z, 1)),
                            (_add_part, (X, Y, Z, 2)),
                            (_add_part, (X, Y, Z, 3))], WAVES, [24, 24, 24, 24])


@pytest.mark.parametrize("waves", [
    (2, 2), (4, 4), (2, 2, 2), (4, 4, 4), (2, 2, 2, 2), (4, 4, 4, 4),
    (1, 1), (1, 4, 4), (1, 8), (2, 2, 4, 4),
])
def test_gluon_partition_waves(waves):
    gen = torch.Generator().manual_seed(41)
    x = torch.randn((len(waves) * 512,), generator=gen, device="cpu")
    y = torch.randn(x.shape, generator=gen, device="cpu")
    golden = x + y
    z = torch.full_like(x, float("nan")).cuda()
    kernel = _partitioned_add[(1,)](x.cuda(), y.cuda(), z, waves, num_warps=waves[0],
                                num_stages=2, wasp_wdra=False,
                                wasp_partition_warps=waves)
    torch.testing.assert_close(z.cpu(), golden)
    assert kernel.metadata.num_warps == sum(waves)


@pytest.mark.parametrize("waves,requested", [
    ((2, 2, 2), (4, 4, 4)), ((2, 2, 2, 2), (4, 4, 4, 4)),
])
def test_gluon_partition_sizes_must_match(capfd, waves, requested):
    # Equal partition counts are insufficient: each logical
    # partition must match, without rewriting Gluon's explicit layouts.
    source = GluonASTSource(
        _partitioned_add, signature={"X": "*fp32", "Y": "*fp32", "Z": "*fp32",
                                 "WAVES": "constexpr"},
        constexprs={"WAVES": waves})
    with pytest.raises(RuntimeError):
        triton.compile(source, target=GPUTarget("hcu", "gfx946", 64),
                       options=dict(num_warps=4, num_stages=2, wasp_wdra=False,
                                    wasp_partition_warps=requested))
    assert "do not match wasp_partition_warps" in capfd.readouterr().err


@triton.jit
def _load_gemm(A, B, C, K: tl.constexpr, TILE: tl.constexpr):
    rows = tl.arange(0, TILE)
    cols = tl.arange(0, TILE)
    ks = tl.arange(0, 32)
    acc = tl.full((TILE, TILE), 0, tl.float32)
    for base in tl.range(0, K, 32, warp_specialize=True):
        a = tl.load(A + rows[:, None] * K + base + ks[None, :])
        b = tl.load(B + (base + ks[:, None]) * TILE + cols[None, :])
        acc = tl.dot(a, b, acc)
    tl.store(C + rows[:, None] * TILE + cols[None, :], acc)


@pytest.mark.parametrize("waves_per_partition", [2, 4])
@pytest.mark.parametrize("tile", [16, 64])
@pytest.mark.parametrize("num_warps", [2, 4, 8])
def test_load_wave_budget(waves_per_partition, tile, num_warps):
    a, b, c, golden = gemm_data(tile, tile, 128, dtype=torch.float32)
    waves = (waves_per_partition,) * (2 if tile == 16 else 3)
    kernel = _load_gemm[(1,)](a, b, c, 128, tile, num_warps=num_warps,
                              num_stages=2, wasp_wdra=False,
                              wasp_partition_warps=waves)
    torch.testing.assert_close(c.cpu(), golden, atol=0.001, rtol=0.001)
    assert f"hcu.wasp_partition_warps = array<i32: {', '.join(map(str, waves))}>" in kernel.asm["ttgir"]
    assert "hcu.wasp_topo" not in kernel.asm["ttgir"]
    assert kernel.metadata.num_warps == sum(waves)


@pytest.mark.parametrize("waves_per_partition", [2, 4])
@pytest.mark.parametrize("stages", [2, 4])
@pytest.mark.parametrize("num_warps", [2, 4, 8])
def test_mls_wave_budget(tmp_path, waves_per_partition, stages, num_warps):
    # Multiple stages and K iterations exercise EMPTY reuse, not just its
    # initialization. The initial layout size need not match partition size.
    waves = (waves_per_partition,) * 3
    m, n, k = 64, 64, 256
    ag, bg, cg, golden = gemm_data(m, n, k, transposed_b=True)
    kernel = matmul_mls_wasp_kernel[(1,)](
        ag, bg, cg, m, n, k, *ag.stride(), *bg.stride(), *cg.stride(),
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64,
        GROUP_SIZE_M=1, ACTIVATION="", num_warps=num_warps,
        num_stages=stages, wasp_wdra=False,
        wasp_partition_warps=waves)
    torch.testing.assert_close(cg.cpu().float(), golden, atol=0.04, rtol=0.01)
    assert kernel.metadata.num_warps == sum(waves)
    ttgir = kernel.asm["ttgir"]
    assert f"hcu.wasp_partition_warps = array<i32: {', '.join(map(str, waves))}>" in ttgir
    assert "hcu.wasp_topo" not in ttgir
    sizes = [int(n) for n in re.findall(r"partition\d+\([^\n]*num_warps\((\d+)\)", ttgir)]
    assert sizes == list(waves)
    starts = [i * waves_per_partition for i in range(len(waves))]
    assert "warpGroupStartIds = array<i32: " + ", ".join(map(str, starts)) + ">" in ttgir
    assert "@llvm.hcu.wdra.init" not in kernel.asm["llir"]
    for stage in ("ttgir", "llir", "amdgcn"):
        (tmp_path / f"gemm.{stage}").write_text(kernel.asm[stage])


@pytest.mark.parametrize("mls", [False, True])
@pytest.mark.parametrize("groups,stages,tile", [
    (2, 2, 64), (2, 2, 128), (3, 2, 64), (3, 2, 128),
    # Keep larger LDS footprints (four stages / two buffer pairs) at tile=64.
    (3, 4, 64), (4, 2, 64),
])
def test_automatic_gemm_producer_regs(tmp_path, mls, tile, groups, stages):
    from test_wasp_mls_regression import gemm_load_kernel, gemm_mls_kernel

    k = 512
    ag, bg, cg, golden = gemm_data(tile, tile, k, transposed_b=True)
    fn = gemm_mls_kernel if mls else gemm_load_kernel
    kernel = fn[(1,)](
        ag, bg, cg, tile, tile, k, *ag.stride(), *bg.stride(), *cg.stride(),
        BLOCK_SIZE_M=tile, BLOCK_SIZE_N=tile, BLOCK_SIZE_K=64,
        GROUP_SIZE_M=1, WARP_SPECIALIZE=True, num_warps=4,
        num_stages=stages, wasp_wdra=True, wasp_partition_warps=(4,) * groups)
    torch.testing.assert_close(cg.cpu(), golden.half(), atol=0.01, rtol=0.01)
    init = re.search(r'@llvm.hcu.wdra.init\(([^)]*)\)', kernel.asm["llir"])
    regs = tuple(map(int, re.findall(r'i16 (\d+)', init[1])))[:groups]
    producers = 2 if groups == 4 else 1
    # 2P2C slices A; each pair still loads its own full B tile.
    a_elems = (tile // producers) * 64 // 256
    b_elems = 64 * tile // 256
    expected = 24 if mls else 24 + (a_elems + b_elems) * 3 // 2
    assert regs[:producers] == (expected,) * producers
    assert wasp.normalize_regs(regs, "gfx946") == regs
    # WDRA compilation itself rejects spills; retain artifacts for review.
    for stage in ("ttgir", "llir", "amdgcn"):
        (tmp_path / f"automatic.{stage}").write_text(kernel.asm[stage])


@pytest.mark.parametrize("waves", [(2, 2, 2, 2), (4, 4, 4, 4)])
@pytest.mark.parametrize("mls", [False, True])
@pytest.mark.parametrize("k", [512, 4096])
def test_two_producer_private_buffers(tmp_path, waves, mls, k):
    # Independent EMPTY/READY pairs must not overwrite a shared B buffer.
    # Use varying data and several buffer-reuse rounds: all-ones inputs or a
    # single K tile would hide this race.
    m = n = 64
    ag, bg, cg, golden = gemm_data(m, n, k, transposed_b=mls)
    options = dict(num_warps=4, num_stages=2, wasp_wdra=False,
                   wasp_partition_warps=waves)
    if mls:
        kernel = matmul_mls_wasp_kernel[(1,)](
            ag, bg, cg, m, n, k, *ag.stride(), *bg.stride(), *cg.stride(),
            BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64,
            GROUP_SIZE_M=1, ACTIVATION="", **options)
    else:
        kernel = _load_gemm[(1,)](ag, bg, cg, k, m, **options)
    torch.testing.assert_close(cg.cpu(), golden.half(), atol=0.01, rtol=0.01)
    assert kernel.metadata.num_warps == sum(waves)
    # A0/A1 are sliced; B0/B1 are full-size but pair-private.
    (tmp_path / "private-buffers.ttgir").write_text(kernel.asm["ttgir"])
    assert kernel.asm["ttgir"].count("ttg.local_alloc") == 4


def test_dense_fa_wdra_spill_repair(monkeypatch):
    from test_wasp_mls_regression import run_dense_fa

    # Observe real backend feedback, without replacing it with synthetic demand.
    # These manual quotas spill; repair must converge within two adjustments
    # and run_dense_fa must still pass its CPU output/LSE reference checks.
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    checks = []
    check = wasp.check_and_adjust_wasp_wdra_regs

    def record_feedback(gcn, src, metadata, options):
        feedback = wasp.read_partition_registers(gcn)
        if feedback is None:
            pytest.skip("Spill repair requires backend per-partition feedback")
        checks.append(feedback)
        return check(gcn, src, metadata, options)

    monkeypatch.setattr(wasp, "check_and_adjust_wasp_wdra_regs", record_feedback)
    run_dense_fa(160, 2, 64, False, partition_regs=(32, 164, 164))
    assert 2 <= len(checks) <= 3
    assert any(entry["vgpr_spill_count"] for entry in checks[0].values())
    assert all(entry["vgpr_spill_count"] == 0 for entry in checks[-1].values())


@pytest.mark.parametrize("waves,stages", [(waves, 2) for waves in [
    # Entire 22-tuple space: 1P1C, 1P2C, 2P2C, C >= P, total <= 16.
    (1, 1), (1, 2), (1, 4), (1, 8), (2, 2), (2, 4), (2, 8),
    (4, 4), (4, 8), (8, 8),
    (1, 1, 1), (1, 2, 2), (1, 4, 4), (2, 2, 2), (2, 4, 4), (4, 4, 4),
    (1, 1, 1, 1), (1, 1, 2, 2), (1, 1, 4, 4), (2, 2, 2, 2),
    (2, 2, 4, 4), (4, 4, 4, 4),
]] + [((1, 4, 4), 4), ((1, 8), 4)])
@pytest.mark.parametrize("mls", [False, True])
def test_mixed_partition_gemm(tmp_path, waves, stages, mls):
    # K spans many LDS reuse rounds; unaligned Consumer starts exercise the
    # relative thread-ID fix as well as EMPTY initialization and READY counts.
    m = n = 64
    k = 512
    a, b, c, golden = gemm_data(m, n, k, transposed_b=mls)
    options = dict(num_warps=4, num_stages=stages, wasp_wdra=False,
                   wasp_partition_warps=waves)
    if mls:
        kernel = matmul_mls_wasp_kernel[(1,)](
            a, b, c, m, n, k, *a.stride(), *b.stride(), *c.stride(),
            BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64,
            GROUP_SIZE_M=1, ACTIVATION="", **options)
    else:
        kernel = _load_gemm[(1,)](a, b, c, k, m, **options)
    torch.testing.assert_close(c.cpu(), golden.half(), atol=0.01, rtol=0.01)
    assert kernel.metadata.num_warps == sum(waves)
    starts = ', '.join(str(sum(waves[:i])) for i in range(len(waves)))
    assert f"warpGroupStartIds = array<i32: {starts}>" in kernel.asm["ttgir"]
    assert "@llvm.hcu.wdra.init" not in kernel.asm["llir"]
    for stage in ("ttgir", "llir", "amdgcn"):
        (tmp_path / f"mixed.{stage}").write_text(kernel.asm[stage])
