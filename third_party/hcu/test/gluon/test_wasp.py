"""Gfx946 Gluon WASP dispatch, GEMM and WDRA tests.

Run in zx-perf-node60 with its PMD environment:
  python3 -m pytest -v -s third_party/hcu/test/gluon/test_wasp.py

HCU has no separate default wave group: use an empty default placeholder,
and put all work in two/three/four four-wave partitions. Besides dispatch,
wasp_gemm demonstrates 4+8 Producer/Consumer GEMM with one LDS buffer and
explicit ebarrier handshakes; it is not an MLS/Abarrier pipelined benchmark.
"""
import pytest
import torch
import triton

libtriton = pytest.importorskip("triton._C.libtriton")
if not hasattr(libtriton, "hcu"):
    pytest.skip("HCU compiler support is not built", allow_module_level=True)

from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.backends.hcu import wasp
from triton.backends.hcu.compiler import HIPBackend
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon._runtime import GluonASTSource


@pytest.fixture(scope="module", autouse=True)
def require_gfx946():
    if not torch.cuda.is_available():
        pytest.skip("Gluon WASP runtime tests require a gfx946 GPU")
    target = triton.runtime.driver.active.get_current_target()
    if target.backend not in ("hcu", "hip") or target.arch != "gfx946":
        pytest.skip("Gluon WASP runtime tests require an HCU gfx946 target")


@gluon.jit
def add_slice(X, Y, Z, N, BLOCK: gl.constexpr, GROUP: gl.constexpr, GROUPS: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    offsets = gl.program_id(0) * (GROUPS * BLOCK) + GROUP * BLOCK + gl.arange(0, BLOCK, layout=layout)
    x = gl.load(X + offsets, offsets < N, other=0)
    y = gl.load(Y + offsets, offsets < N, other=0)
    gl.store(Z + offsets, x + y, offsets < N)


@gluon.jit
def ordinary_add(X, Y, Z, N, BLOCK: gl.constexpr):
    add_slice(X, Y, Z, N, BLOCK, 0, 3)
    add_slice(X, Y, Z, N, BLOCK, 1, 3)
    add_slice(X, Y, Z, N, BLOCK, 2, 3)


@gluon.jit
def empty_default():
    pass


@gluon.jit
def wasp_add(X, Y, Z, N, BLOCK: gl.constexpr, GROUPS: gl.constexpr):
    if GROUPS == 2:
        gl.warp_specialize([
            (empty_default, ()),
            (add_slice, (X, Y, Z, N, BLOCK, 0, GROUPS)),
            (add_slice, (X, Y, Z, N, BLOCK, 1, GROUPS)),
        ], [4, 4], [64, 64])
    elif GROUPS == 3:
        gl.warp_specialize([
            (empty_default, ()),
            (add_slice, (X, Y, Z, N, BLOCK, 0, GROUPS)),
            (add_slice, (X, Y, Z, N, BLOCK, 1, GROUPS)),
            (add_slice, (X, Y, Z, N, BLOCK, 2, GROUPS)),
        ], [4, 4, 4], [64, 64, 64])
    else:
        gl.warp_specialize([
            (empty_default, ()),
            (add_slice, (X, Y, Z, N, BLOCK, 0, GROUPS)),
            (add_slice, (X, Y, Z, N, BLOCK, 1, GROUPS)),
            (add_slice, (X, Y, Z, N, BLOCK, 2, GROUPS)),
            (add_slice, (X, Y, Z, N, BLOCK, 3, GROUPS)),
        ], [4, 4, 4, 4], [64, 64, 64, 64])


@gluon.jit
def nonempty_default_add(X, Y, Z, N, BLOCK: gl.constexpr, GROUPS: gl.constexpr):
    gl.warp_specialize([
        (add_slice, (X, Y, Z, N, BLOCK, 0, GROUPS)),
        (add_slice, (X, Y, Z, N, BLOCK, 0, GROUPS)),
        (add_slice, (X, Y, Z, N, BLOCK, 1, GROUPS)),
        (add_slice, (X, Y, Z, N, BLOCK, 2, GROUPS)),
    ], [4, 4, 4], [64, 64, 64])


@gluon.jit
def gemm_exchange_barrier():
    # This example deliberately synchronizes all 12 waves, unlike the automatic
    # Consumer Ping-Pong barrier (8 waves). ID 14 is outside lowering's reserved
    # IDs and the three partition-local barrier IDs. Drain LDS accesses first.
    gl.inline_asm_elementwise(
        "s_waitcnt lgkmcnt(0)\n\ts_ebarrier_sync 14, 12\n\ts_mov_b32 $0, 0",
        constraints="=s,~{memory}", args=[], dtype=gl.int32, is_pure=False, pack=1)


@gluon.jit
def gemm_producer(A, B, smem_a, smem_b, M, N, K,
                  BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
    rows = gl.program_id(0) * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, layout))
    cols = gl.program_id(1) * BN + gl.arange(0, BN, layout=gl.SliceLayout(0, layout))
    ak = gl.arange(0, BK, layout=gl.SliceLayout(0, layout))
    bk = gl.arange(0, BK, layout=gl.SliceLayout(1, layout))
    for tile in range(gl.cdiv(K, BK)):
        a = gl.load(A + rows[:, None] * K + (tile * BK + ak[None, :]),
                    (rows[:, None] < M) & (tile * BK + ak[None, :] < K), other=0)
        b = gl.load(B + (tile * BK + bk[:, None]) * N + cols[None, :],
                    (tile * BK + bk[:, None] < K) & (cols[None, :] < N), other=0)
        smem_a.store(a)
        smem_b.store(b)
        gemm_exchange_barrier()  # READY: both LDS tiles are visible to Consumers.
        gemm_exchange_barrier()  # EMPTY: both Consumers have finished reading.


@gluon.jit
def gemm_consumer(C, smem_a, smem_b, M, N, K, GROUP: gl.constexpr,
                  BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr):
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3, instr_shape=[16, 16, 16], transposed=False, warps_per_cta=[2, 2])
    dot_a: gl.constexpr = gl.DotOperandLayout(0, mma, 4)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, mma, 4)
    half_a = smem_a.slice(GROUP * (BM // 2), BM // 2, dim=0)
    acc = gl.zeros((BM // 2, BN), gl.float32, layout=mma)
    for tile in range(gl.cdiv(K, BK)):
        gemm_exchange_barrier()  # Wait for Producer to publish the current tile.
        a = half_a.load(dot_a)
        b = smem_b.load(dot_b)
        gemm_exchange_barrier()  # Values are in registers; LDS may be overwritten.
        acc = gl.amd.cdna3.mfma(a, b, acc)
    rows = (gl.program_id(0) * BM + GROUP * (BM // 2)
            + gl.arange(0, BM // 2, layout=gl.SliceLayout(1, mma)))
    cols = gl.program_id(1) * BN + gl.arange(0, BN, layout=gl.SliceLayout(0, mma))
    gl.store(C + rows[:, None] * N + cols[None, :], acc,
             (rows[:, None] < M) & (cols[None, :] < N))


@gluon.jit
def wasp_gemm(A, B, C, M, N, K, BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr):
    # A single shared tile pair, deliberately no multi-stage MLS pipeline.
    shared: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    smem_a = gl.allocate_shared_memory(gl.float16, [BM, BK], shared)
    smem_b = gl.allocate_shared_memory(gl.float16, [BK, BN], shared)
    gl.warp_specialize([
        (empty_default, ()),
        (gemm_producer, (A, B, smem_a, smem_b, M, N, K, BM, BN, BK)),
        (gemm_consumer, (C, smem_a, smem_b, M, N, K, 0, BM, BN, BK)),
        (gemm_consumer, (C, smem_a, smem_b, M, N, K, 1, BM, BN, BK)),
    ], [4, 4, 4], [64, 128, 128])


@pytest.mark.parametrize("mode", ["off", "explicit", "automatic"])
@pytest.mark.parametrize("shape", [(64, 32, 32), (93, 47, 81)])
def test_gluon_wasp_gemm(tmp_path, mode, shape):
    # Second shape covers M/N/K tails, multiple CTAs and three LDS reuse rounds.
    m, n, k = shape
    generator = torch.Generator().manual_seed(2026)
    a = torch.randn((m, k), generator=generator, dtype=torch.float16, device="cpu")
    b = torch.randn((k, n), generator=generator, dtype=torch.float16, device="cpu")
    golden = a.float() @ b.float()
    a_gpu, b_gpu = a.cuda(), b.cuda()
    c_gpu = torch.full((m, n), float("nan"), dtype=torch.float32, device="cpu").cuda()
    options = {"num_warps": 4, "num_stages": 2, "wasp_partition_warps": (4, 4, 4),
               "wasp_wdra": mode != "off"}
    if mode == "explicit":
        options["wasp_partition_regs"] = (64, 128, 132)  # sum=324, divisible by 12.
    kernel = wasp_gemm[(triton.cdiv(m, 64), triton.cdiv(n, 32))](
        a_gpu, b_gpu, c_gpu, m, n, k, 64, 32, 32, **options)
    torch.testing.assert_close(c_gpu.cpu(), golden, rtol=1e-3, atol=1e-3)
    assert kernel.metadata.num_warps == 12
    assert "warpGroupStartIds = array<i32: 0, 4, 8>" in kernel.asm["ttgir"]
    assert "tt.dot" in kernel.asm["ttgir"]
    assert "s_ebarrier_sync 14, 12" in kernel.asm["amdgcn"]
    assert ("@llvm.hcu.wdra.init" in kernel.asm["llir"]) == (mode != "off")
    for stage in ("ttgir", "llir", "amdgcn"):
        (tmp_path / f"gemm.{stage}").write_text(kernel.asm[stage])
    print(f"Gluon WASP GEMM {shape}, 4+8 waves, WDRA={mode}: matched CPU golden")


def make_source(fn, groups):
    return GluonASTSource(
        fn, signature={"X": "*fp32", "Y": "*fp32", "Z": "*fp32", "N": "i32",
                       "BLOCK": "constexpr", "GROUPS": "constexpr"},
        constexprs={"BLOCK": 256, "GROUPS": groups})


def test_ordinary_gluon_add(tmp_path):
    n, block = 1003, 256
    generator = torch.Generator().manual_seed(2026)
    x = torch.randn(n, generator=generator, dtype=torch.float32, device="cpu")
    y = torch.randn(n, generator=generator, dtype=torch.float32, device="cpu")
    golden = x + y
    x_gpu, y_gpu = x.cuda(), y.cuda()
    z_gpu = torch.empty_like(x_gpu)
    kernel = ordinary_add[(triton.cdiv(n, 3 * block),)](
        x_gpu, y_gpu, z_gpu, n, block, num_warps=4, wasp_wdra=False)
    torch.testing.assert_close(z_gpu.cpu(), golden, rtol=0, atol=0)
    (tmp_path / "ordinary.ttgir").write_text(kernel.asm["ttgir"])
    (tmp_path / "ordinary.amdgcn").write_text(kernel.asm["amdgcn"])
    print("Ordinary Gluon: compiled, executed, and matched CPU golden")


@pytest.mark.parametrize("mode", ["off", "explicit", "automatic"])
@pytest.mark.parametrize("total", [8, 12, 16])
def test_gluon_wasp_add(tmp_path, mode, total):
    groups = total // 4
    wdra = mode != "off"
    target = GPUTarget("hcu", "gfx946", 64)
    backend = HIPBackend(target)
    options = {"num_warps": 4, "num_stages": 2, "wasp_partition_warps": (4,) * groups,
               "wasp_wdra": wdra}
    if mode == "explicit":
        options["wasp_partition_regs"] = (64,) * groups
    parsed = backend.parse_options(options)
    source = make_source(wasp_add, groups)
    ctx = ir.context()
    ir.load_dialects(ctx)
    backend.load_dialects(ctx)
    mod = source.make_ir(target, parsed, backend.get_codegen_implementation(parsed),
                         backend.get_module_map(), ctx)
    frontend = str(mod)
    assert "ttg.warp_specialize" in frontend
    assert "partition0(" in frontend and "partition1(" in frontend
    assert frontend.count("num_warps(4)") == groups
    (tmp_path / "wasp-frontend.ttgir").write_text(frontend)
    n, block = groups * 256 + 17, 256
    generator = torch.Generator().manual_seed(2026)
    x = torch.randn(n, generator=generator, device="cpu")
    y = torch.randn(n, generator=generator, device="cpu")
    golden = x + y
    x_gpu, y_gpu = x.cuda(), y.cuda()
    z_gpu = torch.full_like(x, float("nan")).cuda()
    kernel = wasp_add[(triton.cdiv(n, groups * block),)](
        x_gpu, y_gpu, z_gpu, n, block, groups, **options)
    torch.testing.assert_close(z_gpu.cpu(), golden, rtol=0, atol=0)
    assert kernel.metadata.num_warps == total
    assert kernel.metadata.wasp_wdra == wdra
    assert "ttg.warp_specialize" in kernel.asm["ttgir"]
    start_ids = ", ".join(str(i) for i in range(0, total, 4))
    assert f"warpGroupStartIds = array<i32: {start_ids}>" in kernel.asm["ttgir"]
    assert ("@llvm.hcu.wdra.init" in kernel.asm["llir"]) == wdra
    if mode == "explicit":
        assert tuple(kernel.metadata.wasp_partition_regs) == (64,) * groups
    if wdra:
        feedback = wasp.read_partition_registers(kernel.asm["amdgcn"])
        if feedback is not None:
            # New backends report quotas and raw demand even without spills.
            assert all(entry["vgpr_spill_count"] == 0 for entry in feedback.values())
            assert tuple(kernel.metadata.wasp_partition_regs) == tuple(
                feedback[i]["available_vgprs"] for i in range(groups))
            assert tuple(kernel.metadata.wasp_partition_regs_actual) == tuple(
                feedback[i]["required_vgprs"] for i in range(groups))
        elif mode == "automatic":
            # Older backends retain the global no-spill fast path.
            assert kernel.metadata.wasp_partition_regs is None
            assert not hasattr(kernel.metadata, "wasp_partition_regs_actual")
    for stage in ("ttgir", "llir", "amdgcn"):
        (tmp_path / f"wasp.{stage}").write_text(kernel.asm[stage])
    print(f"Gluon WASP ({total} waves, WDRA={mode}): compiled, ran, matched CPU golden")


@pytest.mark.parametrize("fn,total,message", [
    (wasp_add, 16, "HCU WASP partition count does not match topology"),
    (nonempty_default_add, 12, "HCU WASP requires an empty default region"),
])
def test_gluon_wasp_invalid_partition_contract(capfd, fn, total, message):
    with pytest.raises((RuntimeError, ValueError)) as error:
        triton.compile(make_source(fn, 3), target=GPUTarget("hcu", "gfx946", 64),
                       options={"num_warps": 4, "num_stages": 2,
                                "wasp_partition_warps": (4,) * (total // 4), "wasp_wdra": False})
    assert message in str(error.value) + capfd.readouterr().err
