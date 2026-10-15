# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Compile scaled GEMMs through LLVM IR without a GPU or external codegen toolchain."""
import inspect
import re
from types import SimpleNamespace

import pytest
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon._runtime import GluonASTSource

import dot_scaled_kernels as upstream_kernels

TARGET = GPUTarget("hip", "gfx946", 64)
SIGNATURE = {"A": "*u8", "B": "*u8", "SA": "*u8", "SB": "*u8", "C": "*fp32"}


def compile_to_llvm(source, target, options):
    """Run the normal backend stages through LLVM export, without clang/HSACO."""
    from triton._C.libtriton import ir
    from triton.compiler.compiler import IRSource, make_backend

    backend = make_backend(target)
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    is_ir = isinstance(source, str)
    if is_ir:
        source = IRSource(source, context, backend)
    options = backend.parse_options(dict(options, **source.parse_options()))
    metadata = {"target": target, **options.__dict__}
    module = source.make_ir(target, options, backend.get_codegen_implementation(options),
                            backend.get_module_map(), context)
    stages = {}
    backend.add_stages(stages, options, source.language)
    first = list(stages).index(source.ext) + int(is_ir)
    outputs = {source.ext: str(module)}
    for name, lower in list(stages.items())[first:]:
        module = lower(module, metadata)
        outputs[name] = str(module)
        if name == "llir":
            break
    assert "llir" in outputs
    return SimpleNamespace(asm=outputs)


@pytest.fixture(autouse=True)
def forbid_device_access(monkeypatch):
    class CompileOnlyDriver:
        def __getattr__(self, name):
            pytest.fail(f"compile-only test accessed the device driver: {name}")

    monkeypatch.setattr(triton.runtime.driver, "_active", CompileOnlyDriver())
    from triton.backends.hcu.compiler import HIPBackend

    def forbid_clang():
        pytest.fail("compile-only test requested external code generation")

    monkeypatch.setattr(HIPBackend, "path_to_rocm_clang", staticmethod(forbid_clang))


# Shared GEMM kernels and compiler-output checks.

@triton.jit
def bank_kernel(A, B, SA, SB, C, M: tl.constexpr, N: tl.constexpr, BK: tl.constexpr, TILES: tl.constexpr,
                STAGES: tl.constexpr, MLS: tl.constexpr, R_CONTIGUOUS: tl.constexpr,
                WARP_SPECIALIZE: tl.constexpr = False):
    m = tl.arange(0, M)
    n = tl.arange(0, N)
    k = tl.arange(0, BK // 2)
    s = tl.arange(0, BK // 32)
    acc = tl.full((M, N), 0, tl.float32)
    for i in tl.range(0, TILES, num_stages=STAGES, warp_specialize=WARP_SPECIALIZE):
        if MLS:
            a = tl.matrix_load(A, shape=[M, TILES * BK // 2], strides=[TILES * BK // 2, 1], block_shape=[M, BK // 2],
                               offsets=[0, i * BK // 2])
            b = tl.matrix_load(B, shape=[TILES * BK // 2, N], strides=[1, TILES * BK // 2], block_shape=[BK // 2, N],
                               offsets=[i * BK // 2, 0])
        else:
            a = tl.load(A + m[:, None] * (TILES * BK // 2) + i * BK // 2 + k[None, :])
            b = tl.load(B + n[None, :] * (TILES * BK // 2) + i * BK // 2 + k[:, None])
        if R_CONTIGUOUS:
            sa = tl.load(SA + m[:, None] + (i * (BK // 32) + s[None, :]) * M)
            sb = tl.load(SB + n[:, None] + (i * (BK // 32) + s[None, :]) * N)
        else:
            sa = tl.load(SA + m[:, None] * (TILES * BK // 32) + i * (BK // 32) + s[None, :])
            sb = tl.load(SB + n[:, None] * (TILES * BK // 32) + i * (BK // 32) + s[None, :])
        acc = tl.dot_scaled(a, sa, "e2m1", b, sb, "e2m1", acc=acc)
    tl.store(C + m[:, None] * N + n[None, :], acc)


def compile_bank_kernel(target, m, n, bk, tiles=1, stages=1, mls=False, r_contiguous=False,
                        async_copy=False, warp_specialize=False, compiler=compile_to_llvm, **options):
    src = ASTSource(
        bank_kernel, signature={"A": "*u8", "B": "*u8", "SA": "*u8", "SB": "*u8", "C": "*fp32"}, constexprs={
            "M": m, "N": n, "BK": bk, "TILES": tiles, "STAGES": stages, "MLS": mls,
            "R_CONTIGUOUS": r_contiguous, "WARP_SPECIALIZE": warp_specialize
        }, attrs={(i, ): [("tt.divisibility", 16), ("tt.pointer_range", 32)]
                  for i in range(5)})
    return compiler(
        src, target=target,
        options={"num_warps": 4, "num_stages": stages, "mmac_layout_force": 2, "use_async_copy": async_copy, **options})


def check_mmac_banks(llir):
    calls = [line for line in llir.splitlines() if " call " in line and "@llvm.hcu.mmac.scale." in line]
    assert calls
    for call in calls:
        offsets = re.findall(r"\bi32 (-?\d+)(?=[,)])", call)
        assert len(offsets) >= 2, call
        a, b = map(int, offsets[-2:])
        groups = 2 if "16x16x64.fp4" in call else 1
        assert 0 <= a <= 32 - groups and 0 <= b <= 32 - groups, call
        # m0 bases are eight-row aligned; immediate bit2 selects the SRAM.
        for j in range(groups):
            assert ((a + j) ^ (b + j)) & 4, call
    return len(calls)


@triton.jit
def scaled_loop(A, B, SA, SB, C, MLS: tl.constexpr, R_CONTIGUOUS: tl.constexpr, S: tl.constexpr, TILES: tl.constexpr,
                STAGES: tl.constexpr, Limit=None, MIXED_MLS: tl.constexpr = 0, FP8: tl.constexpr = False):
    r = tl.arange(0, 64)
    tile_k: tl.constexpr = S * (32 if FP8 else 16)
    k = tl.arange(0, tile_k)
    s = tl.arange(0, S)
    acc = tl.full((64, 64), 0, tl.float32)
    for i in tl.range(0, TILES if Limit is None else Limit, num_stages=STAGES):
        if MLS and MIXED_MLS != 2:
            a = tl.matrix_load(A, shape=[64, TILES * tile_k], strides=[TILES * tile_k, 1], block_shape=[64, tile_k],
                               offsets=[0, i * tile_k])
        else:
            a = tl.load(A + r[:, None] * (TILES * tile_k) + i * tile_k + k[None, :])
        if MLS and MIXED_MLS != 1:
            b = tl.matrix_load(B, shape=[TILES * tile_k, 64], strides=[1, TILES * tile_k], block_shape=[tile_k, 64],
                               offsets=[i * tile_k, 0])
        else:
            b = tl.load(B + r[None, :] * (TILES * tile_k) + i * tile_k + k[:, None])
        if R_CONTIGUOUS:
            sa = tl.load(SA + (i * S + s[None, :]) * 64 + r[:, None])
            sb = tl.load(SB + (i * S + s[None, :]) * 64 + r[:, None])
        else:
            sa = tl.load(SA + r[:, None] * (TILES * S) + i * S + s[None, :])
            sb = tl.load(SB + r[:, None] * (TILES * S) + i * S + s[None, :])
        acc = tl.dot_scaled(a, sa, "e4m3" if FP8 else "e2m1", b, sb, "e4m3" if FP8 else "e2m1", acc=acc)
    tl.store(C + r[:, None] * 64 + r[None, :], acc)


def assert_scale_publication(assembly):
    # Machine block layout may place a consumer earlier in the text than its
    # producer. Follow CFG edges rather than comparing string positions.
    blocks = {"entry": []}
    current = "entry"
    for line in assembly.splitlines():
        if line.startswith(".Lfunc_end"):
            break
        label = re.match(r"(\.LBB\d+_\d+):", line)
        if label:
            current = label[1]
            blocks[current] = []
        else:
            instruction = line.strip().split(";", 1)[0].strip()
            if instruction and not instruction.startswith((".", "#")):
                blocks[current].append(instruction)
    names = list(blocks)
    pending = [("entry", False, False)]
    seen = set()
    checked = False
    while pending:
        block, waited, published = state = pending.pop()
        if state in seen:
            continue
        seen.add(state)
        terminated = False
        for line in blocks[block]:
            if "ds_scale_copy_ds2buf" in line:
                waited = published = False
            if line.startswith("s_waitcnt") and "lgkmcnt(0)" in line:
                waited = True
            # WASP publishes within the consumer partition; both barrier
            # forms still require the preceding lgkm completion wait.
            if re.match(r"s_(?:barrier|ebarrier_sync)(?:\s|$)", line) and waited:
                published = True
            if "v_mmac_scale_" in line:
                checked = True
                assert published, f"MMAC without scale publication on CFG path into {block}"
            branch = re.match(r"s_(?:cbranch_\w+|branch)\s+(\.LBB\d+_\d+)", line)
            if branch:
                pending.append((branch[1], waited, published))
                if line.startswith("s_branch ") or line.startswith("s_branch\t"):
                    terminated = True
                    break
            if line.startswith("s_endpgm"):
                terminated = True
                break
        i = names.index(block)
        if not terminated and i + 1 < len(names):
            pending.append((names[i + 1], waited, published))
    assert checked and "ds_scale_copy_ds2buf" in assembly


@triton.jit
def optional_scale_kernel(A, B, SA, SB, C, M: tl.constexpr, N: tl.constexpr, MLS: tl.constexpr, SIDE: tl.constexpr,
                          R_CONTIGUOUS: tl.constexpr, STAGES: tl.constexpr, TILES: tl.constexpr,
                          WARP_SPECIALIZE: tl.constexpr = False, BK: tl.constexpr = 256):
    m = tl.arange(0, M)
    n = tl.arange(0, N)
    k = tl.arange(0, BK // 2)
    s = tl.arange(0, BK // 32)
    acc = tl.full((M, N), 0, tl.float32)
    for i in tl.range(0, TILES, num_stages=STAGES, warp_specialize=WARP_SPECIALIZE):
        if MLS:
            a = tl.matrix_load(A, shape=[M, TILES * (BK // 2)], strides=[TILES * (BK // 2), 1], block_shape=[M, BK // 2],
                               offsets=[0, i * (BK // 2)])
            b = tl.matrix_load(B, shape=[TILES * (BK // 2), N], strides=[1, TILES * (BK // 2)], block_shape=[BK // 2, N],
                               offsets=[i * (BK // 2), 0])
        else:
            a = tl.load(A + m[:, None] * (TILES * (BK // 2)) + i * (BK // 2) + k[None, :])
            b = tl.load(B + n[None, :] * (TILES * (BK // 2)) + i * (BK // 2) + k[:, None])
        sa = None
        sb = None
        if SIDE == "a" or SIDE == "both":
            if R_CONTIGUOUS:
                sa = tl.load(SA + (i * (BK // 32) + s[None, :]) * M + m[:, None])
            else:
                sa = tl.load(SA + m[:, None] * (TILES * (BK // 32)) + i * (BK // 32) + s[None, :])
        if SIDE == "b" or SIDE == "both":
            if R_CONTIGUOUS:
                sb = tl.load(SB + (i * (BK // 32) + s[None, :]) * N + n[:, None])
            else:
                sb = tl.load(SB + n[:, None] * (TILES * (BK // 32)) + i * (BK // 32) + s[None, :])
        acc = tl.dot_scaled(a, sa, "e2m1", b, sb, "e2m1", acc=acc)
    tl.store(C + m[:, None] * N + n[None, :], acc)


@triton.jit
def chain_kernel(A, B, SA, SB, C, Bound, MLS: tl.constexpr, R_CONTIGUOUS: tl.constexpr, MASK: tl.constexpr):
    m = tl.arange(0, 64)
    n = tl.arange(0, 64)
    k = tl.arange(0, 128)
    s = tl.arange(0, 8)
    if MLS:
        a = tl.matrix_load(A, shape=[64, 128], strides=[128, 1], block_shape=[64, 128], offsets=[0, 0])
        b = tl.matrix_load(B, shape=[128, 64], strides=[1, 128], block_shape=[128, 64], offsets=[0, 0])
    else:
        a = tl.load(A + m[:, None] * 128 + k[None, :])
        b = tl.load(B + k[:, None] + n[None, :] * 128)
    if R_CONTIGUOUS:
        ap = SA + m[:, None] + s[None, :] * 64
        bp = SB + n[:, None] + s[None, :] * 64
    else:
        ap = SA + m[:, None] * 8 + s[None, :]
        bp = SB + n[:, None] * 8 + s[None, :]
    if MASK:
        sa = tl.load(ap, mask=m[:, None] < Bound, other=127)
        sb = tl.load(bp, mask=n[:, None] < Bound, other=127)
    else:
        sa = tl.load(ap)
        sb = tl.load(bp)
    c = tl.dot_scaled(a, sa, "e2m1", b, sb, "e2m1")
    tl.store(C + m[:, None] * 64 + n[None, :], c)


@triton.jit
def small_scale_kernel(A, B, SA, SB, C):
    m = tl.arange(0, 32)
    k = tl.arange(0, 32)
    s = tl.arange(0, 2)
    a = tl.load(A + m[:, None] * 32 + k[None, :])
    b = tl.load(B + k[:, None] + m[None, :] * 32)
    sa = tl.load(SA + m[:, None] * 2 + s[None, :])
    sb = tl.load(SB + m[:, None] * 2 + s[None, :])
    c = tl.dot_scaled(a, sa, "e2m1", b, sb, "e2m1")
    tl.store(C + m[:, None] * 32 + m[None, :], c)


@triton.jit
def reuse_kernel(A, B, SA, SB, C, DOTS: tl.constexpr):
    a = tl.matrix_load(A, shape=[64, 128], strides=[128, 1], block_shape=[64, 128], offsets=[0, 0])
    b = tl.matrix_load(B, shape=[128, 64], strides=[1, 128], block_shape=[128, 64], offsets=[0, 0])
    m = tl.arange(0, 64)
    s = tl.arange(0, 8)
    c = tl.full((64, 64), 0, tl.float32)
    for d in tl.static_range(0, DOTS):
        sa = tl.load(SA + m[:, None] * (DOTS * 8) + d * 8 + s[None, :])
        sb = tl.load(SB + m[:, None] * (DOTS * 8) + d * 8 + s[None, :])
        c = tl.dot_scaled(a, sa, "e2m1", b, sb, "e2m1", acc=c)
        tl.store(C + d * 4096 + m[:, None] * 64 + m[None, :], c)
        tl.debug_barrier()


@triton.jit
def reuse_without_store_kernel(A, B, SA, SB, C, DOTS: tl.constexpr, MLS: tl.constexpr, BARRIER: tl.constexpr):
    m = tl.arange(0, 64)
    k = tl.arange(0, 128)
    s = tl.arange(0, 8)
    if MLS:
        a = tl.matrix_load(A, shape=[64, 128], strides=[128, 1], block_shape=[64, 128], offsets=[0, 0])
        b = tl.matrix_load(B, shape=[128, 64], strides=[1, 128], block_shape=[128, 64], offsets=[0, 0])
    else:
        a = tl.load(A + m[:, None] * 128 + k[None, :])
        b = tl.load(B + k[:, None] + m[None, :] * 128)
    results = ()
    for d in tl.static_range(0, DOTS):
        sa = tl.load(SA + m[:, None] * (DOTS * 8) + d * 8 + s[None, :])
        sb = tl.load(SB + m[:, None] * (DOTS * 8) + d * 8 + s[None, :])
        result = tl.dot_scaled(a, sa, "e2m1", b, sb, "e2m1")
        results += (result, )
        if BARRIER:
            tl.debug_barrier()
    # Keep every independent dot result live; no intermediate Global store
    # or accumulator dependency can serve as the allocator's reuse proof.
    for d in tl.static_range(0, DOTS):
        tl.store(C + d * 4096 + m[:, None] * 64 + m[None, :], results[d])


@triton.jit
def small_bk_kernel(A, B, SA, SB, C, Bound, MLS: tl.constexpr, PITCH: tl.constexpr, MASK: tl.constexpr,
                    R_CONTIGUOUS: tl.constexpr):
    m = tl.arange(0, 128)
    k = tl.arange(0, 32)
    s = tl.arange(0, 2)
    if MLS:
        a = tl.matrix_load(A, shape=[128, 32], strides=[32, 1], block_shape=[128, 32], offsets=[0, 0])
        b = tl.matrix_load(B, shape=[32, 128], strides=[1, 32], block_shape=[32, 128], offsets=[0, 0])
    else:
        a = tl.load(A + m[:, None] * 32 + k[None, :])
        b = tl.load(B + k[:, None] + m[None, :] * 32)
    if R_CONTIGUOUS:
        ap = SA + m[:, None] + s[None, :] * 128
        bp = SB + m[:, None] + s[None, :] * 128
    else:
        ap = SA + m[:, None] * PITCH + s[None, :]
        bp = SB + m[:, None] * PITCH + s[None, :]
    if MASK:
        sa = tl.load(ap, mask=m[:, None] < Bound, other=127)
        sb = tl.load(bp, mask=m[:, None] < Bound, other=127)
    else:
        sa = tl.load(ap)
        sb = tl.load(bp)
    c = tl.dot_scaled(a, sa, "e2m1", b, sb, "e2m1")
    tl.store(C + m[:, None] * 128 + m[None, :], c)


@triton.jit
def wide_scale_offsets_kernel(A, B, SA, SB, C):
    a = tl.matrix_load(A, shape=[64, 128], strides=[128, 1], block_shape=[64, 128], offsets=[0, 0])
    b = tl.matrix_load(B, shape=[128, 64], strides=[1, 128], block_shape=[128, 64], offsets=[0, 0])
    r = tl.arange(0, 64).to(tl.int64)
    s = tl.arange(0, 8).to(tl.int64)
    sa = tl.load(SA + r[:, None] * 8 + s[None, :])
    sb = tl.load(SB + r[:, None] * 8 + s[None, :])
    c = tl.dot_scaled(a, sa, "e2m1", b, sb, "e2m1")
    tl.store(C + r[:, None] * 64 + r[None, :], c)


@triton.jit
def global_scale_loop(A, B, SA, SB, C, MLS: tl.constexpr, STAGES: tl.constexpr):
    r = tl.arange(0, 64)
    k = tl.arange(0, 128)
    s = tl.arange(0, 8)
    acc = tl.full((64, 64), 0, tl.float32)
    for i in tl.range(0, 3, num_stages=STAGES):
        if MLS:
            a = tl.matrix_load(A, shape=[64, 384], strides=[384, 1], block_shape=[64, 128], offsets=[0, i * 128])
            b = tl.matrix_load(B, shape=[384, 64], strides=[1, 384], block_shape=[128, 64], offsets=[i * 128, 0])
        else:
            a = tl.load(A + r[:, None] * 384 + i * 128 + k[None, :])
            b = tl.load(B + r[None, :] * 384 + i * 128 + k[:, None])
        offsets = r[:, None] * 24 + i * 8 + s[None, :]
        sa = tl.load(SA + offsets)
        sb = tl.load(SB + offsets)
        acc = tl.dot_scaled(a, sa, "e2m1", b, sb, "e2m1", acc=acc)
    tl.store(C + r[:, None] * 64 + r[None, :], acc)


@triton.jit
def scaled_copy_kernel(A, B, SA, SB, C, MLS: tl.constexpr):
    m = tl.arange(0, 64)
    n = tl.arange(0, 64)
    k = tl.arange(0, 128)
    s = tl.arange(0, 8)
    if MLS:
        a = tl.matrix_load(A, shape=[64, 128], strides=[128, 1],
                           block_shape=[64, 128], offsets=[0, 0])
        b = tl.matrix_load(B, shape=[128, 64], strides=[1, 128],
                           block_shape=[128, 64], offsets=[0, 0])
    else:
        a = tl.load(A + m[:, None] * 128 + k[None, :])
        b = tl.load(B + k[:, None] + n[None, :] * 128)
    sa = tl.load(SA + m[:, None] * 8 + s[None, :])
    sb = tl.load(SB + n[:, None] * 8 + s[None, :])
    c = tl.dot_scaled(a, sa, "e2m1", b, sb, "e2m1")
    tl.store(C + m[:, None] * 64 + n[None, :], c)


@triton.jit
def format_kernel(A, B, SA, SB, C, AF: tl.constexpr, BF: tl.constexpr,
                  K: tl.constexpr = 128, A_MLS: tl.constexpr = True,
                  B_MLS: tl.constexpr = True):
    AK: tl.constexpr = K // 2 if AF == "e2m1" else K
    BK: tl.constexpr = K // 2 if BF == "e2m1" else K
    r = tl.arange(0, 64)
    if A_MLS:
        a = tl.matrix_load(A, shape=[64, AK], strides=[AK, 1],
                           block_shape=[64, AK], offsets=[0, 0])
    else:
        a = tl.load(A + r[:, None] * AK + tl.arange(0, AK)[None, :])
    if B_MLS:
        b = tl.matrix_load(B, shape=[BK, 64], strides=[1, BK],
                           block_shape=[BK, 64], offsets=[0, 0])
    else:
        b = tl.load(B + r[None, :] * BK + tl.arange(0, BK)[:, None])
    s = tl.arange(0, K // 32)
    sa = tl.load(SA + r[:, None] * (K // 32) + s[None, :])
    sb = tl.load(SB + r[:, None] * (K // 32) + s[None, :])
    c = tl.dot_scaled(a, sa, AF, b, sb, BF)
    tl.store(C + r[:, None] * 64 + r[None, :], c)


def check_scaled_kernel(kernel, has_scale=True):
    assert kernel.asm["llir"], "compilation must reach LLVM IR"
    assert "unrealized_conversion_cast" not in kernel.asm["llir"]
    assert '"amdg.m0"' not in kernel.asm["llir"]
    if has_scale:
        assert "amdg.scale_write" in kernel.asm["ttgir"]
        assert "amdg.dot_scaled" in kernel.asm["ttgir"]
        check_mmac_banks(kernel.asm["llir"])
        assert "llvm.hcu.ds.scale.copy.ds2buf" in kernel.asm["llir"]
    else:
        assert "amdg.scale_write" not in kernel.asm["ttgir"]
        assert "llvm.hcu.ds.scale.copy.ds2buf" not in kernel.asm["llir"]


# Compilation, transport policy and descriptor verification.

@pytest.mark.parametrize("m,n,bk", [(64, 64, 256), (256, 64, 64)])
@pytest.mark.parametrize("stages", [1, 2])
@pytest.mark.parametrize("mls", [False, True])
@pytest.mark.parametrize("async_copy", [False, True])
@pytest.mark.parametrize("r_contiguous", [False, True])
def test_scaled_compile_paths(m, n, bk, stages, mls, async_copy, r_contiguous):
    kernel = compile_bank_kernel(
        TARGET, m, n, bk, tiles=4, stages=stages, mls=mls,
        r_contiguous=r_contiguous, async_copy=async_copy)
    check_scaled_kernel(kernel)
    scale_shapes = (f"{m}x{bk // 32}xi8", f"{n}x{bk // 32}xi8")
    native = [line for line in kernel.asm["ttgir"].splitlines()
              if any(shape in line for shape in scale_shapes)
              and ("buffer_load_to_local" in line or "async_copy_global_to_local" in line)]
    assert bool(native) == (bk // 32 >= 4 and (mls or async_copy))
    if mls and stages == 2 and native:
        body = kernel.asm["ttgir"].split("scf.for", 1)[1].split("scf.yield", 1)[0]
        waits = [line.split("ttg.async_wait", 1)[1] for line in body.splitlines() if "ttg.async_wait" in line]
        assert len(waits) == 1, "current A/B and scale inputs must share one readiness wait"
        assert len(re.findall(r"%[\w]+", waits[0].split("{", 1)[0])) == 4
        assert body.rindex("ttg.local_load") < body.index("amdg.scale_write") < body.index("amdg.buffer_load_to_local"), (
            "scale writes must follow fragment reads and precede the next scale prefetch")
    assert not any("ttg.convert_layout" in line and any(shape in line for shape in scale_shapes)
                   for line in kernel.asm["ttgir"].splitlines())


def test_scale_write_async_wait_filters_lds_dependencies(tmp_path):
    kernel = compile_bank_kernel(TARGET, 64, 64, 128, tiles=5, stages=2,
                                 mls=False, async_copy=True)
    source = kernel.asm["ttgir"]
    barriers = []
    for waited in (True, False):
        ir = source if waited else re.sub(
            r"(amdg\.scale_write %\w+) token %\w+", r"\1", source)
        assert waited or ir != source
        path = tmp_path / f"scale_wait_{waited}.ttgir"
        path.write_text(ir)
        compiled = compile_to_llvm(str(path), target=TARGET, options={
            "num_warps": 4, "num_stages": 2, "mmac_layout_force": 2,
            "use_async_copy": True,
        })
        check_scaled_kernel(compiled)
        barriers.append(compiled.asm["llir"].count("call void @llvm.amdgcn.s.barrier()"))
    # Only waited LDS reads can use the native asynchronous-copy filter.
    # Scale Buffer publication is still checked in both variants above.
    assert barriers[1] > barriers[0]


@pytest.mark.parametrize("side", ["a", "b", "none"])
@pytest.mark.parametrize("mls", [False, True])
@pytest.mark.parametrize("stages", [1, 2])
def test_scaled_compile_optional(side, mls, stages):
    source = ASTSource(optional_scale_kernel, signature=SIGNATURE, constexprs={
        "M": 64, "N": 64, "MLS": mls, "SIDE": side, "R_CONTIGUOUS": False,
        "STAGES": stages, "TILES": 4, "WARP_SPECIALIZE": False, "BK": 256,
    }, attrs={(i,): [("tt.divisibility", 16)] for i in range(5)})
    kernel = compile_to_llvm(source, target=TARGET, options={
        "num_warps": 4, "num_stages": stages, "mmac_layout_force": 2, "use_async_copy": True,
    })
    check_scaled_kernel(kernel, has_scale=side != "none")
    if side != "none":
        assert "dense<127>" in kernel.asm["ttgir"], "missing scale must be E8M0 one"


@pytest.mark.parametrize("warps,stages,wdra", [
    ((4, 4), 2, False), ((4, 4), 4, False),
    ((4, 4, 4), 2, False), ((4, 4, 4), 2, True),
    ((4, 4, 4), 4, True), ((4, 8), 2, False),
    ((4, 4, 4, 4), 2, False), ((4, 4, 4, 4), 2, True),
])
def test_scaled_compile_wasp(warps, stages, wdra):
    kernel = compile_bank_kernel(
        TARGET, 128, 128, 128, tiles=2 * stages, stages=stages,
        mls=True, async_copy=True, warp_specialize=True,
        wasp_partition_warps=warps, wasp_wdra=wdra)
    check_scaled_kernel(kernel)
    assert "ttg.warp_specialize" in kernel.asm["ttgir"]
    assert "llvm.hcu.s.ebarrier.sync" in kernel.asm["llir"]
    # MLS already transfers A/B directly. Both scale producers must follow
    # the requested native-copy policy, including sliced dual consumers.
    assert kernel.asm["ttgir"].count("amdg.buffer_load_to_local") >= 2
    assert "amdg.memory_counter_wait load(0)" not in kernel.asm["ttgir"]
    assert "llvm.amdgcn.raw.ptr.buffer.load.async.lds" in kernel.asm["llir"]
    for partition in re.split(r"\n    partition\d+\(", kernel.asm["ttgir"])[1:]:
        transfers = re.findall(
            r"\b(amdg.matrix_load_to_local|amdg.buffer_load_to_local)\b", partition)
        pair = ["amdg.matrix_load_to_local", "amdg.buffer_load_to_local"]
        assert transfers == pair * (len(transfers) // 2)
        consumers = re.findall(r"\b(ttg.local_load|amdg.scale_write)\b", partition)
        if "amdg.scale_write" in consumers:
            last_load = max(i for i, op in enumerate(consumers) if op == "ttg.local_load")
            assert last_load < consumers.index("amdg.scale_write")
    if warps == (4, 4, 4, 4):
        # Both empty notifications must follow the paired scale LDS reads.
        # Previously the first pair released its slot before scale_write.
        lines = kernel.asm["ttgir"].splitlines()
        for i, line in enumerate(lines):
            write = re.search(r"amdg.scale_write (%[\w]+)", line)
            if not write:
                continue
            source = next(j for j in range(i - 1, -1, -1)
                          if re.match(rf"\s*{re.escape(write[1])} =", lines[j]))
            assert not any("rocdl.hcu.s.abarrier.arrive" in x for x in lines[source:i]), (
                "WASP releases the producer slot before its scale LDS read")
        writes = [i for i, line in enumerate(lines) if "amdg.scale_write" in line]
        for a, b in zip(writes[::2], writes[1::2]):
            assert not any("rocdl.hcu.s.abarrier.arrive" in x for x in lines[a:b]), (
                "paired scale reads should share the completion wait before EMPTY")


def test_scaled_compile_wasp_untracked_native_copy():
    kernel = compile_bank_kernel(
        TARGET, 128, 128, 128, tiles=4, stages=2, mls=False,
        async_copy=True, warp_specialize=True,
        wasp_partition_warps=(4, 4, 4), wasp_wdra=False)
    check_scaled_kernel(kernel)
    assert kernel.asm["ttgir"].count("amdg.buffer_load_to_local") == 2
    assert kernel.asm["ttgir"].count("amdg.memory_counter_wait load(0)") == 2


@pytest.mark.parametrize("side", ["none", "both"])
def test_scaled_compile_wasp_mls_tile_reselection(side):
    source = ASTSource(optional_scale_kernel, signature=SIGNATURE, constexprs={
        "M": 128, "N": 128, "MLS": True, "SIDE": side,
        "R_CONTIGUOUS": False, "STAGES": 2, "TILES": 4,
        "WARP_SPECIALIZE": True, "BK": 256,
    }, attrs={(i,): [("tt.divisibility", 16)] for i in range(5)})
    kernel = compile_to_llvm(source, target=TARGET, options={
        "num_warps": 4, "num_stages": 2, "mmac_layout_force": 2,
        "use_async_copy": False, "wasp_partition_warps": (4, 4, 4, 4),
        "wasp_wdra": False,
    })
    check_scaled_kernel(kernel, has_scale=side != "none")
    # A's reselected M32 MLS tile must also constrain the MMAC tile.
    assert "mlsTile = [32, 128]" in kernel.asm["ttgir"]
    assert "tilesPerWarp = [2, 2]" in kernel.asm["ttgir"]
    assert "tilesPerWarp = [4, 2]" not in kernel.asm["ttgir"]


@pytest.mark.parametrize("a_format,b_format", [
    ("e2m1", "e2m1"), ("e2m1", "e4m3"), ("e4m3", "e2m1"),
    ("e4m3", "e4m3"), ("e5m2", "e5m2"),
])
@pytest.mark.parametrize("a_mls,b_mls,k", [
    (True, True, 128), (False, False, 32), (False, False, 64),
    (False, False, 128), (False, False, 256),
    (True, False, 128), (False, True, 128),
])
def test_scaled_compile_formats(a_format, b_format, a_mls, b_mls, k):
    types = {"e2m1": "*u8", "e4m3": "*fp8e4nv", "e5m2": "*fp8e5"}
    source = ASTSource(format_kernel, signature={
        **SIGNATURE, "A": types[a_format], "B": types[b_format],
    }, constexprs={
        "AF": a_format, "BF": b_format, "K": k, "A_MLS": a_mls, "B_MLS": b_mls,
    })
    kernel = compile_to_llvm(source, target=TARGET, options={"num_warps": 4, "mmac_layout_force": 2})
    check_scaled_kernel(kernel)


def test_scaled_compile_reuse():
    slots = []
    for dots in (1, 9):
        source = ASTSource(reuse_kernel, signature=SIGNATURE, constexprs={"DOTS": dots})
        kernel = compile_to_llvm(source, target=TARGET, options={"num_warps": 4, "mmac_layout_force": 2})
        check_scaled_kernel(kernel)
        match = re.search(r'"hcu-scale-buffer-size"="(\d+)"', kernel.asm["llir"])
        assert match, "missing backend Scale Buffer reservation"
        slots.append(int(match[1]))
    assert slots[0] == slots[1], "non-overlapping dot lifetimes must reuse scale rows"


@pytest.mark.parametrize("warps,stages", [
    ((4, 4), 2), ((4, 4), 4), ((4, 4, 4), 2),
    ((4, 4, 4), 4), ((4, 4, 4, 4), 2),
])
@pytest.mark.parametrize("mls", [False, True])
@pytest.mark.parametrize("bk", [128, 256])
def test_scaled_compile_wasp_shares_matrix_barriers(warps, stages, mls, bk):
    counts = []
    for side in ("none", "a", "b", "both"):
        source = ASTSource(optional_scale_kernel, signature=SIGNATURE, constexprs={
            "M": 128, "N": 128, "MLS": mls, "SIDE": side,
            "R_CONTIGUOUS": False, "STAGES": stages, "TILES": 2 * stages,
            "WARP_SPECIALIZE": True, "BK": bk,
        }, attrs={(i,): [("tt.divisibility", 16)] for i in range(5)})
        kernel = compile_to_llvm(source, target=TARGET, options={
            "num_warps": 4, "num_stages": stages, "mmac_layout_force": 2,
            "wasp_partition_warps": warps, "wasp_wdra": False,
        })
        check_scaled_kernel(kernel, has_scale=side != "none")
        counts.append(kernel.asm["ttgir"].count("rocdl.hcu.s.abarrier.init"))
    assert 0 < counts[0] <= 16
    assert counts == [counts[0]] * 4, "scale loads must not reserve additional abarrier IDs"


@pytest.mark.parametrize('mls', [False, True])
def test_scaled_copy_m0_export(mls):
    # Compilation-only: exercise LLVM export even without a gfx946 device.
    target = GPUTarget("hip", "gfx946", 64)
    source = ASTSource(scaled_copy_kernel,
                       signature={'A': '*u8', 'B': '*u8', 'SA': '*u8', 'SB': '*u8', 'C': '*fp32'},
                       constexprs={'MLS': mls})
    compiled = compile_to_llvm(source, target=target,
                              options={'num_warps': 4, 'num_stages': 1, 'mmac_layout_force': 2})
    # The complete scale tile stays on the matrix op until LLVM lowering.
    # Export must consume its temporary named-register carrier completely.
    llir = compiled.asm['llir']
    assert '"amdg.m0"' not in llir
    calls = [line.strip() for line in llir.splitlines() if line.strip() and not line.lstrip().startswith(';')]
    scaled_calls = [i for i, line in enumerate(calls) if 'call ' in line and '@llvm.hcu.mmac.scale.' in line]
    assert len(scaled_calls) > 1, 'Expected a whole tile to expand into multiple MMACs'
    for i in scaled_calls:
        assert '@llvm.write_register.i32' in calls[i - 1], calls[i]


@pytest.mark.parametrize("memory_space", ["#ttg.shared_memory", "1"])
def test_scale_write_requires_lds(tmp_path, memory_space):
    from triton._C.libtriton import ir, hcu

    context = ir.context()
    ir.load_dialects(context)
    hcu.load_dialects(context)
    source = f'''#blocked = #ttg.blocked<{{sizePerThread=[1,1], threadsPerWarp=[16,4],
                                        warpsPerCTA=[4,1], order=[1,0]}}>
    #shared = #ttg.swizzled_shared<{{vec=4,perPhase=1,maxPhase=1,order=[1,0]}}>
    module {{
      tt.func @stage(%src: !ttg.memdesc<16x2xi8, #shared, {memory_space}>) {{
        %scale = amdg.scale_write %src {{mode = 0 : i32}} : !ttg.memdesc<16x2xi8, #shared, {memory_space}>
          -> !ttg.memdesc<16x2xi8, #amdg.scale<{{blockShape = [32, 2], rowStride = 8}}>, #amdg.scale_memory>
        tt.return
      }}
    }}'''
    path = tmp_path / "stage-source.mlir"
    path.write_text(source)
    if memory_space == "#ttg.shared_memory":
        assert "amdg.scale_write" in str(ir.parse_mlir_module(str(path), context))
    else:
        with pytest.raises(RuntimeError):
            ir.parse_mlir_module(str(path), context)


@pytest.mark.parametrize("r_contiguous", [False, True])
@pytest.mark.parametrize("mls,async_copy,mask,range_proof", [(False, False, False, True), (True, False, False, True),
                                                             (False, True, False, True), (True, False, True, True),
                                                             (True, False, False, False), (False, True, False, False)])
def test_scale_chain_producer_policy(r_contiguous, mls, async_copy, mask, range_proof):
    target = TARGET
    source = ASTSource(
        chain_kernel, signature={"A": "*u8", "B": "*u8", "SA": "*u8", "SB": "*u8", "C": "*fp32", "Bound":
                                 "i32"}, constexprs={"MLS": mls, "R_CONTIGUOUS": r_contiguous, "MASK": mask},
        attrs={(i, ): [("tt.divisibility", 16)] + ([("tt.pointer_range", 32)] if range_proof else [])
               for i in range(5)})
    compiled = compile_to_llvm(
        source, target=target,
        options={"num_warps": 4, "num_stages": 1, "mmac_layout_force": 2, "use_async_copy": async_copy})
    ttgir = compiled.asm["ttgir"]
    assert "amdg.dot_scaled" in ttgir
    assert "tt.dot_scaled" not in ttgir
    scale_native = [line for line in ttgir.splitlines() if "buffer_load_to_local" in line and "64x8xi8" in line]
    native = (mls or async_copy) and not mask
    assert len(scale_native) == (2 if native and range_proof else 0)
    scale_global = [line for line in ttgir.splitlines() if "async_copy_global_to_local" in line and "64x8xi8" in line]
    assert len(scale_global) == (2 if native and not range_proof else 0)
    for attribute in ("scale.producer", "scale.loader_vector", "scale.chain_copy_plan"):
        assert attribute not in ttgir
    allocs = [line for line in ttgir.splitlines()
              if "ttg.local_alloc" in line and "!ttg.memdesc<64x8xi8," in line]
    assert len(allocs) == 2
    for line in allocs:
        # Native transport allocates an empty destination; REG initializes LDS.
        assert bool(re.search(r"ttg.local_alloc %", line)) == (not native)
        shared = re.search(r'64x8xi8, (#[\w]+),', line)[1]
        definition = next(l for l in ttgir.splitlines() if l.startswith(shared + " ="))
        order = "0, 1" if native and r_contiguous else "1, 0"
        assert f"order = [{order}]" in definition
    assert "#amdg.scale<" in ttgir
    if not mask:
        loads = scale_native + scale_global if native else [
            line for line in ttgir.splitlines()
            if ("tt.load " in line or "amdg.buffer_load " in line)
            and "tensor<64x8xi8," in line]
        width = 8 if native else 2
        shape = f"[{width}, 1]" if r_contiguous else f"[1, {width}]"
        for line in loads:
            layouts = [definition for definition in ttgir.splitlines()
                       if " = #ttg.blocked<" in definition
                       and definition.split(" =", 1)[0] in re.findall(r"#[\w]+", line)]
            assert any(f"sizePerThread = {shape}" in layout for layout in layouts)
    # Pointer-layout conversions are permitted.
    # Numeric E8M0 conversions are not.
    assert not any("ttg.convert_layout" in line and "64x8xi8" in line for line in ttgir.splitlines())


def test_scale_regions_reuse_existing_gpu_barrier():
    target = TARGET
    slots = []
    for dots in (1, 9):
        source = ASTSource(reuse_kernel, signature={"A": "*u8", "B": "*u8", "SA": "*u8", "SB": "*u8", "C": "*fp32"},
                           constexprs={"DOTS": dots})
        compiled = compile_to_llvm(source, target=target,
                                  options={"num_warps": 4, "num_stages": 1, "mmac_layout_force": 2})
        match = re.search(r'"hcu-scale-buffer-size"="(\d+)"', compiled.asm["llir"])
        assert match, "missing scale-buffer allocation metadata"
        slots.append(int(match[1]))
    assert slots[1] == slots[0], "completed dots must reuse their scale regions"


@pytest.mark.parametrize("mode", ["ordinary", "mls", "async"])
@pytest.mark.parametrize("pitch,mask,range_proof", [(2, False, True), (4, False, True), (2, True, True),
                                                    (2, False, False)])
@pytest.mark.parametrize("r_contiguous", [False, True])
def test_small_bk_uses_register_path(mode, pitch, mask, range_proof, r_contiguous):
    target = TARGET
    source = ASTSource(
        small_bk_kernel, signature={"A": "*u8", "B": "*u8", "SA": "*u8", "SB": "*u8", "C": "*fp32", "Bound": "i32"},
        constexprs={"MLS": mode == "mls", "PITCH": pitch, "MASK": mask, "R_CONTIGUOUS": r_contiguous},
        attrs={(i, ): [("tt.divisibility", 16)] + ([("tt.pointer_range", 32)] if range_proof else [])
               for i in range(5)})
    compiled = compile_to_llvm(
        source, target=target,
        options={"num_warps": 4, "num_stages": 1, "mmac_layout_force": 2, "use_async_copy": mode == "async"})
    native = [
        line for line in compiled.asm["ttgir"].splitlines() if "buffer_load_to_local" in line and "128x2xi8" in line
    ]
    # BK/32=2 must never produce a native scale copy, even for compact
    # s-contiguous rows or R-contiguous dword candidates in MLS/async mode.
    assert not native
    assert len([
        line for line in compiled.asm["ttgir"].splitlines()
        if ("ttg.local_store" in line or "ttg.local_alloc %" in line) and "128x2xi8" in line
    ]) == 2
    assert compiled.asm["llir"].count("call void @llvm.hcu.ds.scale.copy.ds2buf") > 0
    assert "llvm.amdgcn.global.load" not in compiled.asm["llir"]
    assert not any("ttg.convert_layout" in line and "128x2xi8" in line for line in compiled.asm["ttgir"].splitlines())


def test_native_scale_does_not_require_pointer_range():
    target = TARGET
    source = ASTSource(wide_scale_offsets_kernel,
                       signature={"A": "*u8", "B": "*u8", "SA": "*u8", "SB": "*u8", "C":
                                  "*fp32"}, attrs={(i, ): [("tt.divisibility", 16)]
                                                   for i in range(5)})
    compiled = compile_to_llvm(source, target=target, options={"num_warps": 4, "num_stages": 1, "mmac_layout_force": 2})
    assert "tt.pointer_range" not in compiled.asm["ttgir"]
    assert len([
        line for line in compiled.asm["ttgir"].splitlines()
        if "async_copy_global_to_local" in line and "64x8xi8" in line
    ]) == 2
    assert "llvm.hcu.global.load.dwordx2.lds" in compiled.asm["llir"]
    assert "llvm.amdgcn.global.load" not in compiled.asm["llir"]


def test_scale_descriptor_owns_layout_and_base(tmp_path):
    from pathlib import Path
    from buffer_lds_test_utils import run_pass
    from triton._C.libtriton import hcu

    root = Path(__file__).resolve().parents[3]
    source = (root / "test/Conversion/hcu/scaled-dot-lowering.mlir").read_text().split("// -----", 1)[0]
    result = run_pass(tmp_path, source, hcu.passes.ttgpuir.add_allocate_scale_buffer)
    assert "#amdg.scale_memory" in result
    assert "blockShape = [32, 8], rowStride = 16" in result
    assert "row_base = 0 : i32" in result
    assert "row_base = 32 : i32" in result
    assert "hcu.scale_buffer = 4 : i32" in result
    assert "copy_plan" not in result
    dot = next(line for line in result.splitlines() if "amdg.dot_scaled" in line)
    assert "row_base" not in dot and "scale_base" not in dot


# Gluon keeps operand layouts explicit and reuses the same late scale lowering.
@gluon.jit
def gluon_scaled_kernel(A, B, SA, SB, C, A_FORMAT: gl.constexpr, B_FORMAT: gl.constexpr,
                        SIDE: gl.constexpr, TILES: gl.constexpr = 1):
    K: gl.constexpr = 256
    AK: gl.constexpr = K // 2 if A_FORMAT == "e2m1" else K
    BK: gl.constexpr = K // 2 if B_FORMAT == "e2m1" else K
    load_layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
    IK: gl.constexpr = 64 if A_FORMAT == "e2m1" and B_FORMAT == "e2m1" else 32
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, IK], transposed=True, warps_per_cta=[2, 2])
    r = gl.arange(0, 64, layout=gl.SliceLayout(1, load_layout))
    ak = gl.arange(0, AK, layout=gl.SliceLayout(0, load_layout))
    bk = gl.arange(0, BK, layout=gl.SliceLayout(0, load_layout))
    sk = gl.arange(0, K // 32, layout=gl.SliceLayout(0, load_layout))
    acc = gl.full((64, 64), 1, gl.float32, mma)
    for i in range(TILES):
        a = gl.load(A + r[:, None] * (TILES * AK) + i * AK + ak[None, :])
        b = gl.load(B + r[:, None] * (TILES * BK) + i * BK + bk[None, :])
        a = gl.convert_layout(a, gl.DotOperandLayout(0, mma, IK // (8 if A_FORMAT == "e2m1" else 4)))
        b = gl.convert_layout(b.T, gl.DotOperandLayout(1, mma, IK // (8 if B_FORMAT == "e2m1" else 4)))
        sa = None
        sb = None
        if SIDE == "a" or SIDE == "both":
            sa = gl.load(SA + r[:, None] * (TILES * K // 32) + i * (K // 32) + sk[None, :])
        if SIDE == "b" or SIDE == "both":
            sb = gl.load(SB + r[:, None] * (TILES * K // 32) + i * (K // 32) + sk[None, :])
        if SIDE == "scalar":
            sa = 0x7f
            sb = 0x80
        acc = gl.amd.cdna4.mfma_scaled(a, sa, A_FORMAT, b, sb, B_FORMAT, acc)
    output = gl.convert_layout(acc, load_layout)
    c = gl.arange(0, 64, layout=gl.SliceLayout(0, load_layout))
    gl.store(C + r[:, None] * 64 + c[None, :], output)


def compile_gluon_scaled(a_format="e2m1", b_format="e2m1", side="both", tiles=1, async_copy=False,
                         compiler=compile_to_llvm):
    types = {"e2m1": "*u8", "e4m3": "*fp8e4nv", "e5m2": "*fp8e5"}
    source = GluonASTSource(gluon_scaled_kernel, signature={
        **SIGNATURE, "A": types[a_format], "B": types[b_format],
    }, constexprs={
        "A_FORMAT": a_format, "B_FORMAT": b_format, "SIDE": side, "TILES": tiles,
    }, attrs={(i,): [("tt.divisibility", 16)] for i in range(5)})
    return compiler(source, target=TARGET, options={"num_warps": 4, "use_async_copy": async_copy})


@pytest.mark.parametrize("a_format,b_format", [
    ("e2m1", "e2m1"), ("e2m1", "e4m3"), ("e4m3", "e2m1"),
    ("e4m3", "e4m3"), ("e5m2", "e5m2"),
])
@pytest.mark.parametrize("side", ["none", "a", "b", "both"])
@pytest.mark.parametrize("async_copy", [False, True])
def test_gluon_scaled_compile(a_format, b_format, side, async_copy):
    compiled = compile_gluon_scaled(a_format, b_format, side, tiles=2, async_copy=async_copy)
    # The existing Gluon helper expands None into unit E8M0 tensors.
    check_scaled_kernel(compiled)
    if async_copy and side != "none":
        assert "ttg.async_copy_global_to_local" in compiled.asm["ttgir"]
        assert "llvm.hcu.global.load." in compiled.asm["llir"]



def test_gluon_scaled_scalar_compile():
    compiled = compile_gluon_scaled(side="scalar", tiles=2)
    check_scaled_kernel(compiled)


# Unmodified upstream kernels; signatures preserve their scale types and layouts.
FP8, U8, BF16, F32, I32 = "*fp8e4nv", "*u8", "*bf16", "*fp32", "*i32"


def upstream_scaled_cases():
    entries = [
        ("vllm_linear", "_mxfp8_linear_kernel", "linear", {}),
        ("vllm_block32", "_block32_tiled_kernel", "block32", {}),
        ("vllm_block32", "_block32_packed_kernel", "packed", {}),
        ("vllm_block32", "_block32_packed_kernel", "packed-fused", {"FUSED_SPLITS": 2}),
        ("vllm_moe", "_mxfp8_grouped_gemm_kernel", "moe", {}),
        ("vllm_mla", "_mxfp8_wo_a_bmm_kernel", "mla-row", {"SCALE_ROWS": 1}),
        ("vllm_mla", "_mxfp8_wo_a_bmm_kernel", "mla-block32", {"SCALE_ROWS": 32}),
        ("vllm_attention", "kernel_ultraquant_unified_attention_2d", "attention2d-d128", {}),
        ("vllm_attention", "kernel_ultraquant_unified_attention_3d", "attention3d-d128", {}),
        ("vllm_attention", "kernel_ultraquant_unified_attention_2d", "attention2d-d64", {"D": 64}),
        ("sglang_linear", "_mxfp8_linear_kernel", "linear", {}),
        ("sglang_shuffled", "_mxfp8_shuffled_gemm_kernel", "shuffled", {}),
        ("sglang_shuffled", "_mxfp8_shuffled_gemm_kernel", "shuffled-tail", {"K": 992, "K_TAIL": True}),
        ("sglang_moe", "_mxfp8_grouped_gemm_kernel", "moe", {}),
    ]
    return [dict(id=f"{module}-{variant}-async{int(direct)}", module=module,
                 function=function, variant=variant, overrides=overrides, direct=direct)
            for module, function, variant, overrides in entries for direct in (False, True)]


def upstream_scaled_arguments(case, fn):
    """Explicit pointer dtypes and launcher-like specializations; runtime K stays runtime."""
    values = dict(M=64, N=128, K=1024, E=8, EM=256, num_valid_tokens=128, top_k=2,
                  num_tokens=64, R=64, SCALE_ROWS=1, BLOCK_M=32, BLOCK_N=64,
                  BLOCK_K=256, K_PACK=2, K_PER_SPLIT=512, N_FIRST=True, EVEN_K=True,
                  FUSED_SPLITS=1, A_DIV=2, MUL_WEIGHT=True, GROUP_SIZE_M=4,
                  k_per_split=512, K_PAD=1024, OUT_F32=False, K_TAIL=False)
    values.update(case["overrides"])
    if case["module"].endswith("moe"):
        values.update(BLOCK_N=128)
    if case["module"] == "sglang_shuffled":
        values.update(BLOCK_N=64)
    if "attention" in case["module"]:
        d = case["overrides"].get("D", 128)
        slot = d + 2 * (d // 32)
        values.update(num_query_heads=8, num_queries_per_kv=8, block_table_stride=64,
                      query_stride_0=8*d, query_stride_1=d,
                      output_stride_0=8*d, output_stride_1=d,
                      stride_cache_block=16*slot, stride_cache_pos=slot,
                      stride_cache_head=slot, BLOCK_SIZE=16, TILE_SIZE=16,
                      HEAD_SIZE=d, HEAD_SIZE_PADDED=d, BLOCK_Q=2, BLOCK_M=16,
                      num_seqs=4, NUM_SEGMENTS_PER_SEQ=16, K_SCALES_OFFSET=d//2,
                      V_CODES_OFFSET=d//2+d//32, V_SCALES_OFFSET=d+d//32,
                      GROUP_SIZE_C=32, N_GROUPS_C=d//32, UE8M0_BIAS_C=127,
                      USE_SINKS=0, SLIDING_WINDOW=0, scale=0.08838834764831845)
    k, n = values["K"], values["N"]
    values.update(stride_xm=k, stride_xk=1, stride_xsm=k//32, stride_xsk=1,
                  stride_wn=k, stride_wk=1, stride_wsn=k//32, stride_wsk=1,
                  stride_om=n, stride_on=1, stride_am=k, stride_ak=1,
                  stride_asm=k//32, stride_ask=1, stride_be=n*k, stride_bn=k,
                  stride_bk=1, stride_bse=n*(k//32), stride_bsn=k//32,
                  stride_bsk=1, stride_cm=n, stride_cn=1,
                  stride_at=4*k, stride_ast=4*(k//32), stride_out=4*values["R"])
    pointers = dict(x_ptr=FP8, xs_ptr=U8, w_ptr=FP8, ws_ptr=U8, out_ptr=BF16,
                    a_ptr=FP8, as_ptr=U8, a_scale_ptr=U8, b_ptr=FP8,
                    b_scale_ptr=U8, c_ptr=BF16, slot_ptr=F32, count_ptr=I32,
                    topk_weights_ptr=F32, sorted_token_ids_ptr=I32,
                    expert_ids_ptr=I32, num_tokens_post_padded_ptr=I32,
                    output_ptr=BF16, query_ptr=FP8, KV_cache_ptr=U8,
                    block_tables_ptr=I32, seq_lens_ptr=I32, query_start_len_ptr=I32,
                    sinks_ptr=F32, segm_output_ptr=F32, segm_max_ptr=F32,
                    segm_expsum_ptr=F32)
    if case["module"] == "sglang_shuffled":
        # The upstream launcher deliberately passes the FP8 values through a uint8 view.
        pointers.update(x_ptr=U8, w_ptr=U8)
        if values["OUT_F32"]:
            pointers["out_ptr"] = F32
    constants, signature, attrs = {}, {}, {}
    params = inspect.signature(fn.fn).parameters
    for i, name in enumerate(fn.arg_names):
        if i in fn.constexprs:
            constants[name] = values[name] if name in values else params[name].default
            assert constants[name] is not inspect.Parameter.empty, name
        elif name in pointers:
            signature[name] = pointers[name]
            if not fn.params[i].do_not_specialize_on_alignment:
                attrs[(i,)] = [("tt.divisibility", 16)]
        else:
            assert name in values, (case["id"], name)
            value = values[name]
            if name.startswith("stride_") and value == 1:
                constants[name] = 1
            else:
                annotation = str(params[name].annotation)
                signature[name] = "fp32" if isinstance(value, float) else (
                    "i64" if "int64" in annotation else "i32")
                if (isinstance(value, int) and value % 16 == 0
                        and not fn.params[i].do_not_specialize
                        and not fn.params[i].do_not_specialize_on_alignment):
                    attrs[(i,)] = [("tt.divisibility", 16)]
    return signature, constants, attrs, values



UPSTREAM_SCALED_CASES = upstream_scaled_cases()


@pytest.mark.parametrize("case", UPSTREAM_SCALED_CASES, ids=[c["id"] for c in UPSTREAM_SCALED_CASES])
def test_upstream_scaled_compile(case, tmp_path):
    name = f"{case['module']}_{case['function'].lstrip('_')}"
    fn = getattr(upstream_kernels, name)
    signature, constants, attrs, _ = upstream_scaled_arguments(case, fn)
    attention = "attention" in case["module"]
    stages = (3 if "3d" in case["variant"] else 1) if attention else 2
    warps = (2 if "3d" in case["variant"] else 4) if attention else (
        8 if case["module"].endswith("moe") else 4)
    compiled = compile_to_llvm(
        ASTSource(fn, signature=signature, constexprs=constants, attrs=attrs), target=TARGET,
        options={"num_warps": warps, "num_stages": stages,
                 "use_async_copy": case["direct"], "mmac_layout_force": 2})
    for kind in ("ttir", "ttgir", "llir"):
        content = compiled.asm[kind]
        (tmp_path / f"kernel.{kind}").write_bytes(content if isinstance(content, bytes) else content.encode())
    assert "tt.dot_scaled" in compiled.asm["ttir"]
    assert "llvm.hcu.mmac.scale." in compiled.asm["llir"]
