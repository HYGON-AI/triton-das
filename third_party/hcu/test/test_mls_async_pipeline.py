"""Unified MLS/buffer-to-LDS pipeline and per-config async-copy regressions."""

import re
import pytest
import torch
import triton
import triton.language as tl
from triton._C.libtriton import amd, hcu, ir
from triton.backends.compiler import GPUTarget
from triton.backends.hcu.compiler import HIPBackend
from triton.compiler import ASTSource


_target = triton.runtime.driver.active.get_current_target() if torch.cuda.is_available() else None
pytestmark = pytest.mark.skipif(
    _target is None or _target.backend not in ("hip", "hcu") or _target.arch != "gfx938",
    reason="These MLS layout/ISA regressions require gfx938",
)


@pytest.mark.parametrize("enabled", [False, True])
def test_schedule_binding(enabled):
    context = ir.context()
    pm = ir.pass_manager(context)
    amd.passes.ttgpuir.add_schedule_loops(pm, 3)
    amd.passes.ttgpuir.add_pipeline(pm, enabled, False)
    pipeline = pm.get_pipeline_str()
    assert "num_stages=3" in pipeline
    assert pipeline.count("use_async_copy=") == 1
    assert f"use_async_copy={str(enabled).lower()}" in pipeline
    assert "enable_mls" not in pipeline
    assert not hasattr(hcu.passes.ttgpuir, "add_schedule_loops")


def test_async_copy_default(monkeypatch):
    monkeypatch.delenv("TRITON_HIP_USE_ASYNC_COPY", raising=False)
    backend = HIPBackend(GPUTarget("hip", "gfx938", 64))
    assert backend.parse_options({}).use_async_copy is False


@pytest.mark.parametrize("env_value", ["0", "1"])
def test_async_copy_option(monkeypatch, env_value):
    monkeypatch.setenv("TRITON_HIP_USE_ASYNC_COPY", env_value)
    backend = HIPBackend(GPUTarget("hip", "gfx938", 64))
    assert backend.parse_options({}).use_async_copy == (env_value == "1")
    assert backend.parse_options({"use_async_copy": None}).use_async_copy == (env_value == "1")
    hashes = []
    for enabled in (False, True):
        config = triton.Config({"use_async_copy": enabled}, num_stages=3)
        options = backend.parse_options(config.all_kwargs())
        assert options.use_async_copy is enabled
        hashes.append(options.hash())
    assert hashes[0] != hashes[1]


@triton.jit
def _mixed_kernel(A, B, S, C, K, A_MLS: tl.constexpr, B_MLS: tl.constexpr,
                  LOOP_STAGES: tl.constexpr = None):
    rows = tl.arange(0, 32)
    cols = tl.arange(0, 32)
    ks = tl.arange(0, 64)
    acc = tl.full((32, 32), 0, tl.float32)
    for start in tl.range(0, K, 64, num_stages=LOOP_STAGES):
        if A_MLS:
            a = tl.matrix_load(A, shape=[32, K], strides=[K, 1],
                               block_shape=[32, 64], offsets=[0, start],
                               boundary_check=(0, 1))
        else:
            a = tl.load(A + rows[:, None] * K + start + ks[None, :],
                        start + ks[None, :] < K, 0.0)
        if B_MLS:
            b = tl.matrix_load(B, shape=[K, 32], strides=[1, K],
                               block_shape=[64, 32], offsets=[start, 0],
                               boundary_check=(0, 1))
        else:
            b = tl.load(B + start + ks[:, None] + cols[None, :] * K,
                        start + ks[:, None] < K, 0.0)
        scale = tl.load(S + start // 64 * 32 + rows)
        acc += tl.dot(a, b) * scale[:, None]
    tl.store(C + rows[:, None] * 32 + cols[None, :], acc)


def _inputs(k, dtype=torch.float16):
    generator = torch.Generator().manual_seed(2026)
    a = torch.randn((32, k), generator=generator, dtype=torch.float16).to(dtype)
    b = torch.randn((32, k), generator=generator, dtype=torch.float16).to(dtype)
    scales = torch.randn((triton.cdiv(k, 64), 32), generator=generator)
    expected = torch.zeros((32, 32), dtype=torch.float32)
    for start in range(0, k, 64):
        expected += (a[:, start:start + 64].float() @ b[:, start:start + 64].float().T
                     * scales[start // 64, :, None])
    return a.cuda(), b.cuda(), scales.cuda(), expected


@triton.jit
def _mls_predicate_kernel(A, B, C, END, K: tl.constexpr):
    rows = tl.arange(0, 32)
    acc = tl.full((32, 32), 0, tl.float32)
    # The loop bound is independent of the matrix dimensions. Speculative
    # prefetches remain in bounds, so boundary checks cannot mask a bad pred.
    for start in range(0, END, 64):
        a = tl.matrix_load(A, shape=[32, K], strides=[K, 1],
                           block_shape=[32, 64], offsets=[0, start])
        b = tl.matrix_load(B, shape=[K, 32], strides=[1, K],
                           block_shape=[64, 32], offsets=[start, 0])
        acc += tl.dot(a, b)
    tl.store(C + rows[:, None] * 32 + rows[None, :], acc)


@pytest.mark.parametrize("stages", [2, 3, 4])
def test_mls_predicate_short_loop(stages):
    generator = torch.Generator().manual_seed(2026)
    a = torch.randn((32, 512), generator=generator, dtype=torch.float16)
    b = torch.randn((32, 512), generator=generator, dtype=torch.float16)
    a_gpu, b_gpu = a.cuda(), b.cuda()
    for end in (0, 64, 128, 192):
        out = torch.full((32, 32), float("nan"), dtype=torch.float32).cuda()
        kernel = _mls_predicate_kernel[(1,)](
            a_gpu, b_gpu, out, end, 512, num_stages=stages,
            use_async_copy=True, use_block_pingpong=False)
        assert "pred =" in kernel.asm["ttgir"]
        expected = a[:, :end].float() @ b[:, :end].float().T
        torch.testing.assert_close(out.cpu(), expected, atol=5e-3, rtol=1e-3)


@triton.jit
def _non_dot_mls_kernel(A, B, C, D, K):
    rows = tl.arange(0, 32)
    ks = tl.arange(0, 64)
    acc = tl.full((32, 32), 0, tl.float32)
    for start in range(0, K, 64):
        a = tl.load(A + rows[:, None] * K + start + ks[None, :])
        b = tl.load(B + start + ks[:, None] + rows[None, :] * K)
        extra = tl.matrix_load(A, shape=[32, K], strides=[K, 1],
                               block_shape=[32, 64], offsets=[0, start])
        tl.store(D + rows[:, None] * K + start + ks[None, :], extra)
        acc += tl.dot(a, b)
    tl.store(C + rows[:, None] * 32 + rows[None, :], acc)


def test_non_dot_mls_fallback():
    # The extra MLS is not a dot operand and is not a scheduling candidate.
    # Keep the whole loop for fallback instead of partially pipelining it.
    kernel = triton.compile(
        ASTSource(_non_dot_mls_kernel,
                  signature={"A": "*fp16", "B": "*fp16", "C": "*fp32", "D": "*fp16", "K": "i32"},
                  attrs={(i,): [("tt.divisibility", 16)] for i in range(5)}),
        target=GPUTarget("hip", "gfx938", 64),
        options={"use_async_copy": True, "num_stages": 3},
    )
    ttgir = kernel.asm["ttgir"]
    assert "scf.for" in ttgir
    assert ttgir.count("amdg.matrix_load_to_local") == 1
    assert "pred =" not in ttgir
    assert "tt.matrix_load " not in ttgir


@triton.jit
def _chained_kernel(A, B, S, C, K, SHARED_MLS: tl.constexpr):
    rows = tl.arange(0, 32)
    cols = tl.arange(0, 32)
    ks = tl.arange(0, 32)
    acc = tl.full((32, 32), 0, tl.float32)
    for start in range(0, K, 32):
        a = tl.matrix_load(A, shape=[32, K], strides=[K, 1],
                           block_shape=[32, 32], offsets=[0, start])
        b = tl.matrix_load(B, shape=[K, 32], strides=[1, K],
                           block_shape=[32, 32], offsets=[start, 0])
        product = tl.dot(a, b)
        if SHARED_MLS:
            product = tl.dot(a, b, product)
        else:
            a2 = tl.load(A + rows[:, None] * K + start + ks[None, :],
                         start + ks[None, :] < K, 0)
            b2 = tl.load(B + start + ks[:, None] + cols[None, :] * K,
                         start + ks[:, None] < K, 0)
            product = tl.dot(a2, b2, product)
        scale = tl.load(S + start // 64 * 32 + rows)
        acc += product * scale[:, None]
    tl.store(C + rows[:, None] * 32 + cols[None, :], acc)


@pytest.mark.parametrize("a_mls,b_mls", [(True, True), (True, False), (False, True), (False, False)])
@pytest.mark.parametrize("stages", [2, 3, 4])
@pytest.mark.parametrize("small_tensor", [True, False])
def test_unified_pipeline_ir(tmp_path, a_mls, b_mls, stages, small_tensor):
    attrs = {(i,): [("tt.divisibility", 16)] for i in range(5)}
    if small_tensor:
        # Match normal HCU JIT specialization for tensors within 4 GiB.
        # Without this, pointer canonicalization retains i64 offsets and the
        # copies take the global-to-LDS path instead of buffer-to-LDS.
        for i in range(4):
            attrs[(i,)].append(("tt.pointer_range", 32))
    kernel = triton.compile(
        ASTSource(_mixed_kernel,
                  signature={"A": "*fp16", "B": "*fp16", "S": "*fp32", "C": "*fp32", "K": "i32"},
                  constexprs={"A_MLS": a_mls, "B_MLS": b_mls},
                  attrs=attrs),
        target=GPUTarget("hip", "gfx938", 64),
        options={"use_async_copy": True, "num_stages": stages, "use_block_pingpong": False},
    )
    for name in ("ttgir", "llir", "amdgcn"):
        (tmp_path / f"mixed.{name}").write_text(kernel.asm[name])
    ttgir = kernel.asm["ttgir"]
    assert "tt.matrix_load " not in ttgir
    assert "ttg.pipeline.hcu_mls" not in ttgir
    assert "ttg.async_commit_group" in ttgir
    assert "ttg.async_wait" in ttgir
    if a_mls or b_mls:
        assert "amdg.matrix_load_to_local" in ttgir
        assert "pred =" in ttgir  # dynamic prologue/epilogue MLS guards
        assert "llvm.hcu.matrix.load" in kernel.asm["llir"]
    if not a_mls or not b_mls:
        if small_tensor:
            assert "amdg.buffer_load_to_local" in ttgir
            assert "ttg.async_copy_global_to_local" not in ttgir
            opcode = "buffer_load_dword"
        else:
            assert "ttg.async_copy_global_to_local" in ttgir
            assert "amdg.buffer_load_to_local" not in ttgir
            opcode = "global_load_dword"
        # The current gfx938 TargetInfo allows 32-bit direct-to-LDS loads
        # (two fp16 elements per lane), regardless of a larger tile size.
        assert re.search(rf"^\s*{opcode}\s+[^\n]*\blds\b", kernel.asm["amdgcn"], re.MULTILINE)
    assert "llvm.amdgcn.wait.asyncmark" in kernel.asm["llir"]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("a_mls,b_mls", [(True, True), (True, False), (False, True), (False, False)])
@pytest.mark.parametrize("stages", [2, 3, 4])
def test_mixed_correctness(tmp_path, monkeypatch, enabled, a_mls, b_mls, stages):
    # Explicit launch options must override the opposite environment default.
    monkeypatch.setenv("TRITON_HIP_USE_ASYNC_COPY", "0" if enabled else "1")
    # Exercise zero/short pipelines with complete MLS tiles. Partial MLS tiles
    # have a pre-existing failure (K=1 faults even without pipelining, with
    # identical code for both options); its root cause is not resolved here.
    # The legacy MLS pipeline's zero/short-loop predication is unchanged.
    sizes = [0, 64, 128, 192, 512] if enabled else [512]
    if enabled and not a_mls and not b_mls:
        sizes += [1, 63, 65]
    for k in sizes:
        print(f"K={k}, stages={stages}, MLS={a_mls}/{b_mls}, async={enabled}", flush=True)
        a, b, scales, expected = _inputs(k)
        out = torch.empty((32, 32), device="cuda", dtype=torch.float32)
        kernel = _mixed_kernel[(1,)](a, b, scales, out, k, a_mls, b_mls,
                                    use_async_copy=enabled, num_stages=stages,
                                    use_block_pingpong=False)
        assert kernel.metadata.use_async_copy is enabled
        if enabled and k >= 64 and k % 64 == 0 and (not a_mls or not b_mls):
            assert "amdg.buffer_load_to_local" in kernel.asm["ttgir"]
        for name in ("ttgir", "llir", "amdgcn"):
            (tmp_path / f"k{k}.{name}").write_text(kernel.asm[name])
        torch.testing.assert_close(out.cpu(), expected, atol=5e-3, rtol=1e-3)


@pytest.mark.parametrize("a_mls,b_mls", [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize("stages", [2, 3, 4])
def test_fp8_mls_correctness(tmp_path, a_mls, b_mls, stages):
    for k in (0, 64, 512):
        a, b, scales, expected = _inputs(k, torch.float8_e4m3fn)
        out = torch.empty((32, 32), device="cuda", dtype=torch.float32)
        kernel = _mixed_kernel[(1,)](a, b, scales, out, k, a_mls, b_mls,
                                    use_async_copy=True, num_stages=stages,
                                    use_block_pingpong=False)
        for name in ("ttgir", "llir", "amdgcn"):
            (tmp_path / f"fp8_k{k}.{name}").write_text(kernel.asm[name])
        torch.testing.assert_close(out.cpu(), expected, atol=5e-3, rtol=1e-3)


@pytest.mark.parametrize("shared_mls", [False, True])
@pytest.mark.parametrize("stages", [2, 3, 4])
def test_multidot_mls_fallback(shared_mls, stages):
    a, b, scales, _ = _inputs(512)
    out = torch.empty((32, 32), device="cuda", dtype=torch.float32)
    # Compile only: skipping the pipeline does not establish numerical support
    # for all multi-dot MLS layouts.
    kernel = _chained_kernel.warmup(
        a, b, scales, out, 512, shared_mls, grid=(1,),
        use_async_copy=True, num_stages=stages, use_block_pingpong=False)
    ttgir = kernel.asm["ttgir"]
    assert ttgir.count("scf.for") == 1
    assert ttgir.count("amdg.matrix_load_to_local") == 2
    assert "tt.matrix_load " not in ttgir
    assert "ttg.async_copy_global_to_local" not in ttgir
    assert "amdg.buffer_load_to_local" not in ttgir
    assert "pred =" not in ttgir


@pytest.mark.xfail(strict=True, raises=RuntimeError,
                   reason="Legacy shared MLS feeding two dot layouts fails MLS/MFMA element-count validation")
def test_shared_mls_two_dot_layouts(capfd, monkeypatch):
    monkeypatch.setenv("TRITON_HCU_ENABLE_DS_READ_M", "1")
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    a, b, scales, _ = _inputs(512)
    out = torch.empty((32, 32), device="cuda", dtype=torch.float32)
    # Compile only: preserve the known failure without launching invalid code.
    try:
        _chained_kernel.warmup(a, b, scales, out, 512, True, grid=(1,),
                               use_async_copy=False, num_stages=4,
                               use_block_pingpong=False)
    except RuntimeError:
        # Do not let the known-failure marker hide a different compiler error.
        assert "MLS LocalLoad elem count must match MFMA operand" in capfd.readouterr().err
        raise


def test_async_copy_autotune(monkeypatch):
    monkeypatch.setenv("TRITON_HIP_USE_ASYNC_COPY", "1")
    tuned = triton.autotune(
        configs=[triton.Config({"use_async_copy": enabled}, num_stages=2)
                 for enabled in (False, True)], key=["K"],
    )(_mixed_kernel)
    a, b, scales, expected = _inputs(512)
    out = torch.empty((32, 32), device="cuda", dtype=torch.float32)
    tuned[(1,)](a, b, scales, out, 512, True, False)
    torch.testing.assert_close(out.cpu(), expected, atol=5e-3, rtol=1e-3)
    assert {c.kwargs["use_async_copy"] for c in tuned.configs_timings} == {False, True}
    assert tuned.best_config.kwargs["use_async_copy"] in (False, True)


@pytest.mark.parametrize("a_mls,b_mls", [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize("schedule_hint", ["none", "local-prefetch"])
def test_mls_loop_stage_override(tmp_path, a_mls, b_mls, schedule_hint):
    a, b, scales, expected = _inputs(512)
    irs = []
    for kernel_stages in (1, 2):
        out = torch.empty((32, 32), device="cuda", dtype=torch.float32)
        kernel = _mixed_kernel[(1,)](
            a, b, scales, out, 512, a_mls, b_mls, LOOP_STAGES=2,
            num_stages=kernel_stages, use_async_copy=False,
            use_block_pingpong=False, schedule_hint=schedule_hint)
        for stage in ("ttgir", "amdgcn"):
            (tmp_path / f"kernel{kernel_stages}.{stage}").write_text(kernel.asm[stage])
        torch.testing.assert_close(out.cpu(), expected, atol=5e-3, rtol=1e-3)
        # Both launches must use the same legacy MLS schedule. Comparing IR
        # catches the single-buffer write-before-read race even if timing lets
        # a small numerical probe pass. Ignore source-location metadata only.
        ttgir = kernel.asm["ttgir"]
        irs.append(re.sub(r" loc\(#loc\d*\)", "", "\n".join(
            line for line in ttgir.splitlines() if not line.startswith("#loc"))))
    assert irs[0] == irs[1]
