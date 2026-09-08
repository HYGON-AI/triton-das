"""Regressions for overlapping MLS with ordinary buffer loads on HCU.

Run with the HCU Triton package and its configured AICC:
  python -m pytest third_party/hcu/test/test_asyncmark_waitcnt.py -v
"""

import re

import pytest
import torch
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.backends.hcu.compiler import HIPBackend
from triton.compiler import ASTSource


_target = triton.runtime.driver.active.get_current_target() if torch.cuda.is_available() else None
pytestmark = pytest.mark.skipif(
    _target is None or _target.backend not in ("hip", "hcu") or _target.arch != "gfx938",
    reason="These MLS layout/ISA regressions require gfx938",
)


@triton.jit
def _mls_sync_kernel(A, B, S0, S1, S2, C, K: tl.constexpr):
    rows = tl.arange(0, 32)
    cols = tl.arange(0, 32)
    acc = tl.full((32, 32), 0, tl.float32)
    for k in range(tl.cdiv(K, 64)):
        a = tl.matrix_load(A, shape=[32, K], strides=[K, 1],
                           block_shape=[32, 64], offsets=[0, k * 64])
        b = tl.matrix_load(B, shape=[K, 32], strides=[1, K],
                           block_shape=[64, 32], offsets=[k * 64, 0])
        s0 = tl.load(S0 + k * 32 + rows)
        s1 = tl.load(S1 + k * 32 + cols)
        s2 = tl.load(S2 + k * 32 + cols)
        product = tl.dot(a, b)
        acc += product * s0[:, None] * s1[None, :] + s2[None, :]
    tl.store(C + rows[:, None] * 32 + cols[None, :], acc)


@pytest.mark.parametrize("use_async_copy", [0, 1])
def test_mls_sync_overlap_waitcnt(tmp_path, monkeypatch, use_async_copy):
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    monkeypatch.setenv("TRITON_HIP_USE_ASYNC_COPY", str(use_async_copy))
    kernel = triton.compile(
        ASTSource(_mls_sync_kernel,
                  signature={"A": "*fp16", "B": "*fp16", "S0": "*fp32",
                             "S1": "*fp32", "S2": "*fp32", "C": "*fp32"},
                  constexprs={"K": 512}),
        target=GPUTarget("hip", "gfx938", 64),
        options={"num_warps": 4, "num_stages": 2, "use_block_pingpong": False},
    )
    for stage in ("ttgir", "llir", "amdgcn"):
        (tmp_path / f"mls_sync_overlap.{stage}").write_text(kernel.asm[stage])
    llir = kernel.asm["llir"]
    asm = kernel.asm["amdgcn"]
    ttgir = kernel.asm["ttgir"]
    assert "amdg.matrix_load_to_local" in ttgir
    assert "amdg.buffer_load " in ttgir
    assert "amdg.buffer_load_to_local" not in ttgir
    assert "ttg.async_copy_global_to_local" not in ttgir
    assert "llvm.hcu.matrix.load" in llir
    assert "@llvm.amdgcn.asyncmark(" in llir
    assert "@llvm.amdgcn.wait.asyncmark(i16 0)" in llir
    # The row scale emits one load and each column scale emits four on this
    # layout. All nine ordinary loads must remain in flight at the MLS wait.
    _assert_partial_wait(asm, 9)


def _assert_partial_wait(asm, pending_loads):
    checked = 0
    for wait in re.finditer(r"; wait_asyncmark\(0\)", asm):
        mark = asm.rfind("; asyncmark", 0, wait.start())
        if mark < 0:
            continue
        before = asm[mark:wait.start()]
        loads = re.findall(r"^\s*buffer_load_dword(?:x\d+)?\s", before, re.MULTILINE)
        if len(loads) != pending_loads:
            continue
        read = re.search(r"^\s*ds_read\w*\s", asm[wait.end():], re.MULTILINE)
        assert read, "Missing LDS read after the MLS wait"
        after = asm[wait.end():wait.end() + read.start()]
        assert re.search(r"^\s*s_barrier\s*$", after, re.MULTILINE), after
        counts = [int(n) for n in re.findall(r"\bvmcnt\((\d+)\)", after)]
        assert counts and all(n == pending_loads for n in counts), after
        # An earlier full wait would make a later vmcnt(N) assertion vacuous.
        assert not re.search(r"\bvmcnt\(", before), before
        checked += 1
    assert checked, f"No MLS wait retained {pending_loads} ordinary buffer loads"


# The original MLS + three sync-load probe, with non-adjacent offsets to keep
# AICC from combining the three dword loads into one dwordx3 transaction.
# There is no hand-written s_waitcnt: the production code generator derives it.
LLVM_PROBE = """
target triple = "amdgcn-amd-amdhsa"
declare void @llvm.hcu.matrix.load.128x16.b8.p3(<4 x i32>, ptr addrspace(3), i32 immarg, i1 immarg, i1 immarg, i1 immarg, i1 immarg, i1 immarg)
declare void @llvm.amdgcn.asyncmark()
declare void @llvm.amdgcn.wait.asyncmark(i16 immarg)
declare void @llvm.amdgcn.wave.barrier()
declare void @llvm.amdgcn.s.barrier()
declare i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8), i32, i32, i32 immarg)
define amdgpu_kernel void @mls_three_sync_loads(<4 x i32> inreg %rsrc,
    ptr addrspace(3) inreg %lds, ptr addrspace(8) inreg %buf, ptr addrspace(1) %out) {
  call void @llvm.hcu.matrix.load.128x16.b8.p3(<4 x i32> %rsrc, ptr addrspace(3) %lds, i32 0, i1 false, i1 true, i1 false, i1 false, i1 false)
  call void @llvm.amdgcn.asyncmark()
  %lds1 = getelementptr i8, ptr addrspace(3) %lds, i32 8192
  call void @llvm.hcu.matrix.load.128x16.b8.p3(<4 x i32> %rsrc, ptr addrspace(3) %lds1, i32 32, i1 false, i1 true, i1 false, i1 false, i1 false)
  call void @llvm.amdgcn.asyncmark()
  %v0 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %buf, i32 0, i32 0, i32 0)
  %v1 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %buf, i32 1024, i32 0, i32 0)
  %v2 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %buf, i32 2048, i32 0, i32 0)
  call void @llvm.amdgcn.wave.barrier()
  call void @llvm.amdgcn.wait.asyncmark(i16 0)
  call void @llvm.amdgcn.s.barrier()
  %lv = load i32, ptr addrspace(3) %lds
  %s0 = add i32 %v0, %v1
  %s1 = add i32 %s0, %v2
  %s2 = add i32 %s1, %lv
  store i32 %s2, ptr addrspace(1) %out
  ret void
}
"""


def test_three_sync_loads_waitcnt(tmp_path):
    backend = HIPBackend(GPUTarget("hip", "gfx938", 64))
    asm = backend.make_amdgcn(LLVM_PROBE, {}, backend.parse_options({}))
    (tmp_path / "mls_three_sync_loads.amdgcn").write_text(asm)
    assert len(re.findall(r"^\s*matrix_load_", asm, re.MULTILINE)) == 2
    assert len(re.findall(r"^\s*buffer_load_dword\s", asm, re.MULTILINE)) == 3
    _assert_partial_wait(asm, 3)
    # The normal load results still need their own waits before consumption.
    assert re.search(r"ds_read_b32[\s\S]*s_waitcnt vmcnt\(0\)", asm)
    # Guard against false positives: draining at or before the partial wait
    # must fail even if a redundant vmcnt(3) is still present in the assembly.
    with pytest.raises(AssertionError):
        _assert_partial_wait(asm.replace("vmcnt(3)", "vmcnt(0)"), 3)
    with pytest.raises(AssertionError):
        _assert_partial_wait(asm.replace("; wait_asyncmark(0)",
                                        "s_waitcnt vmcnt(0)\n; wait_asyncmark(0)"), 3)


@pytest.mark.parametrize("use_async_copy", [0, 1])
@pytest.mark.parametrize("num_stages", [2, 3, 4])
def test_mls_sync_overlap_correctness(monkeypatch, use_async_copy, num_stages):
    monkeypatch.setenv("TRITON_HIP_USE_ASYNC_COPY", str(use_async_copy))
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    generator = torch.Generator().manual_seed(123)
    k_size = 512
    a_cpu = torch.randn((32, k_size), generator=generator, dtype=torch.float16)
    b_cpu = torch.randn((32, k_size), generator=generator, dtype=torch.float16)
    scales_cpu = [torch.randn((k_size // 64, 32), generator=generator) for _ in range(3)]
    expected = torch.zeros((32, 32), dtype=torch.float32)
    for k in range(k_size // 64):
        product = a_cpu[:, k * 64:(k + 1) * 64].float() @ b_cpu[:, k * 64:(k + 1) * 64].float().T
        expected += product * scales_cpu[0][k, :, None] * scales_cpu[1][k, None, :] + scales_cpu[2][k, None, :]
    a, b = a_cpu.cuda(), b_cpu.cuda()
    scales = [s.cuda() for s in scales_cpu]
    actual = torch.empty((32, 32), dtype=torch.float32, device="cuda")
    _mls_sync_kernel[(1,)](a, b, *scales, actual, k_size,
                          num_warps=4, num_stages=num_stages, use_block_pingpong=False)
    torch.testing.assert_close(actual.cpu(), expected, atol=5e-3, rtol=1e-3)
