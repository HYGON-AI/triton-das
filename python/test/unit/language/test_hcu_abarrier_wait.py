"""Native HCU abarrier probes: returned status, phase reuse and LDS ordering."""

import re

import pytest
import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon._runtime import GluonASTSource
from triton.experimental.gluon.language.amd.hcu import (
    abarrier_arrive,
    abarrier_init,
    abarrier_invalidate,
    abarrier_test_wait,
    abarrier_try_wait,
)


@g.jit
def _probe(id, phase, TRY: gl.constexpr):
    if TRY:
        return abarrier_try_wait(id, phase)
    return abarrier_test_wait(id, phase)


@g.jit
def wait_status_kernel(OUT, TRY: gl.constexpr, DYNAMIC: gl.constexpr):
    id = gl.program_id(0) + 3 if DYNAMIC else 3
    abarrier_init(3, 4)
    gl.thread_barrier()
    for phase in gl.static_range(2):
        pending_test = abarrier_test_wait(id, phase)
        gl.static_assert(pending_test.dtype == gl.int32)
        abarrier_arrive(id)
        gl.thread_barrier()
        done = _probe(id, phase, TRY)
        while done == 0:
            done = _probe(id, phase, TRY)
        other = _probe(id, phase, not TRY)
        gl.static_assert(done.dtype == gl.int32)
        gl.store(OUT + phase * 3, pending_test)
        gl.store(OUT + phase * 3 + 1, done)
        gl.store(OUT + phase * 3 + 2, other)
    gl.thread_barrier()
    abarrier_invalidate(3)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("try_wait", [False, True])
def test_abarrier_wait_compile(try_wait, dynamic):
    source = GluonASTSource(
        wait_status_kernel,
        signature={"OUT": "*i32", "TRY": "constexpr", "DYNAMIC": "constexpr"},
        constexprs={"TRY": try_wait, "DYNAMIC": dynamic},
    )
    compiled = triton.compile(source, target=GPUTarget("hip", "gfx946", 64),
                              options={"num_warps": 4, "num_stages": 1})
    for kind in ("try", "test"):
        assert re.search(r"call i32(?: \([^)]*\))? @llvm\.hcu\.s\.abarrier\."
                         + kind + r"\.wait\(", compiled.asm["llir"])
        assert f"s_abarrier_{kind}_wait" in compiled.asm["amdgcn"]
    assert "define internal" not in compiled.asm["llir"]


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("try_wait", [False, True])
def test_abarrier_wait_runtime(try_wait, dynamic):
    if (not torch.cuda.is_available()
            or triton.runtime.driver.active.get_current_target().arch != "gfx946"):
        pytest.skip("requires gfx946")
    out = torch.full((6,), -1, dtype=torch.int32, device="cuda")
    wait_status_kernel[(1,)](out, try_wait, dynamic, num_warps=4, num_stages=1)
    statuses = out.cpu().reshape(2, 3)
    assert torch.all(statuses[:, 0] == 0), statuses
    assert torch.all(statuses[:, 1:] != 0), statuses


@g.jit
def pending_test_shared_kernel(OUT):
    writer: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    reader: gl.constexpr = gl.BlockedLayout([4], [64], [4], [0])
    abarrier_init(3, 4)
    gl.thread_barrier()
    offsets = gl.arange(0, 256, writer)
    storage = gl.allocate_shared_memory(gl.int32, [256], gl.SwizzledSharedLayout(1, 1, 1, [0]))
    storage.store(offsets)
    status = abarrier_test_wait(3, 0)
    # Different lane/wave ownership requires a shared-memory fence.
    values = storage.load(reader)
    gl.store(OUT + gl.arange(0, 256, reader), values)
    gl.store(OUT + 256, status)
    abarrier_arrive(3)
    gl.thread_barrier()
    abarrier_invalidate(3)


def test_pending_test_wait_preserves_shared_dependencies():
    source = GluonASTSource(pending_test_shared_kernel, signature={"OUT": "*i32"})
    tested = triton.compile(source, target=GPUTarget("hip", "gfx946", 64),
                           options={"num_warps": 4, "num_stages": 1})
    barrier = (r"call void(?: \([^)]*\))? @llvm\."
               r"(?:amdgcn\.s\.barrier|hcu\.s\.ebarrier\.sync)\(")
    assert len(re.findall(barrier, tested.asm["llir"])) >= 3
