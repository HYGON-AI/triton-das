"""Explicit HCU MMAC layouts: identity, round trips and target legality."""
import pytest
import torch
import triton
from triton.experimental.gluon.language._core import builtin, _unwrap_if_constexpr
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import MmacLayout


def require_hcu():
    if (
        not torch.cuda.is_available()
        or triton.runtime.driver.active.get_current_target().arch != "gfx946"
    ):
        pytest.skip("requires gfx946")


def test_layout_identity():
    args = dict(version=4, instr_shape=[16, 16, 32], warps_per_cta=[1, 4])
    a = gl.amd.AMDMFMALayout(**args, mmac_layout=MmacLayout.LTS)
    b = gl.amd.AMDMFMALayout(**args, mmac_layout=MmacLayout.LIT)
    assert a != b and a.mangle() != b.mangle()
    assert len({a, b}) == 2
    with pytest.raises(TypeError, match="transposed"):
        gl.amd.AMDMFMALayout(**args, transposed=True)
    with pytest.raises(TypeError, match="mmac_layout"):
        gl.amd.AMDMFMALayout(**args)
    for invalid in (0, 5):
        with pytest.raises(ValueError):
            gl.amd.AMDMFMALayout(**args, mmac_layout=invalid)
    for invalid in (True, False):
        with pytest.raises(TypeError, match="boolean"):
            gl.amd.AMDMFMALayout(4, [16, 16, 32], invalid, [1, 4])


@builtin
def check_mmac_layout_roundtrip(value, expected, _semantic=None):
    expected = _unwrap_if_constexpr(expected)
    restored = _semantic.builder.get_gluon_layout_from_tensor(value.handle)
    assert int(restored.mmac_layout) == expected
    assert restored == value.type.layout
    original = value.type.to_ir(_semantic.builder)
    roundtrip = gl.distributed_type(value.dtype, value.shape, restored).to_ir(_semantic.builder)
    assert str(roundtrip) == str(original)
    return value


@g.jit
def mmac_layout_roundtrip_kernel(OUT, MODE: gl.constexpr):
    f: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], mmac_layout=MODE,
        warps_per_cta=[1, 4], tiles_per_warp=[4, 4],
    )
    value = gl.full((64, 256), 0, gl.float32, f)
    check_mmac_layout_roundtrip(value, MODE)
    gl.store(OUT, 0)


@pytest.mark.parametrize("mode", [1, 2, 3, 4])
def test_mmac_layout_roundtrip(mode):
    require_hcu()
    mmac_layout_roundtrip_kernel.warmup(
        torch.empty((), device="cuda"), mode, grid=(1,), num_warps=4,
    )


@g.jit
def mmac_add_one(x):
    return x + 1


@g.jit
def mmac_slice_broadcast_kernel(Out, MODE: gl.constexpr,
                                REDUCE: gl.constexpr = False, OTHER_MODE: gl.constexpr = 0):
    f: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], mmac_layout=MODE,
        warps_per_cta=[1, 4], tiles_per_warp=[4, 2],
    )
    row = gl.arange(0, 64, layout=gl.SliceLayout(1, f))
    col = gl.arange(0, 128, layout=gl.SliceLayout(0, f))
    acc = gl.full((64, 128), 1, gl.float32, f)
    acc *= (row + 1)[:, None] * (col + 1)[None, :]
    if REDUCE:
        acc += gl.sum(acc, 1)[:, None]
    if OTHER_MODE:
        other: gl.constexpr = gl.amd.AMDMFMALayout(
            version=4, instr_shape=[16, 16, 32], mmac_layout=OTHER_MODE,
            warps_per_cta=[1, 4], tiles_per_warp=[4, 2],
        )
        acc += gl.full((64, 128), 1, gl.float32, other)
        for _ in range(2):
            acc = mmac_add_one(acc)
    gl.store(Out + row[:, None] * 128 + col[None, :], acc)


@pytest.mark.parametrize("mode", [1, 2, 3, 4])
def test_mmac_slice_broadcast(mode):
    require_hcu()
    out = torch.empty((64, 128), device="cuda", dtype=torch.float32)
    mmac_slice_broadcast_kernel[(1,)](out, mode, num_warps=4)
    ref = (torch.arange(64) + 1)[:, None] * (torch.arange(128) + 1)[None, :]
    torch.testing.assert_close(out.cpu(), ref.float(), atol=0, rtol=0)


@pytest.mark.parametrize("mode", [1, 2, 3, 4])
def test_mmac_same_mode_and_helper(mode):
    require_hcu()
    out = torch.empty((64, 128), device="cuda", dtype=torch.float32)
    mmac_slice_broadcast_kernel[(1,)](out, mode, OTHER_MODE=mode, num_warps=4)
    ref = (torch.arange(64) + 1)[:, None] * (torch.arange(128) + 1)[None, :] + 3
    torch.testing.assert_close(out.cpu(), ref.float(), atol=0, rtol=0)


@pytest.mark.parametrize("arch", ["gfx936", "gfx938", "gfx946", "gfx948", "gfx950"])
@pytest.mark.parametrize("mode", [1, 2, 3, 4])
def test_mmac_layout_cross_target_roundtrip(arch, mode):
    from triton.backends.compiler import GPUTarget
    from triton.experimental.gluon._runtime import GluonASTSource
    source = GluonASTSource(
        mmac_layout_roundtrip_kernel,
        signature={"OUT": "*fp32", "MODE": "constexpr"},
        constexprs={"MODE": mode},
    )
    if arch == "gfx950" or (mode != 1 and arch == "gfx936"):
        with pytest.raises(triton.CompilationError, match="HCU target|does not support"):
            triton.compile(source, target=GPUTarget("hip", arch, 64), options={"num_warps": 4})
    else:
        triton.compile(source, target=GPUTarget("hip", arch, 64), options={"num_warps": 4})


@pytest.mark.parametrize("mode", [1, 2, 3, 4])
def test_mmac_reduction_layout_roundtrip(mode):
    require_hcu()
    # Frontend inference only; the separate pre-existing LIT numerical issue remains.
    mmac_slice_broadcast_kernel.warmup(
        torch.empty((64, 128), device="cuda"), mode, True,
        grid=(1,), num_warps=4,
    )


def test_mmac_rejects_distinct_modes():
    require_hcu()
    with pytest.raises(triton.CompilationError, match="Layout mismatch in broadcast"):
        mmac_slice_broadcast_kernel.warmup(
            torch.empty((64, 128), device="cuda"), 3, OTHER_MODE=2,
            grid=(1,), num_warps=4,
        )
