"""Correctness boundaries of the explicit HCU Gluon primitives."""

import pytest
import torch
import triton
from triton.experimental.gluon.language._core import builtin, _unwrap_if_constexpr
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.hcu import (
    MmacLayout,
    matrix_store,
    matrix_load,
    MmacInterleave2Layout,
    MlsSharedLayout,
    get_mmac_instruction,
    get_wave_id,
    set_priority,
    wait_lgkmcnt,
    abarrier_init,
    abarrier_arrive,
    abarrier_invalidate,
    ebarrier_arrive,
    ebarrier_sync,
)
from triton.experimental.gluon.language.amd.cdna4 import async_copy


def require_hcu():
    if (
        not torch.cuda.is_available()
        or triton.runtime.driver.active.get_current_target().arch != "gfx946"
    ):
        pytest.skip("requires gfx946")



def test_mls_layout_identity():
    assert MlsSharedLayout(0, (64, 128)) != MlsSharedLayout(0, (64, 128), alt=1)



@g.jit
def matrix_store_kernel(
    X,
    Y,
    DTYPE: gl.constexpr = gl.bfloat16,
    M: gl.constexpr = 64,
    N: gl.constexpr = 256,
    WM: gl.constexpr = 1,
    STORE_TILE: gl.constexpr = (32, 32),
    STORE_VERSION: gl.constexpr = 3,
    BOUNDARY_CHECK: gl.constexpr = (),
    OUTPUT_M: gl.constexpr = 0,
    OUTPUT_N: gl.constexpr = 0,
):
    f: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        mmac_layout=MmacLayout.LTS,
        warps_per_cta=[WM, 4 // WM],
        tiles_per_warp=[4, 4],
    )
    row = gl.arange(0, M, layout=gl.SliceLayout(1, f))
    col = gl.arange(0, N, layout=gl.SliceLayout(0, f))
    value = (
        gl.load(X + row[:, None] * N + col[None, :])
        .to(DTYPE)
        .to(gl.float16, bitcast=True)
    )
    matrix_store(
        Y.to(gl.pointer_type(gl.float16)),
        gl.convert_layout(value, MmacInterleave2Layout(f)),
        (OUTPUT_M if OUTPUT_M else M, OUTPUT_N if OUTPUT_N else N),
        (OUTPUT_N if OUTPUT_N else N, 1),
        (0, 0),
        tile=STORE_TILE,
        version=STORE_VERSION,
        boundary_check=BOUNDARY_CHECK,
    )



@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("M,N,WM", [(64, 256, 1)])
def test_matrix_store_raw_bits(dtype, M, N, WM):
    require_hcu()
    x = ((torch.arange(M * N).reshape(M, N) % 513 - 256).float() / 8).to(dtype)
    y = torch.full((M * N + 64,), -320.0, dtype=dtype).cuda()
    c = matrix_store_kernel[(1,)](
        x.float().cuda(),
        y[32:-32],
        M=M,
        N=N,
        WM=WM,
        DTYPE=gl.bfloat16 if dtype == torch.bfloat16 else gl.float16,
        num_warps=4,
    )
    actual = y.cpu()
    torch.testing.assert_close(actual[32:-32].reshape(M, N), x, atol=0, rtol=0)
    assert (actual[:32] == -320).all() and (actual[-32:] == -320).all()
    assert "matrix_store_" in c.asm["amdgcn"]
    assert "_b16" in c.asm["amdgcn"]
    assert "_b8" not in c.asm["amdgcn"]



@pytest.mark.parametrize("tile,version", [((16, 32), 2), ((32, 32), 2), ((32, 32), 3)])
def test_matrix_store_configuration(tile, version):
    require_hcu()
    x = ((torch.arange(64 * 256) % 513 - 256).float() / 8).bfloat16()
    y = torch.full((64 * 256 + 64,), -320.0, dtype=torch.bfloat16).cuda()
    matrix_store_kernel[(1,)](
        x.float().cuda(), y[32:-32], STORE_TILE=tile,
        STORE_VERSION=version, num_warps=4,
    )
    actual = y.cpu()
    torch.testing.assert_close(actual[32:-32], x, atol=0, rtol=0)
    assert (actual[:32] == -320).all() and (actual[-32:] == -320).all()



@g.jit
def mls_b_roundtrip_kernel(X, Y, ALT: gl.constexpr, BOUNDARY_CHECK: gl.constexpr = (), INPUT_K: gl.constexpr = 128, INPUT_N: gl.constexpr = 256):
    f: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], mmac_layout=MmacLayout.LTS,
        warps_per_cta=[1, 4], tiles_per_warp=[4, 4],
    )
    shared = gl.allocate_shared_memory(
        gl.float8e4nv, (128, 256),
        MlsSharedLayout(1, (128, 64), alt=ALT, order=(0, 1)),
    )
    matrix_load(shared, X, (INPUT_K, INPUT_N), (1, INPUT_K), (0, 0), boundary_check=BOUNDARY_CHECK)
    gl.inline_asm_elementwise(
        "s_waitcnt lgkmcnt(0)\n\ts_barrier\n\ts_mov_b32 $0, 0",
        constraints="=s,~{memory}", args=[], dtype=gl.int32,
        is_pure=False, pack=1,
    )
    if ALT == 1:
        value = shared.load(MmacInterleave2Layout(gl.DotOperandLayout(1, f, 8)))
    else:
        value = shared.load(gl.DotOperandLayout(1, f, 8))
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
    value = gl.convert_layout(value, layout)
    k = gl.arange(0, 128, layout=gl.SliceLayout(1, layout))
    n = gl.arange(0, 256, layout=gl.SliceLayout(0, layout))
    gl.store(Y + k[:, None] * 256 + n[None, :], value)



@pytest.mark.parametrize("alt", [0])
def test_mls_b_to_ordinary_store(alt):
    require_hcu()
    torch.manual_seed(13)
    x = (torch.randint(-16, 17, (256, 128)).float() / 8).to(torch.float8_e4m3fn)
    y = torch.empty((128, 256), device="cuda", dtype=torch.float8_e4m3fn)
    mls_b_roundtrip_kernel[(1,)](x.cuda(), y, ALT=alt, num_warps=4)
    assert torch.equal(y.cpu().view(torch.uint8), x.T.contiguous().view(torch.uint8))



@g.jit
def barrier_api_helper(COUNT: gl.constexpr):
    abarrier_init(0, 4)
    abarrier_arrive(0)
    abarrier_arrive(0, count=COUNT)
    abarrier_invalidate(0)



@g.jit
def barrier_api_kernel(COUNT: gl.constexpr):
    barrier_api_helper(COUNT)



def test_barrier_api_compile():
    require_hcu()
    compiled = barrier_api_kernel.warmup(COUNT=3, grid=(1,), num_warps=4)
    llir = compiled.asm["llir"]
    import re
    assert re.search(r"@llvm\.hcu\.s\.abarrier\.init\(i32 0, i32 4\)", llir)
    for count in (1, 3):
        assert re.search(r"@llvm\.hcu\.s\.abarrier\.arrive\(i32 0, i32 " + str(count) + r"\)", llir)
    assert "@llvm.hcu.s.abarrier.inv" in llir



@pytest.mark.parametrize("operation", ["load", "store"])
@pytest.mark.parametrize("boundary", [(), (0,), (1,), (0, 1)])
def test_matrix_boundary_forwarding(operation, boundary):
    require_hcu()
    if operation == "load":
        x = torch.empty((256, 128), dtype=torch.float8_e4m3fn, device="cuda")
        y = torch.empty_like(x)
        compiled = mls_b_roundtrip_kernel.warmup(x, y, ALT=0, BOUNDARY_CHECK=boundary, grid=(1,), num_warps=4)
        name = "amdg.matrix_load_to_local"
    else:
        x = torch.empty((64, 256), dtype=torch.bfloat16, device="cuda")
        y = torch.empty_like(x)
        compiled = matrix_store_kernel.warmup(x, y, BOUNDARY_CHECK=boundary, grid=(1,), num_warps=4)
        name = "amdg.matrix_store_from_reg"
    lines = [line for line in compiled.asm["ttgir"].splitlines() if name in line]
    assert len(lines) == 1
    if boundary:
        assert "boundaryCheck = [" + ", ".join(map(str, boundary)) + "]" in lines[0]
    else:
        assert "boundaryCheck" not in lines[0] or "boundaryCheck = []" in lines[0]



def test_matrix_store_boundary_numeric():
    require_hcu()
    x = (torch.arange(64 * 256).reshape(64, 256).float() % 127).to(torch.bfloat16)
    out = torch.full((61 * 249 + 64,), -999, dtype=torch.bfloat16, device="cuda")
    matrix_store_kernel[(1,)](x.cuda(), out, OUTPUT_M=61, OUTPUT_N=249,
                              BOUNDARY_CHECK=(0, 1), num_warps=4)
    got = out.cpu()
    torch.testing.assert_close(got[:61 * 249].reshape(61, 249), x[:61, :249], atol=0, rtol=0)
    assert (got[61 * 249:] == -999).all()



def test_matrix_load_boundary_numeric():
    require_hcu()
    x = (torch.arange(249 * 125).reshape(249, 125).float() % 7).to(torch.float8_e4m3fn)
    out = torch.empty((128, 256), dtype=x.dtype, device="cuda")
    mls_b_roundtrip_kernel[(1,)](x.cuda(), out, ALT=0, INPUT_K=125, INPUT_N=249,
                                 BOUNDARY_CHECK=(0, 1), num_warps=4)
    expected = torch.zeros((128, 256))
    expected[:125, :249] = x.float().T
    torch.testing.assert_close(out.cpu().float(), expected, atol=0, rtol=0)



@g.jit
def ebarrier_api_helper(id, COUNT: gl.constexpr):
    ebarrier_arrive(id, COUNT if COUNT else 4)
    ebarrier_sync(id + 1, COUNT)



@g.jit
def ebarrier_api_kernel(X, Y, ID: gl.constexpr, COUNT: gl.constexpr,
                        DYNAMIC_ID: gl.constexpr = False):
    if DYNAMIC_ID:
        ebarrier_api_helper(gl.program_id(0), COUNT)
    else:
        ebarrier_api_helper(ID, COUNT)
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    i = gl.arange(0, 256, layout=layout)
    gl.store(Y + i, gl.load(X + i) + 1)



@pytest.mark.parametrize("count", [0, 4, 8, 16])
@pytest.mark.parametrize("dynamic", [False, True])
def test_ebarrier_api_compile(count, dynamic):
    require_hcu()
    x = torch.empty(256, device="cuda")
    y = torch.empty_like(x)
    compiled = ebarrier_api_kernel.warmup(x, y, ID=3, COUNT=count,
                                          DYNAMIC_ID=dynamic, grid=(2,), num_warps=4)
    import re
    for op in ("arrive", "sync"):
        assert re.search(r"@llvm\.hcu\.s\.ebarrier\." + op +
                         r"\(i32 [^,]+, i32 " + str((count or 4) if op == "arrive" else count) + r"\)", compiled.asm["llir"])
        assert "s_ebarrier_" + op in compiled.asm["amdgcn"]
    assert "tt.call" not in compiled.asm["ttgir"]



@pytest.mark.parametrize("id,count", [(-1, 4), (16, 4), (True, 4),
                                      (0, -1), (0, True), (0, 2**31)])
def test_ebarrier_invalid_arguments(id, count):
    require_hcu()
    from triton.compiler.errors import CompilationError
    x = torch.empty(256, device="cuda")
    with pytest.raises(CompilationError, match="ebarrier"):
        ebarrier_api_kernel.warmup(x, x, ID=id, COUNT=count, grid=(1,), num_warps=4)



def test_ebarrier_api_runtime():
    require_hcu()
    x = torch.arange(256, dtype=torch.float32)
    y = torch.empty_like(x, device="cuda")
    ebarrier_api_kernel[(1,)](x.cuda(), y, ID=3, COUNT=4, num_warps=4)
    torch.testing.assert_close(y.cpu(), x + 1, atol=0, rtol=0)



@g.jit
def ebarrier_zero_arrive_kernel():
    ebarrier_arrive(3, 0)



def test_ebarrier_zero_arrive_rejected():
    require_hcu()
    from triton.compiler.errors import CompilationError
    with pytest.raises(CompilationError, match="ebarrier count"):
        ebarrier_zero_arrive_kernel.warmup(grid=(1,), num_warps=4)


@g.jit
def wave_id_helper():
    return get_wave_id()


@g.jit
def wave_id_kernel(Y, WARPS: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [WARPS], [0])
    x = gl.arange(0, WARPS * 64, layout=layout)
    gl.store(Y + gl.program_id(0) * WARPS * 64 + x, wave_id_helper())


@g.jit
def wave_id_partition(Y, GROUP: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    x = gl.arange(0, 256, layout=layout)
    gl.store(Y + gl.program_id(0) * 1024 + GROUP * 256 + x, wave_id_helper())


@g.jit
def wave_id_empty():
    pass


@g.jit
def wave_id_partitioned_kernel(Y):
    gl.warp_specialize(
        [(wave_id_empty, ()), (wave_id_partition, (Y, 0)),
         (wave_id_partition, (Y, 1)), (wave_id_partition, (Y, 2)),
         (wave_id_partition, (Y, 3))],
        [4, 4, 4, 4], [8, 8, 240, 240])


@pytest.mark.parametrize("warps", [1, 4, 8])
def test_hcu_wave_id(warps):
    require_hcu()
    y = torch.empty(2 * warps * 64, device="cuda", dtype=torch.int32)
    compiled = wave_id_kernel[(2,)](y, WARPS=warps, num_warps=warps)
    expected = (torch.arange(warps * 64, dtype=torch.int32) // 64).repeat(2)
    torch.testing.assert_close(y.cpu(), expected, atol=0, rtol=0)
    assert "@llvm.amdgcn.s.getreg(i32 8215)" in compiled.asm["llir"]
    assert "s_getreg_b32" in compiled.asm["amdgcn"]
    assert "tt.call" not in compiled.asm["ttgir"]


def test_hcu_wave_id_explicit_partitions():
    require_hcu()
    y = torch.empty(2048, device="cuda", dtype=torch.int32)
    compiled = wave_id_partitioned_kernel[(2,)](
        y, num_warps=4, num_stages=2, wasp_wdra=True,
        wasp_partition_warps=(4, 4, 4, 4))
    assert compiled.metadata.wasp_partition_regs_source == "warp_specialize"
    assert tuple(compiled.metadata.wasp_partition_regs_requested) == (8, 8, 240, 240)
    expected = (torch.arange(1024, dtype=torch.int32) // 64).repeat(2)
    torch.testing.assert_close(y.cpu(), expected, atol=0, rtol=0)
    assert "@llvm.amdgcn.s.getreg(i32 8215)" in compiled.asm["llir"]
    assert "tt.call" not in compiled.asm["ttgir"]


@g.jit
def wait_priority_helper(PRIORITY: gl.constexpr, COUNT: gl.constexpr):
    set_priority(PRIORITY)
    wait_lgkmcnt(COUNT)


@g.jit
def wait_priority_kernel(Y, PRIORITY: gl.constexpr, COUNT: gl.constexpr):
    wait_priority_helper(PRIORITY, COUNT)
    set_priority(0)
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    x = gl.arange(0, 256, layout=layout)
    gl.store(Y + x, x)


@pytest.mark.parametrize("priority", [0, 3])
@pytest.mark.parametrize("count", [0, 9, 14])
def test_hcu_wait_priority_compile(priority, count):
    require_hcu()
    y = torch.empty(256, device="cuda", dtype=torch.int32)
    compiled = wait_priority_kernel.warmup(y, PRIORITY=priority, COUNT=count,
                                           grid=(1,), num_warps=4)
    # GFX9: VM=63, EXP=7 are unconstrained; only LGKM receives the threshold.
    encoded = 0xc07f | (count << 8)
    assert f"@llvm.amdgcn.s.waitcnt(i32 {encoded})" in compiled.asm["llir"]
    assert f"@llvm.amdgcn.s.setprio(i16 {priority})" in compiled.asm["llir"]
    assert f"lgkmcnt({count})" in compiled.asm["amdgcn"]
    assert "tt.call" not in compiled.asm["ttgir"]


@pytest.mark.parametrize("priority,count,message", [
    (-1, 0, "priority"), (4, 0, "priority"), (True, 0, "priority"),
    (0, -1, "lgkmcnt"), (0, 16, "lgkmcnt"), (0, True, "lgkmcnt"),
])
def test_hcu_wait_priority_invalid(priority, count, message):
    require_hcu()
    y = torch.empty(256, device="cuda", dtype=torch.int32)
    with pytest.raises(triton.CompilationError) as caught:
        wait_priority_kernel.warmup(y, PRIORITY=priority, COUNT=count,
                                    grid=(1,), num_warps=4)
    error = caught.value
    while error.__cause__ is not None:
        error = error.__cause__
    assert message in str(error)


def test_hcu_wait_priority_runtime():
    require_hcu()
    y = torch.empty(256, device="cuda", dtype=torch.int32)
    wait_priority_kernel[(1,)](y, PRIORITY=3, COUNT=0, num_warps=4)
    torch.testing.assert_close(y.cpu(), torch.arange(256, dtype=torch.int32), atol=0, rtol=0)
