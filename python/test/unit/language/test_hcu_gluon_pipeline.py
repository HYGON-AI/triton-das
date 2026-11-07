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
    packed_scale_fma,
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
def packed_kernel(X, S, C, B, Y, ALIAS: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([2], [64], [4], [0])
    i = gl.arange(0, 512, layout=layout)
    x = gl.load(X + i)
    s = x if ALIAS else gl.load(S + i)
    c = x if ALIAS else gl.load(C + i)
    b = gl.load(B)
    y = packed_scale_fma(x, s, b, c)
    gl.store(Y + i, y)


@pytest.mark.parametrize("alias", [False, True])
def test_packed_scale_fma(alias):
    require_hcu()
    # Dyadic inputs make the separately rounded multiply and fused add exact.
    x = ((torch.arange(512) % 17 - 8).float() / 8).contiguous()
    s = torch.full_like(x, 0.25)
    c = torch.full_like(x, -0.5)
    b = torch.tensor([0.5])
    y = torch.empty_like(x, device="cuda")
    compiled = packed_kernel[(1,)](
        *(t.cuda() for t in (x, s, c, b)), y, ALIAS=alias, num_warps=4
    )
    expected = (x * (x if alias else s)) * b + (x if alias else c)
    torch.testing.assert_close(y.cpu(), expected, atol=0, rtol=0)
    assert "v_pk_mul_f32" in compiled.asm["amdgcn"]
    assert "v_pk_fma_f32" in compiled.asm["amdgcn"]


@g.jit
def narrow_copy_kernel(X, Y):
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    i = gl.arange(0, 64, layout=layout)
    smem = gl.allocate_shared_memory(
        gl.float32, (2, 64), gl.SwizzledSharedLayout(1, 1, 1, [0])
    )
    smem.index(0).store(gl.full((64,), -123.0, gl.float32, layout))
    smem.index(1).store(gl.full((64,), -456.0, gl.float32, layout))
    # Two-element destination needs redundant lanes predicated off. Guard the
    # adjacent allocation and verify both rows remain intact.
    pair = gl.allocate_shared_memory(
        gl.float32, (2,), gl.SwizzledSharedLayout(1, 1, 1, [0])
    )
    async_copy.buffer_load_to_shared(pair, X, gl.arange(0, 2, layout=layout))
    async_copy.wait_group(0)
    raw = smem.index(1).load(layout)
    gl.store(Y + i, raw)
    gl.store(Y + 64 + i, smem.index(0).load(layout))
    j = gl.arange(0, 2, layout=layout)
    gl.store(Y + 128 + j, pair.load(layout))


def test_narrow_copy_preserves_adjacent_shared_slices():
    require_hcu()
    x = torch.tensor([1.25, -3.5])
    y = torch.empty(130, device="cuda")
    narrow_copy_kernel[(1,)](x.cuda(), y, num_warps=4)
    expected = torch.cat((torch.full((64,), -456.0), torch.full((64,), -123.0), x))
    torch.testing.assert_close(y.cpu(), expected, atol=0, rtol=0)


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
def dot_store_kernel(A, B, C, STORE: gl.constexpr):
    F: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        mmac_layout=MmacLayout.LTS,
        warps_per_cta=[1, 4],
        tiles_per_warp=[4, 2],
    )
    L: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
    m = gl.arange(0, 64, layout=gl.SliceLayout(1, L))
    k = gl.arange(0, 32, layout=gl.SliceLayout(0, L))
    kk = gl.arange(0, 32, layout=gl.SliceLayout(1, L))
    n = gl.arange(0, 128, layout=gl.SliceLayout(0, L))
    a = gl.load(A + m[:, None] * 32 + k[None, :])
    b = gl.load(B + kk[:, None] * 128 + n[None, :])
    a = gl.convert_layout(a, gl.DotOperandLayout(0, F, 8))
    b = gl.convert_layout(b, gl.DotOperandLayout(1, F, 8))
    c = gl.amd.cdna4.mfma(a, b, gl.full((64, 128), 0, gl.float32, F))
    if STORE:
        matrix_store(
            C,
            c.to(gl.bfloat16),
            (64, 128),
            (128, 1),
            (0, 0),
            tile=(16, 32),
        )
    else:
        mm = gl.arange(0, 64, layout=gl.SliceLayout(1, F))
        nn = gl.arange(0, 128, layout=gl.SliceLayout(0, F))
        gl.store(C + mm[:, None] * 128 + nn[None, :], c.to(gl.bfloat16))


@pytest.mark.parametrize("matrix_output", [False, True])
def test_dot_store(matrix_output):
    require_hcu()
    torch.manual_seed(1)
    a = (torch.randn(64, 32) * 0.5).to(torch.float8_e4m3fn)
    b = (torch.randn(32, 128) * 0.5).to(torch.float8_e4m3fn)
    ref = (a.float() @ b.float()).bfloat16()
    out = torch.empty_like(ref, device="cuda")
    if matrix_output:
        with pytest.raises(triton.CompilationError, match="explicit interleave2 result layout"):
            dot_store_kernel[(1,)](a.cuda(), b.cuda(), out, STORE=True, num_warps=4)
        return
    dot_store_kernel[(1,)](a.cuda(), b.cuda(), out, STORE=False, num_warps=4)
    torch.testing.assert_close(out.cpu(), ref, atol=0, rtol=0)


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


@pytest.mark.parametrize("alt", [0, 1])
def test_mls_b_to_ordinary_store(alt):
    require_hcu()
    torch.manual_seed(13)
    x = (torch.randint(-16, 17, (256, 128)).float() / 8).to(torch.float8_e4m3fn)
    y = torch.empty((128, 256), device="cuda", dtype=torch.float8_e4m3fn)
    mls_b_roundtrip_kernel[(1,)](x.cuda(), y, ALT=alt, num_warps=4)
    assert torch.equal(y.cpu().view(torch.uint8), x.T.contiguous().view(torch.uint8))


@builtin
def check_interleave_layout_roundtrip(value, _semantic=None):
    restored = _semantic.builder.get_gluon_layout_from_tensor(value.handle)
    assert restored == value.type.layout
    roundtrip = gl.distributed_type(value.dtype, value.shape, restored).to_ir(_semantic.builder)
    assert str(roundtrip) == str(value.type.to_ir(_semantic.builder))
    return value


@g.jit
def native_mmac_store_kernel(A, B, C, MATRIX: gl.constexpr, BAD: gl.constexpr = 0):
    instruction: gl.constexpr = get_mmac_instruction(A.dtype.element_ty, B.dtype.element_ty, gl.float32, 128)
    f: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=instruction[0], mmac_layout=MmacLayout.LTS,
        warps_per_cta=[1, 4], tiles_per_warp=[4, 4],
    )
    bits: gl.constexpr = B.dtype.element_ty.primitive_bitwidth
    sb: gl.constexpr = MlsSharedLayout(1, (64, 64) if bits == 16 else (128, 64),
                                     bits=bits, alt=1, version=2 if bits == 16 else 3, order=(0, 1))
    rf: gl.constexpr = MmacInterleave2Layout(f)
    shared = gl.allocate_shared_memory(B.dtype.element_ty, (128, 256), sb)
    matrix_load(shared, B, (128, 256), (1, 128), (0, 0))
    gl.inline_asm_elementwise(
        "s_waitcnt lgkmcnt(0)\n\ts_barrier\n\ts_mov_b32 $0, 0",
        constraints="=s,~{memory}", args=[], dtype=gl.int32, is_pure=False, pack=1,
    )
    l: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
    m = gl.arange(0, 64, layout=gl.SliceLayout(1, l))
    k = gl.arange(0, 128, layout=gl.SliceLayout(0, l))
    a = gl.convert_layout(gl.load(A + m[:, None] * 128 + k[None, :]), gl.DotOperandLayout(0, f, instruction[1]))
    b = async_copy.load_shared_relaxed(shared, MmacInterleave2Layout(gl.DotOperandLayout(1, f, instruction[1])))
    b = check_interleave_layout_roundtrip(b)
    acc = check_interleave_layout_roundtrip(gl.full((64, 256), 0, gl.float32, rf))
    if BAD == 1:
        selected_b = gl.convert_layout(b, gl.DotOperandLayout(1, f, instruction[1]))
    else:
        selected_b = b
    if BAD == 2:
        selected_acc = gl.full((64, 256), 0, gl.float32, f)
    else:
        selected_acc = acc
    c = gl.amd.cdna4.mfma(a, selected_b, selected_acc).to(gl.bfloat16)
    if MATRIX:
        matrix_store(C, c, (64, 256), (256, 1), (0, 0))
    else:
        mm = gl.arange(0, 64, layout=gl.SliceLayout(1, rf))
        nn = gl.arange(0, 256, layout=gl.SliceLayout(0, rf))
        gl.store(C + mm[:, None] * 256 + nn[None, :], c)


@pytest.mark.parametrize("matrix", [False, True])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2, torch.float16, torch.bfloat16])
def test_native_mmac_logical_store(matrix,dtype):
    require_hcu()
    torch.manual_seed(37)
    a = (torch.randint(-8, 9, (64, 128)).float() / 8).to(dtype)
    b = (torch.randint(-8, 9, (256, 128)).float() / 8).to(dtype)
    out = torch.empty((64, 256), device="cuda", dtype=torch.bfloat16)
    native_mmac_store_kernel[(1,)](a.cuda(), b.cuda(), out, MATRIX=matrix, num_warps=4)
    torch.testing.assert_close(out.cpu(), (a.float() @ b.float().T).bfloat16(), atol=0, rtol=0)


@pytest.mark.parametrize("operand", [0, 1])
def test_interleave_layout_identity_and_invalid_parent(operand):
    f = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], mmac_layout=MmacLayout.LTS,
        warps_per_cta=[1, 4], tiles_per_warp=[4, 4],
    )
    result = MmacInterleave2Layout(f)
    if operand == 0:
        with pytest.raises(ValueError, match="only B"):
            MmacInterleave2Layout(gl.DotOperandLayout(0, f, 8))
    else:
        b = MmacInterleave2Layout(gl.DotOperandLayout(1, f, 8))
        assert result != b and result.mangle() != b.mangle()
        assert len({result, b}) == 2
    with pytest.raises(ValueError, match="LTS"):
        MmacInterleave2Layout(gl.BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0]))


@pytest.mark.parametrize("bad", [1, 2])
def test_interleave_dot_rejects_mismatched_layouts(bad, capfd):
    require_hcu()
    from triton.compiler.errors import CompilationError
    a = torch.empty((64, 128), device="cuda", dtype=torch.float8_e4m3fn)
    b = torch.empty((256, 128), device="cuda", dtype=torch.float8_e4m3fn)
    out = torch.empty((64, 256), device="cuda", dtype=torch.bfloat16)
    with pytest.raises((CompilationError, RuntimeError)) as error:
        native_mmac_store_kernel.warmup(a, b, out, MATRIX=True, BAD=bad, grid=(1,), num_warps=4)
    diagnostics = capfd.readouterr()
    assert "interleave2" in str(error.value) + diagnostics.err + diagnostics.out


@g.jit
def interleave2_global_dot_kernel(A, B, C, MATRIX: gl.constexpr):
    instruction: gl.constexpr = get_mmac_instruction(A.dtype.element_ty, B.dtype.element_ty, gl.float32, 128)
    f: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=instruction[0], mmac_layout=MmacLayout.LTS,
        warps_per_cta=[1, 4], tiles_per_warp=[4, 4],
    )
    result_layout: gl.constexpr = MmacInterleave2Layout(f)
    b_layout: gl.constexpr = MmacInterleave2Layout(gl.DotOperandLayout(1, f, instruction[1]))
    ordinary: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
    m = gl.arange(0, 64, layout=gl.SliceLayout(1, ordinary))
    ka = gl.arange(0, 128, layout=gl.SliceLayout(0, ordinary))
    a = gl.convert_layout(gl.load(A + m[:, None] * 128 + ka[None, :]), gl.DotOperandLayout(0, f, instruction[1]))
    kb = gl.arange(0, 128, layout=gl.SliceLayout(1, b_layout))
    n = gl.arange(0, 256, layout=gl.SliceLayout(0, b_layout))
    b = gl.load(B + n[None, :] * 128 + kb[:, None])
    acc = gl.full((64, 256), 0, gl.float32, result_layout)
    c = gl.amd.cdna4.mfma(a, b, acc).to(gl.bfloat16)
    if MATRIX:
        matrix_store(C, c, (64, 256), (256, 1), (0, 0))
    else:
        mm = gl.arange(0, 64, layout=gl.SliceLayout(1, result_layout))
        nn = gl.arange(0, 256, layout=gl.SliceLayout(0, result_layout))
        gl.store(C + mm[:, None] * 256 + nn[None, :], c)


@pytest.mark.parametrize("matrix", [False, True])
@pytest.mark.parametrize("ad,bd", [(torch.float8_e4m3fn,torch.float8_e4m3fn),
    (torch.float8_e5m2,torch.float8_e5m2), (torch.float8_e4m3fn,torch.float8_e5m2),
    (torch.float16,torch.float16),(torch.bfloat16,torch.bfloat16),(torch.float32,torch.float32)])
def test_interleave2_global_load_dot_store(matrix,ad,bd):
    require_hcu()
    torch.manual_seed(43)
    a = (torch.randint(-8, 9, (64, 128)).float() / 8).to(ad)
    b = (torch.randint(-8, 9, (256, 128)).float() / 8).to(bd)
    out = torch.empty((64, 256), device="cuda", dtype=torch.bfloat16)
    interleave2_global_dot_kernel[(1,)](a.cuda(), b.cuda(), out, MATRIX=matrix, num_warps=4)
    torch.testing.assert_close(out.cpu(), (a.float() @ b.float().T).bfloat16(), atol=0, rtol=0)


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


@g.jit
def matrix_load_loop(A, C, tiles):
    storage = gl.allocate_shared_memory(gl.float16, [256, 64], MlsSharedLayout(0, (64, 64), 16, order=(1, 0)))
    mma: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 16], MmacLayout.LIT, [4, 1], tiles_per_warp=[4, 1])
    regs: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
    rows = gl.arange(0, 256, gl.SliceLayout(1, regs))
    cols = gl.arange(0, 64, gl.SliceLayout(0, regs))
    for tile in range(tiles):
        matrix_load(storage, A + tile * 16384, (256, 64), (64, 1), (0, 0), (4, 1))
        async_copy.commit_group()
        async_copy.wait_group()
        values = storage.load(gl.DotOperandLayout(0, mma, 4))
        values = gl.convert_layout(values, regs)
        gl.store(C + tile * 16384 + rows[:, None] * 64 + cols[None, :], values)
        # Finish reads before the next iteration overwrites the same allocation.
        gl.thread_barrier()


@g.jit
def matrix_load_loop_kernel(A, C, tiles, PARTITIONED: gl.constexpr):
    if PARTITIONED:
        gl.warp_specialize(
            [(wave_id_empty, ()), (matrix_load_loop, (A, C, tiles)),
             (matrix_load_loop, (A, C + tiles * 16384, tiles))], [4, 4], [64, 64])
    else:
        matrix_load_loop(A, C, tiles)


@pytest.mark.parametrize("partitioned", [False, True])
def test_hcu_pipeline_matrix_load_loop(partitioned):
    import re

    require_hcu()
    tiles = 3  # Runtime trip count keeps the producer loop in LLVM IR.
    data = ((torch.arange(tiles * 16384, device="cpu") % 251) - 125).half()
    expected = data.repeat(2 if partitioned else 1)
    output = torch.full_like(expected, float("nan")).cuda()
    options = {"num_warps": 4, "num_stages": 1, "use_async_copy": True}
    if partitioned:
        options.update(wasp_partition_warps=(4, 4), wasp_wdra=False, num_stages=2)
    compiled = matrix_load_loop_kernel[(1,)](data.cuda(), output, tiles, partitioned, **options)
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    blocks = re.split(r"(?m)^[a-zA-Z0-9_.]+:.*$", compiled.asm["llir"])
    wave_id_blocks = [block for block in blocks if "call i32 @llvm.amdgcn.readfirstlane" in block]
    assert len(wave_id_blocks) == (2 if partitioned else 1)
    # In this straight-line loop, the induction phi marks the loop body.
    # Wave IDs must stay in the function/partition entry, outside that body.
    assert all(" = phi " not in block for block in wave_id_blocks)


