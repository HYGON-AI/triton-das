# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""CPU-generated inputs/golden, HCU execution, and D2H descriptor comparisons."""
import pytest
import torch
import triton

from test_tensor_descriptor_compile import (
    assert_descriptor_stores_use_pointers,
    descriptor_copy,
    descriptor_dot,
    descriptor_gemm,
    descriptor_transpose_copy,
    pipeline_options,
    shared_load_mixed_dots,
    two_gemms,
)

COPY_DTYPES = [torch.float16, torch.bfloat16, torch.int8, torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2]
COPY_DTYPE_IDS = ["fp16", "bf16", "i8", "u8", "fp8e4nv", "fp8e5"]


def copy_input(dtype, rows, stride):
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        # Include every byte code, including NaNs, infinities and signed zero.
        return torch.arange(rows * stride, dtype=torch.int32).to(torch.uint8).reshape(rows, stride).view(dtype)
    return ((torch.arange(rows * stride, dtype=torch.int32) % 101) - 50).reshape(rows, stride).to(dtype)


@pytest.fixture(autouse=True)
def require_hcu():
    if not torch.cuda.is_available():
        pytest.skip("descriptor execution requires HCU")
    target = triton.runtime.driver.active.get_current_target()
    if target.arch not in ("gfx946", "gfx948"):
        pytest.skip("descriptor execution requires gfx946/gfx948")


@pytest.mark.parametrize("dtype", COPY_DTYPES, ids=COPY_DTYPE_IDS)
@pytest.mark.parametrize("rows,cols,stride,om,on,padding", [
    (64, 64, 64, 0, 0, "zero"),
    (48, 56, 64, 0, 0, "zero"),
    (48, 64, 64, 0, 0, "zero"),
    (64, 56, 64, 0, 0, "zero"),
    (48, 56, 64, 64, 64, "zero"),
    (64, 64, 64, 560, 0, "zero"),
    (64, 64, 1024, 0, 544, "zero"),
    (64, 64, 64, -1, -8, "zero"),
    (64, 64, 64, 0, 1, "zero"),
    (64, 64, 65, 0, 0, "zero"),
])
def test_descriptor_copy_values(dtype, rows, cols, stride, om, on, padding):
    fp8 = dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    # Keep positive-OOB addresses allocated: leaked padding reads must compare
    # against zero instead of depending on the allocator or fault behavior.
    source = copy_input(dtype, max(rows, om + 64), stride)
    output = torch.full((64 * 64 + 32, ), 17, dtype=dtype)
    a, c = source.cuda(), output.cuda()
    compiled = descriptor_copy[(1, )](a, c, rows, cols, 64, 64, stride, om, on, padding, num_warps=4, num_stages=1)
    result = c.cpu()
    # Torch CPU arithmetic on FP8 is limited; byte comparisons also verify that
    # descriptor copies preserve the exact floating-point representation.
    values = source.view(torch.uint8) if fp8 else source
    golden = torch.zeros((64, 64), dtype=values.dtype)
    for row in range(64):
        for col in range(64):
            if 0 <= row + om < rows and 0 <= col + on < cols:
                golden[row, col] = values[row + om, col + on]
    actual = result.view(torch.uint8) if fp8 else result
    reference = output.view(torch.uint8) if fp8 else output
    torch.testing.assert_close(actual[:4096].reshape(64, 64), golden, rtol=0, atol=0)
    torch.testing.assert_close(actual[4096:], reference[4096:], rtol=0, atol=0)
    assert "matrix_load_to_local" not in compiled.asm["ttgir"]
    assert_descriptor_stores_use_pointers(compiled)


def test_descriptor_nan_padding_values():
    source = torch.arange(48 * 64, dtype=torch.float32).reshape(48, 64).to(torch.float16)
    a = source.cuda()
    c = torch.empty((64, 64), dtype=torch.float16).cuda()
    compiled = descriptor_copy[(1, )](a, c, 48, 56, 64, 64, 64, 0, 0, "nan", num_stages=1)
    golden = torch.full((64, 64), float("nan"), dtype=torch.float16)
    golden[:48, :56] = source[:, :56]
    torch.testing.assert_close(c.cpu(), golden, rtol=0, atol=0, equal_nan=True)
    assert "matrix_load_to_local" not in compiled.asm["ttgir"]


@pytest.mark.parametrize("layout", [1, 2, 3, 4])
@pytest.mark.parametrize("stages", [1, 2])
@pytest.mark.parametrize("tail", [False, True, "both"])
def test_descriptor_gemm_values(layout, stages, tail):
    m, n, k = (60, 56 if tail == "both" else 64, 96) if tail else (64, 64, 128)
    rng = torch.Generator(device="cpu").manual_seed(2718)
    a = (torch.randint(-3, 4, (m, k), generator=rng).float() / 8).to(torch.float16)
    b = (torch.randint(-3, 4, (k, n), generator=rng).float() / 8).to(torch.float16)
    golden = (a.float() @ b.float()).to(torch.float16)
    output = torch.full((m * n + 32, ), 17, dtype=torch.float16)
    da, db, dc = a.cuda(), b.cuda(), output.cuda()
    compiled = descriptor_gemm[(1, )](da, db, dc, m, n, k, 2, num_warps=4, num_stages=stages, mmac_layout_force=layout,
                                      optimize_epilogue=True)
    actual = dc.cpu()
    torch.testing.assert_close(actual[:m * n].reshape(m, n), golden, rtol=0, atol=0)
    torch.testing.assert_close(actual[m * n:], output[m * n:], rtol=0, atol=0)
    assert "amdg.matrix_load_to_local" in compiled.asm["ttgir"]
    assert_descriptor_stores_use_pointers(compiled)


def test_shared_descriptor_load_values():
    rng = torch.Generator(device="cpu").manual_seed(31415)
    a, b, b2 = [(torch.randint(-3, 4, (64, 64), generator=rng).float() / 8).half() for _ in range(3)]
    golden = [(a.float() @ x.float()).half() for x in (b, b2)]
    da, db, db2 = [x.cuda() for x in (a, b, b2)]
    outputs = [torch.full((4096 + 32, ), 17, dtype=torch.float16) for _ in range(2)]
    dc, dc2 = [x.cuda() for x in outputs]
    compiled = two_gemms[(1, )](da, db, db2, dc, dc2, num_warps=4, num_stages=1, mmac_layout_force=3,
                                optimize_epilogue=True)
    for actual, ref in zip((dc.cpu(), dc2.cpu()), golden):
        torch.testing.assert_close(actual[:4096].reshape(64, 64), ref, rtol=0, atol=0)
        torch.testing.assert_close(actual[4096:], outputs[0][4096:], rtol=0, atol=0)
    assert_descriptor_stores_use_pointers(compiled)


# BF16/FP8 also check exact on/off IR parity in the compile regression. Their
# descriptor MLS execution is covered by the GEMM tests below.
@pytest.mark.parametrize("dtype", [torch.float16, torch.int8], ids=["fp16", "i8"])
@pytest.mark.parametrize("narrow_first", [False, True])
def test_shared_load_mixed_dot_values(dtype, narrow_first):
    source = ((torch.arange(4096, dtype=torch.int32) % 7) - 3).reshape(64, 64).to(dtype)
    golden = source.float().sum(dim=1, keepdim=True).half()
    outputs = [torch.full((64 * width + 32, ), 17, dtype=torch.float16) for width in (64, 8)]
    a = source.cuda()
    c, d = [output.cuda() for output in outputs]
    compiled = shared_load_mixed_dots[(1, )](a, c, d, narrow_first, num_warps=4, num_stages=1)
    for device, output, width in zip((c, d), outputs, (64, 8)):
        actual = device.cpu()
        torch.testing.assert_close(actual[:64 * width].reshape(64, width), golden.expand(64, width), rtol=0, atol=0)
        torch.testing.assert_close(actual[64 * width:], output[64 * width:], rtol=0, atol=0)
    assert "matrix_load_to_local" not in compiled.asm["ttgir"]


@pytest.mark.parametrize(
    "mode", ["async", "wasp-single", "wasp", "wasp-tail", "wasp-single-trans", "wasp-trans", "wasp-tail-trans"])
@pytest.mark.parametrize("layout", [2, 3])
def test_descriptor_gemm_pipeline_values(mode, layout):
    base = mode.removesuffix("-trans")
    transposed = mode.endswith("-trans")
    block = 128 if base == "wasp-single" else 256 if base in ("wasp", "wasp-tail") else 64
    m, n, k = (240, 248, 96) if base == "wasp-tail" else (block, block, 128)
    rng = torch.Generator(device="cpu").manual_seed(1618)
    a = (torch.randint(-3, 4, (m, k), generator=rng).float() / 8).half()
    b = (torch.randint(-3, 4, (n, k) if transposed else (k, n), generator=rng).float() / 8).half()
    golden = (a.float() @ (b.float().T if transposed else b.float())).half()
    output = torch.full((m * n + 32, ), 17, dtype=torch.float16)
    da, db, dc = a.cuda(), b.cuda(), output.cuda()
    compiled = descriptor_gemm[(1, )](da, db, dc, m, n, k, 2, BM=block, BN=block, WARP_SPECIALIZE=base != "async",
                                      TRANS_B=transposed, **dict(pipeline_options(mode), mmac_layout_force=layout))
    actual = dc.cpu()
    torch.testing.assert_close(actual[:m * n].reshape(m, n), golden, rtol=0, atol=0)
    torch.testing.assert_close(actual[m * n:], output[m * n:], rtol=0, atol=0)
    assert "amdg.matrix_load_to_local" in compiled.asm["ttgir"]
    assert_descriptor_stores_use_pointers(compiled)


@pytest.mark.parametrize("dtype", COPY_DTYPES, ids=COPY_DTYPE_IDS)
@pytest.mark.parametrize("rows,cols", [(64, 64), (48, 64), (64, 48)])
def test_descriptor_transpose_values(dtype, rows, cols):
    source = copy_input(dtype, rows, 64)
    output = torch.full((4096 + 32, ), 17, dtype=dtype)
    a, c = source.cuda(), output.cuda()
    compiled = descriptor_transpose_copy[(1, )](a, c, rows, cols, num_warps=4, num_stages=1)
    result = c.cpu()
    fp8 = dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    values = source.view(torch.uint8) if fp8 else source
    actual = result.view(torch.uint8) if fp8 else result
    reference = output.view(torch.uint8) if fp8 else output
    golden = torch.zeros((64, 64), dtype=values.dtype)
    golden[:cols, :rows] = values[:, :cols].T
    torch.testing.assert_close(actual[:4096].reshape(64, 64), golden, rtol=0, atol=0)
    torch.testing.assert_close(actual[4096:], reference[4096:], rtol=0, atol=0)
    assert "matrix_load_to_local" not in compiled.asm["ttgir"]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int8, torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("rows,cols,stride,om,on,trans", [
    (64, 64, 64, 0, 0, True),
    (48, 64, 64, 0, 0, True),
    (64, 48, 64, 0, 0, True),
    (48, 56, 64, 0, 0, False),
    (64, 64, 64, 560, 0, False),
    (64, 64, 1024, 0, 544, False),
])
def test_descriptor_dot_values(dtype, rows, cols, stride, om, on, trans):
    storage_rows = max(64, om + 64)
    source = ((torch.arange(storage_rows * stride, dtype=torch.int32) % 7) - 3).reshape(storage_rows, stride)
    b = ((torch.arange(4096, dtype=torch.int32) % 5) - 2).reshape(64, 64)
    if dtype != torch.int8:
        source, b = source.float() / 8, b.float() / 8
    source, b = source.to(dtype), b.to(dtype)
    padded = torch.zeros((64, 64), dtype=torch.float32)
    valid_rows, valid_cols = max(0, min(64, rows - om)), max(0, min(64, cols - on))
    padded[:valid_rows, :valid_cols] = source[om:om + valid_rows, on:on + valid_cols].float()
    golden = ((padded.T if trans else padded) @ b.float()).half()
    output = torch.full((4096 + 32, ), 17, dtype=torch.float16)
    da, db, dc = source.cuda(), b.cuda(), output.cuda()
    compiled = descriptor_dot[(1, )](da, db, dc, rows, cols, 64, 64, stride, om, on, "zero", TRANS=trans, num_stages=1)
    actual = dc.cpu()
    torch.testing.assert_close(actual[:4096].reshape(64, 64), golden, rtol=0, atol=0)
    torch.testing.assert_close(actual[4096:], output[4096:], rtol=0, atol=0)
    assert "tt.trans " not in compiled.asm["ttir"]
    assert "amdg.matrix_load_to_local" in compiled.asm["ttgir"]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("block,layout,tail", [(128, 2, False), (128, 3, False), (128, 2, True), (128, 3, True),
                                               (64, 3, False)])
def test_descriptor_gemm_transposed_b_values(dtype, block, layout, tail):
    m, n, k = (block - 4, block - 8, 96) if tail else (block, block, 128)
    rng = torch.Generator(device="cpu").manual_seed(1732)
    a = (torch.randint(-3, 4, (m, k), generator=rng).float() / 8).to(dtype)
    b = (torch.randint(-3, 4, (n, k), generator=rng).float() / 8).to(dtype)
    golden = (a.float() @ b.float().T).to(dtype)
    output = torch.full((m * n + 32, ), 17, dtype=dtype)
    da, db, dc = a.cuda(), b.cuda(), output.cuda()
    compiled = descriptor_gemm[(1, )](da, db, dc, m, n, k, 2, BM=block, BN=block, TRANS_B=True, num_warps=4,
                                      num_stages=2, mmac_layout_force=layout, optimize_epilogue=True)
    actual = dc.cpu()
    torch.testing.assert_close(actual[:m * n].reshape(m, n), golden, rtol=0, atol=0)
    torch.testing.assert_close(actual[m * n:], output[m * n:], rtol=0, atol=0)
    assert compiled.asm["ttir"].count("tt.matrix_load") == 2
    assert "tt.trans " not in compiled.asm["ttir"]
    assert "amdg.matrix_load_to_local" in compiled.asm["ttgir"]
    assert_descriptor_stores_use_pointers(compiled)


def test_standalone_transpose_keeps_the_original_read_position():
    source = copy_input(torch.float16, 64, 64)
    da = source.cuda()
    dc = torch.full((64, 64), 17, dtype=torch.float16).cuda()
    compiled = descriptor_transpose_copy[(1, )](da, dc, 64, 64, WRITE_INPUT=True, num_warps=4, num_stages=1)
    torch.testing.assert_close(dc.cpu(), source.T, rtol=0, atol=0)
    torch.testing.assert_close(da.cpu(), torch.zeros_like(source), rtol=0, atol=0)
    assert "matrix_load_to_local" not in compiled.asm["ttgir"]


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("layout", [2, 3])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("tail", [False, True])
def test_fp8_descriptor_gemm_values(dtype, layout, transposed, tail):
    m, n, k = (60, 48, 96) if tail else (64, 64, 128)
    rng = torch.Generator(device="cpu").manual_seed(811)
    a = (torch.randint(-3, 4, (m, k), generator=rng).float() / 8).to(dtype)
    b = (torch.randint(-3, 4, (n, k) if transposed else (k, n), generator=rng).float() / 8).to(dtype)
    golden = (a.float() @ (b.float().T if transposed else b.float())).half()
    output = torch.full((m * n + 32, ), 17, dtype=torch.float16)
    da, db, dc = a.cuda(), b.cuda(), output.cuda()
    compiled = descriptor_gemm[(1, )](da, db, dc, m, n, k, 2, TRANS_B=transposed, num_warps=4, num_stages=2,
                                      mmac_layout_force=layout)
    actual = dc.cpu()
    torch.testing.assert_close(actual[:m * n].reshape(m, n), golden, rtol=0, atol=0)
    torch.testing.assert_close(actual[m * n:], output[m * n:], rtol=0, atol=0)
    assert "amdg.matrix_load_to_local" in compiled.asm["ttgir"]
    assert_descriptor_stores_use_pointers(compiled)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bm,bn,warps", [(16, 32, 1), (64, 64, 8), (64, 64, 16)])
def test_descriptor_dot_warp_counts_values(dtype, bm, bn, warps):
    source = copy_input(dtype, 64, 64)
    b = torch.ones((bn, 64), dtype=dtype)
    golden = (source[:bm, :bn].float() @ b.float()).to(dtype)
    output = torch.full((bm * 64 + 32, ), 17, dtype=dtype)
    da, db, dc = source.cuda(), b.cuda(), output.cuda()
    compiled = descriptor_dot[(1, )](da, db, dc, 64, 64, bm, bn, 64, 0, 0, "zero", num_warps=warps, num_stages=1)
    actual = dc.cpu()
    torch.testing.assert_close(actual[:bm * 64].reshape(bm, 64), golden, rtol=0, atol=0)
    torch.testing.assert_close(actual[bm * 64:], output[bm * 64:], rtol=0, atol=0)
    assert_descriptor_stores_use_pointers(compiled)
    assert "amdg.matrix_load_to_local" in compiled.asm["ttgir"]
