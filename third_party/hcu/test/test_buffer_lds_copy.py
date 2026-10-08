# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""Buffer-to-LDS fill, rotation and drain; no dependency on boltOps or a wheel patch."""
import pytest
import re
import torch
import triton
import triton.language as tl


@pytest.mark.parametrize("async_copy", [False, True])
@pytest.mark.parametrize("pingpong", [False, True])
@pytest.mark.parametrize("matrix_read", [False, True])
def test_async_copy_dispatch(
        monkeypatch, async_copy, pingpong, matrix_read):
    if not torch.cuda.is_available():
        pytest.skip("HCU device required")
    if "gfx936" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("gfx936 validation")
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", "1")
    monkeypatch.setenv("TRITON_HCU_ENABLE_DS_READ_M", str(int(matrix_read)))
    # Explicit False must win even when the environment requests async-copy.
    monkeypatch.setenv("TRITON_HIP_USE_ASYNC_COPY", "1")
    a = torch.randn((64, 128), device="cuda", dtype=torch.float16)
    b = torch.randn((128, 64), device="cuda", dtype=torch.float16)
    c = torch.empty((64, 64), device="cuda", dtype=torch.float16)
    compiled = direct_gemm[(1,)](
        a, b, c, 64, 64, 128, BM=64, BN=64, BK=16, num_warps=4,
        num_stages=2, use_block_pingpong=pingpong, use_async_copy=async_copy,
        matrix_instr_nonkdim=16, optimize_epilogue=False)
    torch.testing.assert_close(c, a @ b, atol=0.02, rtol=0.02)
    assert compiled.metadata.use_async_copy is async_copy
    uses_buffer_lds = async_copy
    assert compiled.metadata.uses_bank_conflict_aware_lds_placement is uses_buffer_lds
    assert ("#ttg.hcu_buffer_lds_shared" in compiled.asm["ttgir"]) is uses_buffer_lds
    if uses_buffer_lds:
        assert "buffer_load_dword" in compiled.asm["amdgcn"]


@triton.jit
def direct_gemm(a, b, c, m, n, k,
                BM: tl.constexpr = 256, BN: tl.constexpr = 256,
                BK: tl.constexpr = 16, REVERSE_A: tl.constexpr = False,
                A_COLUMN_MAJOR: tl.constexpr = False,
                B_COLUMN_MAJOR: tl.constexpr = False,
                BATCHED: tl.constexpr = False,
                AFFINE_COORDS: tl.constexpr = False,
                K_STEP_RANGE: tl.constexpr = False):
    tl.assume(m > 0)
    tl.assume(n > 0)
    tl.assume(k > 0)
    if BATCHED:
        batch = tl.program_id(1)
        a += batch * m * k
        b += batch * k * n
        c += batch * m * n
    pm = tl.program_id(0) // tl.cdiv(n, BN)
    pn = tl.program_id(0) % tl.cdiv(n, BN)
    rows = pm * BM + tl.arange(0, BM)
    if not AFFINE_COORDS:
        rows %= m
    if REVERSE_A:
        rows = m - 1 - rows
    cols = pn * BN + tl.arange(0, BN)
    if not AFFINE_COORDS:
        cols %= n
    ks = tl.arange(0, BK)
    ap = a + rows[:, None] * (1 if A_COLUMN_MAJOR else k) + ks[None, :] * (m if A_COLUMN_MAJOR else 1)
    bp = b + ks[:, None] * (1 if B_COLUMN_MAJOR else n) + cols[None, :] * (k if B_COLUMN_MAJOR else 1)
    acc = tl.zeros((BM, BN), tl.int32 if a.dtype.element_ty == tl.int8 else tl.float32)
    if K_STEP_RANGE:
        for _ in range(0, k, BK):
            av = tl.load(ap)
            bv = tl.load(bp)
            acc = tl.dot(av, bv, acc, out_dtype=acc.dtype)
            ap += BK * (m if A_COLUMN_MAJOR else 1)
            bp += BK * (1 if B_COLUMN_MAJOR else n)
    else:
        for _ in range(k // BK):
            av = tl.load(ap)
            bv = tl.load(bp)
            acc = tl.dot(av, bv, acc, out_dtype=acc.dtype)
            ap += BK * (m if A_COLUMN_MAJOR else 1)
            bp += BK * (1 if B_COLUMN_MAJOR else n)
    out_rows = pm * BM + tl.arange(0, BM)
    out_cols = pn * BN + tl.arange(0, BN)
    tl.store(c + out_rows[:, None] * n + out_cols[None, :], acc,
             (out_rows[:, None] < m) & (out_cols[None, :] < n))


@triton.jit
def wasp_gemm(a, b, c, m: tl.constexpr, n: tl.constexpr, k: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.arange(0, BM)
    cols = tl.arange(0, BN)
    ks = tl.arange(0, BK)
    ap = a + rows[:, None] * k + ks[None, :]
    bp = b + ks[:, None] * n + cols[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in tl.range(0, tl.cdiv(k, BK), warp_specialize=True):
        av = tl.load(ap)
        bv = tl.load(bp)
        acc = tl.dot(av, bv, acc)
        ap += BK
        bp += BK * n
    tl.store(c + rows[:, None] * n + cols[None, :], acc)


def test_shaobo_wasp_async_buffer_lds(monkeypatch):
    """WASP load waves use async BufferLds without a whole-CTA barrier."""
    if not torch.cuda.is_available():
        pytest.skip("HCU device required")
    if "gfx946" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("gfx946 validation")
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", "1")
    m, n, k = 64, 64, 112
    generator = torch.Generator().manual_seed(1946)
    a_host = torch.randn((m, k), generator=generator, dtype=torch.float16)
    b_host = torch.randn((k, n), generator=generator, dtype=torch.float16)
    a, b = a_host.cuda(), b_host.cuda()
    c = torch.empty((m, n), device="cuda", dtype=torch.float16)
    compiled = wasp_gemm[(1,)](
        a, b, c, m, n, k, BM=m, BN=n, BK=16, num_stages=2,
        use_async_copy=True, use_block_pingpong=False,
        wasp_wdra=False, wasp_partition_warps=(4, 4),
        matrix_instr_nonkdim=16, optimize_epilogue=False)
    expected = (a_host.float() @ b_host.float()).half()
    torch.testing.assert_close(c.cpu(), expected, atol=0.02, rtol=0.02)
    assert compiled.metadata.uses_bank_conflict_aware_lds_placement
    assert "#ttg.hcu_buffer_lds_shared" in compiled.asm["ttgir"]
    assert "amdg.buffer_load_to_local" in compiled.asm["ttgir"]
    assert "ttg.local_barrier" not in compiled.asm["ttgir"]
    assert "llvm.amdgcn.raw.ptr.buffer.load.async.lds" in compiled.asm["llir"]
    assert re.search(r"buffer_load_\w+.*\blds\b", compiled.asm["amdgcn"])


@pytest.mark.parametrize("wdra,waves", [
    pytest.param(False, (1, 1, 1, 1), id="wasp-1wave"),
    pytest.param(False, (2, 2, 2, 2), id="wasp-2wave"),
    pytest.param(False, (4, 4, 4, 4), id="wasp-4wave"),
    pytest.param(True, (4, 4, 4, 4), id="wdra-4wave"),
])
def test_shaobo_wasp_two_producer_async_buffer_lds(
        monkeypatch, tmp_path, wdra, waves):
    """Two WASP producers issue async BufferLds into pair-private tiles."""
    if not torch.cuda.is_available():
        pytest.skip("HCU device required")
    if "gfx946" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("gfx946 validation")
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", "1")
    m, n, k = 64, 64, 512
    generator = torch.Generator().manual_seed(2946)
    a_host = torch.randn((m, k), generator=generator, dtype=torch.float16)
    b_host = torch.randn((k, n), generator=generator, dtype=torch.float16)
    a, b = a_host.cuda(), b_host.cuda()
    c = torch.empty((m, n), device="cuda", dtype=torch.float16)
    compiled = wasp_gemm[(1,)](
        a, b, c, m, n, k, BM=m, BN=n, BK=16, num_stages=2,
        use_async_copy=True, use_block_pingpong=False,
        wasp_wdra=wdra, wasp_partition_warps=waves,
        matrix_instr_nonkdim=16, optimize_epilogue=False)
    expected = (a_host.float() @ b_host.float()).half()
    torch.testing.assert_close(c.cpu(), expected, atol=0.02, rtol=0.02)
    assert compiled.metadata.uses_bank_conflict_aware_lds_placement
    ttgir = compiled.asm["ttgir"]
    assert "hcu.buffer_lds_producer_config" in ttgir
    assert ttgir.count("ttg.local_alloc") == 4
    assert ttgir.count("amdg.buffer_load_to_local") >= 4
    assert "ttg.local_barrier" not in ttgir
    assert "llvm.amdgcn.raw.ptr.buffer.load.async.lds" in compiled.asm["llir"]
    assert re.search(r"buffer_load_\w+.*\blds\b", compiled.asm["amdgcn"])
    for stage in ("ttgir", "llir", "amdgcn"):
        (tmp_path / f"wasp-2p-buffer-lds.{stage}").write_text(
            compiled.asm[stage])


def test_shaobo_wdra_async_buffer_lds(monkeypatch):
    """WDRA 4+8 keeps async BufferLds across consumer tile views."""
    if not torch.cuda.is_available():
        pytest.skip("HCU device required")
    if "gfx946" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("gfx946 validation")
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", "1")
    m, n, k = 64, 64, 112
    generator = torch.Generator().manual_seed(4948)
    a_host = torch.randn((m, k), generator=generator, dtype=torch.float16)
    b_host = torch.randn((k, n), generator=generator, dtype=torch.float16)
    a, b = a_host.cuda(), b_host.cuda()
    c = torch.empty((m, n), device="cuda", dtype=torch.float16)
    compiled = wasp_gemm[(1,)](
        a, b, c, m, n, k, BM=m, BN=n, BK=16, num_stages=2,
        use_async_copy=True, use_block_pingpong=False,
        wasp_wdra=True, wasp_partition_warps=(4, 4, 4),
        wasp_partition_regs=(88, 144, 140),
        matrix_instr_nonkdim=16, optimize_epilogue=False)
    expected = (a_host.float() @ b_host.float()).half()
    torch.testing.assert_close(c.cpu(), expected, atol=0.02, rtol=0.02)
    assert compiled.metadata.uses_bank_conflict_aware_lds_placement
    assert "#ttg.hcu_buffer_lds_shared" in compiled.asm["ttgir"]
    assert "ttg.memdesc_subslice" in compiled.asm["ttgir"]
    assert "amdg.buffer_load_to_local" in compiled.asm["ttgir"]
    assert "ttg.local_barrier" not in compiled.asm["ttgir"]
    assert "llvm.amdgcn.raw.ptr.buffer.load.async.lds" in compiled.asm["llir"]
    assert re.search(r"buffer_load_\w+.*\blds\b", compiled.asm["amdgcn"])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("column_b", [False, True])
@pytest.mark.parametrize("bm,bn,bk,waves", [(64, 64, 16, 4),
                                         (128, 64, 32, 4),
                                         (128, 128, 16, 8)])
def test_standalone_buffer_lds_copy(monkeypatch, dtype, column_b, bm, bn, bk, waves):
    if not torch.cuda.is_available():
        pytest.skip("HCU device required")
    if "gfx936" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("gfx936 validation")
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", "1")
    k = bk * 7
    a = torch.randn((bm, k), device="cuda", dtype=dtype)
    b = torch.randn((bn, k) if column_b else (k, bn),
                    device="cuda", dtype=dtype)
    if column_b:
        b = b.T
    c = torch.empty((bm, bn), device="cuda", dtype=dtype)
    compiled = direct_gemm[(1,)](
        a, b, c, bm, bn, k, BM=bm, BN=bn, BK=bk,
        B_COLUMN_MAJOR=column_b, num_warps=waves, num_stages=2,
        use_block_pingpong=False, use_async_copy=True,
        matrix_instr_nonkdim=16, optimize_epilogue=False)
    torch.testing.assert_close(c, a @ b, atol=0.02, rtol=0.02)
    assert compiled.metadata.uses_bank_conflict_aware_lds_placement
    assert "isTransposed = true" not in compiled.asm["ttgir"]
    assert "hcu.block_pingpong" not in compiled.asm["ttgir"]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("column_b", [False, True])
@pytest.mark.parametrize("bm,bn,bk,waves", [(64, 64, 16, 4),
                                             (128, 128, 32, 8)])
def test_nmz_buffer_lds_copy(monkeypatch, dtype, column_b, bm, bn, bk, waves):
    """MmacLayout changes C, not the BufferLds A/B placement contract."""
    if not torch.cuda.is_available():
        pytest.skip("HCU device required")
    if "gfx938" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("gfx938 validation")
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", "1")
    k = bk * 7
    a = torch.randn((bm, k), device="cuda", dtype=dtype)
    b = torch.randn((bn, k) if column_b else (k, bn),
                    device="cuda", dtype=dtype)
    if column_b:
        b = b.T
    c = torch.empty((bm, bn), device="cuda", dtype=dtype)
    compiled = direct_gemm[(1,)](
        a, b, c, bm, bn, k, BM=bm, BN=bn, BK=bk,
        B_COLUMN_MAJOR=column_b, num_warps=waves, num_stages=2,
        use_block_pingpong=False, use_async_copy=True,
        matrix_instr_nonkdim=16, optimize_epilogue=False)
    torch.testing.assert_close(c, a @ b, atol=0.02, rtol=0.02)
    assert compiled.metadata.uses_bank_conflict_aware_lds_placement
    assert "#ttg.hcu_buffer_lds_shared" in compiled.asm["ttgir"]
    assert "llvm.amdgcn.raw.ptr.buffer.load.async.lds" in compiled.asm["llir"]
    assert re.search(r"buffer_load_\w+.*\blds\b", compiled.asm["amdgcn"])
    if not column_b:
        assert "ds_read_m32x16_b16" in compiled.asm["amdgcn"]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("column_b", [False, True], ids=["b-nonk-contiguous", "b-k-contiguous"])
def test_shaobo_buffer_lds_copy(monkeypatch, dtype, column_b):
    """Cover gfx946 buffer-to-LDS copies with both weight layouts."""
    if not torch.cuda.is_available():
        pytest.skip("HCU device required")
    if "gfx946" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("gfx946 validation")
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", "1")
    m, n, k = 64, 64, 112
    generator = torch.Generator().manual_seed(946)
    a_host = torch.randn((m, k), generator=generator).to(dtype)
    b_host = torch.randn((n, k) if column_b else (k, n),
                         generator=generator).to(dtype)
    a = a_host.cuda()
    b = b_host.cuda()
    if column_b:
        b_host = b_host.T
        b = b.T
    expected = (a_host.float() @ b_host.float()).to(dtype)
    c = torch.empty((m, n), device="cuda", dtype=dtype)
    compiled = direct_gemm[(1,)](
        a, b, c, m, n, k, BM=m, BN=n, BK=16,
        B_COLUMN_MAJOR=column_b, num_warps=4, num_stages=2,
        use_block_pingpong=False, use_async_copy=True,
        matrix_instr_nonkdim=16, optimize_epilogue=False)
    torch.testing.assert_close(c.cpu(), expected, atol=0.02, rtol=0.02)
    assert compiled.metadata.mmac_layout_force == -1
    assert compiled.metadata.uses_bank_conflict_aware_lds_placement
    assert "mmacLayout = 2" in compiled.asm["ttgir"]
    assert "#ttg.hcu_buffer_lds_shared" in compiled.asm["ttgir"]
    assert "llvm.amdgcn.raw.ptr.buffer.load.async.lds" in compiled.asm["llir"]
    assert re.search(r"buffer_load_\w+.*\blds\b", compiled.asm["amdgcn"])
    if not column_b:
        assert "ds_read_m32x16_b16" in compiled.asm["amdgcn"]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("bm,bn,bk,waves", [
    pytest.param(128, 64, 32, 4, id="128x64x32-w4"),
    pytest.param(128, 128, 32, 8, id="128x128x32-w8"),
])
def test_shaobo_async_copy_matrix_read(monkeypatch, dtype, bm, bn, bk, waves):
    """Exercise gfx946 async BufferLds through its native b16 matrix read."""
    if not torch.cuda.is_available():
        pytest.skip("HCU device required")
    if "gfx946" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("gfx946 validation")
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", "1")
    monkeypatch.setenv("TRITON_HCU_ENABLE_DS_READ_M", "1")
    k = bk * 7
    generator = torch.Generator().manual_seed(946 + bm + bn)
    a_host = torch.randn((bm, k), generator=generator).to(dtype)
    b_host = torch.randn((k, bn), generator=generator).to(dtype)
    expected = (a_host.float() @ b_host.float()).to(dtype)
    a = a_host.cuda()
    b = b_host.cuda()
    c = torch.empty((bm, bn), device="cuda", dtype=dtype)
    compiled = direct_gemm[(1,)](
        a, b, c, bm, bn, k, BM=bm, BN=bn, BK=bk,
        num_warps=waves, num_stages=2, use_block_pingpong=False,
        use_async_copy=True, matrix_instr_nonkdim=16,
        optimize_epilogue=False)
    torch.testing.assert_close(c.cpu(), expected, atol=0.02, rtol=0.02)
    assert compiled.metadata.use_async_copy
    assert compiled.metadata.uses_bank_conflict_aware_lds_placement
    assert "mmacLayout = 2" in compiled.asm["ttgir"]
    assert "#ttg.hcu_buffer_lds_shared" in compiled.asm["ttgir"]
    assert "llvm.amdgcn.raw.ptr.buffer.load.async.lds" in compiled.asm["llir"]
    assert re.search(r"buffer_load_\w+.*\blds\b", compiled.asm["amdgcn"])
    assert "ds_read_m32x16_b16" in compiled.asm["amdgcn"]
