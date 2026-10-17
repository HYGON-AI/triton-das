# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""Buffer-to-LDS fill, rotation and drain; no dependency on boltOps or a wheel patch."""
import pytest
import re
import torch
import triton
import triton.language as tl
from buffer_lds_test_utils import bf16_schedule_gemm, _runtime_k_gemm


from test_buffer_lds_copy import direct_gemm as _transport_gemm


@triton.jit
def direct_gemm(a, b, c, m, n, k,
                BM: tl.constexpr = 256, BN: tl.constexpr = 256,
                BK: tl.constexpr = 16, REVERSE_A: tl.constexpr = False,
                A_COLUMN_MAJOR: tl.constexpr = False,
                B_COLUMN_MAJOR: tl.constexpr = False,
                BATCHED: tl.constexpr = False,
                AFFINE_COORDS: tl.constexpr = False,
                K_STEP_RANGE: tl.constexpr = False):
    _transport_gemm(a, b, c, m, n, k, BM, BN, BK, REVERSE_A,
                    A_COLUMN_MAJOR, B_COLUMN_MAJOR, BATCHED, AFFINE_COORDS,
                    K_STEP_RANGE)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("k", [4064, 4096])
@pytest.mark.parametrize("optimize_epilogue", [False, True])
def test_planned_copy_cache_swizzle_stride(monkeypatch, dtype, k, optimize_epilogue):
    if not torch.cuda.is_available():
        pytest.skip("HCU device required")
    if "gfx936" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("gfx936 buffer-to-LDS policy")
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", "1")
    monkeypatch.setenv("TRITON_HCU_ENABLE_DS_READ_M", "1")
    torch.manual_seed(42)
    a = torch.randn((4096, k), device="cuda", dtype=dtype)
    b = torch.randn((k, 4096), device="cuda", dtype=dtype)
    c = torch.empty((4096, 4096), device="cuda", dtype=dtype)
    h = direct_gemm[(256,)](
        a, b, c, 4096, 4096, k, num_warps=8, num_stages=4,
        K_STEP_RANGE=True,
        use_block_pingpong=True, use_async_copy=True,
        buffer_cache_swizzle=True, block_pingpong_async_global_prefetch=4,
        block_pingpong_async_local_prefetch=1, matrix_instr_nonkdim=16,
        optimize_epilogue=optimize_epilogue,
        amdgpu_enable_max_ilp_scheduling_strategy=False)
    torch.testing.assert_close(c, a @ b, atol=0.02, rtol=0.02)
    assert h.metadata.uses_bank_conflict_aware_lds_placement
    assert "llvm.amdgcn.raw.ptr.buffer.load.async.lds" in h.asm["llir"]
    assert "idxen offen lds" not in h.asm["amdgcn"]
    assert re.search(r"buffer_load_dwordx4 .* offen lds", h.asm["amdgcn"])
    # Global producer interleave and LDS consumer placement must agree.
    assert "rowsPerChunk = 4, wrapCount = 4, wrapStepBytes = 16" in h.asm["ttgir"]
    assert "wavesPerRowGroup = 2" in h.asm["ttgir"]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("k", [4112, 5120, 8192])
def test_planned_copy_crosses_descriptor_stride(monkeypatch, dtype, k):
    if not torch.cuda.is_available():
        pytest.skip("HCU device required")
    if "gfx936" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("gfx936 buffer-to-LDS policy")
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", "1")
    monkeypatch.setenv("TRITON_HCU_ENABLE_DS_READ_M", "1")
    torch.manual_seed(42)
    m = n = 256
    a = torch.randn((m, k), device="cuda", dtype=dtype)
    b = torch.randn((k, n), device="cuda", dtype=dtype)
    c = torch.empty((m, n), device="cuda", dtype=dtype)

    def run():
        return direct_gemm[(1,)](
            a, b, c, m, n, k, num_warps=8, num_stages=5,
            use_block_pingpong=True, use_async_copy=True,
            buffer_cache_swizzle=True, block_pingpong_async_global_prefetch=4,
            block_pingpong_async_local_prefetch=1, matrix_instr_nonkdim=16,
            optimize_epilogue=True,
            amdgpu_enable_max_ilp_scheduling_strategy=False)

    h = run()
    torch.testing.assert_close(c, a @ b, atol=0.02, rtol=0.02)
    assert h.metadata.uses_bank_conflict_aware_lds_placement
    assert "llvm.amdgcn.raw.ptr.buffer.load.async.lds" in h.asm["llir"]
    # The K loop crosses the descriptor cache stride. Raw addressing must keep
    # using the original byte offset with the common descriptor unchanged.
    for generation in sorted({0, 255, 256, k // 16 - 1}):
        a.zero_()
        b.zero_()
        a[:, generation * 16:(generation + 1) * 16] = 1
        b[generation * 16:(generation + 1) * 16, :] = 1
        run()
        torch.testing.assert_close(c, torch.full_like(c, 16), atol=0, rtol=0)


def check_buffer_to_lds(monkeypatch, k, seed, active_tile=None, reverse_a=False):
    if not torch.cuda.is_available():
        pytest.skip("HCU device required")
    if "gfx936" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("gfx936 buffer-to-LDS policy")
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", "1")
    monkeypatch.setenv("TRITON_HCU_ENABLE_DS_READ_M", "1")
    monkeypatch.setenv("TRITON_HIP_USE_ASYNC_COPY", "0")
    torch.manual_seed(seed)
    m, n = 234, 512
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16)
    if active_tile is not None:
        a.zero_()
        b.zero_()
        a[:, active_tile * 16:(active_tile + 1) * 16] = 1
        b[active_tile * 16:(active_tile + 1) * 16, :] = 1
    c = torch.empty((m, n), device="cuda", dtype=torch.float16)
    h = direct_gemm[(triton.cdiv(m, 256) * triton.cdiv(n, 256),)](
        a, b, c, m, n, k, REVERSE_A=reverse_a, num_warps=8, num_stages=2,
        use_block_pingpong=True, use_async_copy=True,
        matrix_instr_nonkdim=16, optimize_epilogue=False,
        amdgpu_enable_max_ilp_scheduling_strategy=False)
    torch.testing.assert_close(c, (a.flip(0) if reverse_a else a) @ b,
                               atol=0.02, rtol=0.02)
    assert h.metadata.shared <= 65536
    assert h.metadata.use_async_copy
    assert h.metadata.uses_bank_conflict_aware_lds_placement
    assert "hcu.buffer_to_lds.selected" not in h.asm["ttgir"]
    assert "hcu.buffer_to_lds =" not in h.asm["ttgir"]
    assert "hcu.block_pingpong.static_buffers" in h.asm["ttgir"]
    assert "ttg.async_commit_group" in h.asm["ttgir"]
    assert "hcu.buffer_to_lds.published" not in h.asm["ttgir"]
    assert "ds_read_m32x16_b16" in h.asm["amdgcn"]
    assert "ds_read_b64" in h.asm["amdgcn"]
    assert "buffer_load_dwordx4" in h.asm["amdgcn"]
    assert "llvm.amdgcn.raw.ptr.buffer.load.async.lds" in h.asm["llir"]
    assert "llvm.amdgcn.asyncmark" in h.asm["llir"]
    assert "llvm.amdgcn.wait.asyncmark" in h.asm["llir"]
    assert 'asm sideeffect "ds_read' not in h.asm["llir"]
    assert 'asm sideeffect "s_waitcnt vmcnt' not in h.asm["llir"]


@pytest.mark.parametrize("n,k", [(4928, 1232), (1232, 2464)])
def test_batched_four_slot_epilogue_drain(monkeypatch, n, k):
    if not torch.cuda.is_available():
        pytest.skip("HCU device required")
    if "gfx936" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("gfx936 buffer-to-LDS policy")
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", "1")
    monkeypatch.setenv("TRITON_HCU_ENABLE_DS_READ_M", "1")
    torch.manual_seed(42)
    batch, m = 56, 234
    a = torch.randn((batch, m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((batch, k, n), device="cuda", dtype=torch.float16)
    out = torch.empty((batch, m, n), device="cuda", dtype=torch.float16)
    expected = torch.bmm(a, b)
    for _ in range(3):
        h = direct_gemm[(triton.cdiv(n, 256), batch)](
            a, b, out, m, n, k, BATCHED=True, num_warps=8, num_stages=2,
            K_STEP_RANGE=True,
            use_block_pingpong=True, use_async_copy=True,
            matrix_instr_nonkdim=16, optimize_epilogue=False,
            amdgpu_enable_max_ilp_scheduling_strategy=False)
        torch.testing.assert_close(out, expected, atol=0.02, rtol=0.02)
    assert 'llvm.amdgcn.wait.asyncmark(i16 4)' in h.asm['llir']
    assert 'llvm.amdgcn.wait.asyncmark(i16 0)' in h.asm['llir']
    assert 'pending_copy_groups' not in h.asm['ttgir']


@pytest.mark.parametrize("k", [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 256, 1232, 2464])
@pytest.mark.parametrize("seed", [0, 42])
def test_buffer_to_lds_generations(monkeypatch, k, seed):
    check_buffer_to_lds(monkeypatch, k, seed)


@pytest.mark.parametrize("active_tile", range(5))
def test_buffer_to_lds_first_wrap(monkeypatch, active_tile):
    check_buffer_to_lds(monkeypatch, 80, 42, active_tile)


def test_buffer_to_lds_preserves_row_indices(monkeypatch):
    # Reconstructing addresses from an inferred positive row stride would
    # silently lose this indexing; the offset layout conversion must retain it.
    check_buffer_to_lds(monkeypatch, 80, 42, reverse_a=True)


@pytest.mark.parametrize("active_tile", [0, 3, 4, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_buffer_to_lds_multitransfer_slot_addresses(monkeypatch, active_tile, dtype):
    """Exercise register transfer offsets and nonzero/reused four-slot bases."""
    if not torch.cuda.is_available():
        pytest.skip("HCU device required")
    if "gfx936" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("gfx936 buffer-to-LDS policy")
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", "1")
    monkeypatch.setenv("TRITON_HCU_ENABLE_DS_READ_M", "1")
    m, n, bk, k = 64, 64, 64, 576
    a = torch.zeros((m, k), device="cuda", dtype=dtype)
    b = torch.zeros((k, n), device="cuda", dtype=dtype)
    rows = torch.arange(m, device="cuda")[:, None]
    ks = torch.arange(bk, device="cuda")
    cols = torch.arange(n, device="cuda")[None, :]
    # Exactly representable output; distinguish rows, columns and K segments
    # without a tolerance hiding an incorrect transfer or slot address.
    a[:, active_tile * bk:(active_tile + 1) * bk] = (rows + ks) % 2
    b[active_tile * bk:(active_tile + 1) * bk, :] = (ks[:, None] // 8 + cols) % 3
    c = torch.empty((m, n), device="cuda", dtype=dtype)
    compiled = direct_gemm[(1,)](
        a, b, c, m, n, k, BM=64, BN=64, BK=bk, num_warps=4,
        num_stages=2, use_block_pingpong=True, use_async_copy=True,
        matrix_instr_nonkdim=16, optimize_epilogue=False,
        amdgpu_enable_max_ilp_scheduling_strategy=False)
    torch.testing.assert_close(c, a @ b, atol=0, rtol=0)
    ttgir = compiled.asm["ttgir"]
    assert "hcu.block_pingpong.static_buffers" in ttgir
    copies = re.findall(r"rows = (\d+), rowBytes = (\d+), waves = (\d+), copyBytesPerLane = (\d+)", ttgir)
    assert copies
    # A complete wave region contains more than one 64-lane transfer.
    assert any((p[0] * p[1] // p[2]) > p[3] * 64
               for p in (list(map(int, copy)) for copy in copies))
    assert "llvm.amdgcn.raw.ptr.buffer.load.async.lds" in compiled.asm["llir"]


@pytest.mark.parametrize("bm,bn,bk,waves", [
    (256, 256, 16, 8), (128, 128, 16, 4), (128, 256, 16, 4),
    (256, 128, 16, 4), (128, 128, 32, 4), (64, 128, 32, 2),
    (32, 64, 16, 1), (16, 32, 16, 1), (128, 128, 16, 16),
    (16, 16, 16, 1),
    (64, 64, 64, 4), (32, 32, 128, 2),
])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_buffer_to_lds_dynamic_contract(monkeypatch, bm, bn, bk, waves, dtype):
    if not torch.cuda.is_available():
        pytest.skip("HCU device required")
    if "gfx936" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("gfx936 buffer-to-LDS policy")
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", "1")
    monkeypatch.setenv("TRITON_HCU_ENABLE_DS_READ_M", "1")
    monkeypatch.setenv("TRITON_HIP_USE_ASYNC_COPY", "0")
    torch.manual_seed(42)
    m, n, k = bm - 3, bn * 2, bk * 9
    a = torch.randn((m, k), device="cuda", dtype=dtype)
    b = torch.randn((k, n), device="cuda", dtype=dtype)
    c = torch.empty((m, n), device="cuda", dtype=dtype)
    compiled = direct_gemm[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
        a, b, c, m, n, k, BM=bm, BN=bn, BK=bk, num_warps=waves,
        num_stages=2, use_block_pingpong=True, use_async_copy=True,
        matrix_instr_nonkdim=16, optimize_epilogue=False,
        amdgpu_enable_max_ilp_scheduling_strategy=False)
    torch.testing.assert_close(c, a @ b, atol=0.02, rtol=0.02)
    assert "#ttg.hcu_buffer_lds_shared" in compiled.asm["ttgir"]
    assert "hcu.block_pingpong.static_buffers" in compiled.asm["ttgir"]
    assert "buffer_load_dword" in compiled.asm["amdgcn"]
    if bn >= 32:
        assert "ds_read_m32x16_b16" in compiled.asm["amdgcn"]


@pytest.mark.parametrize("column_a,column_b", [(False, True), (True, False), (True, True)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_buffer_to_lds_source_order(monkeypatch, column_a, column_b, dtype):
    if not torch.cuda.is_available():
        pytest.skip("HCU device required")
    if "gfx936" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("gfx936 buffer-to-LDS policy")
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", "1")
    monkeypatch.setenv("TRITON_HCU_ENABLE_DS_READ_M", "1")
    monkeypatch.setenv("TRITON_HIP_USE_ASYNC_COPY", "0")
    torch.manual_seed(42)
    m, n, k = 128, 256, 288
    a = torch.randn((k, m) if column_a else (m, k), device="cuda", dtype=dtype)
    b = torch.randn((n, k) if column_b else (k, n), device="cuda", dtype=dtype)
    if column_a:
        a = a.T
    if column_b:
        b = b.T
    c = torch.empty((m, n), device="cuda", dtype=dtype)
    compiled = direct_gemm[(2,)](
        a, b, c, m, n, k, BM=128, BN=128, BK=32, num_warps=4,
        A_COLUMN_MAJOR=column_a, B_COLUMN_MAJOR=column_b,
        num_stages=2, use_block_pingpong=True, use_async_copy=True,
        matrix_instr_nonkdim=16, optimize_epilogue=False,
        amdgpu_enable_max_ilp_scheduling_strategy=False)
    torch.testing.assert_close(c, a @ b, atol=0.02, rtol=0.02)
    assert "#ttg.hcu_buffer_lds_shared" in compiled.asm["ttgir"]
    assert "hcu.block_pingpong.static_buffers" in compiled.asm["ttgir"]


# Runtime reductions, including one-binary K coverage.

@pytest.mark.parametrize("column_b", [False, True])
@pytest.mark.parametrize("k", [256, 4096, 5120])
@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_runtime_k_gemm_gpu(monkeypatch, column_b, k, dtype_name):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("HCU device required")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch not in ("gfx936", "gfx938"):
        pytest.skip("gfx936/gfx938 schedule validation")
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", "1")
    monkeypatch.setenv("TRITON_HCU_ENABLE_DS_READ_M", "1")
    torch.manual_seed(42)
    dtype = getattr(torch, dtype_name)
    a = torch.randn((256, k), device="cuda", dtype=dtype)
    b = torch.randn((256, k) if column_b else (k, 256),
                    device="cuda", dtype=dtype)
    c = torch.empty((256, 256), device="cuda", dtype=dtype)
    compiled = _runtime_k_gemm[(1,)](
        a, b, c, k, column_b, 32 if column_b else 16,
        use_async_copy=True, use_block_pingpong=True, num_warps=8,
        num_stages=2, block_pingpong_async_global_prefetch=2 if column_b else 4,
        matrix_instr_nonkdim=16, optimize_epilogue=False)
    ref = (a.float() @ (b.T if column_b else b).float()).to(dtype)
    torch.testing.assert_close(c, ref, atol=0.125 if dtype_name == "bfloat16" else 0.02,
                               rtol=0.02)
    assert compiled.metadata.uses_bank_conflict_aware_lds_placement


@pytest.mark.parametrize("column_b,g", [(False, 3), (True, 2)])
@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_dynamic_k_generic_buffer_lds_pingpong_gpu(monkeypatch, column_b, g, dtype_name):
    import torch
    from buffer_lds_test_utils import bf16_schedule_gemm

    if not torch.cuda.is_available():
        pytest.skip("HCU device required")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch not in ("gfx936", "gfx938"):
        pytest.skip("gfx936/gfx938 schedule validation")
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", "1")
    monkeypatch.setenv("TRITON_HCU_ENABLE_DS_READ_M", "1")
    torch.manual_seed(51)
    dtype = getattr(torch, dtype_name)
    bk = 32 if column_b else 16
    # Include an empty steady state, fill/drain boundaries, and non-multiples
    # of the ring's unroll factor. No static reduction proof is supplied.
    for tiles in sorted({1, g, g + 1, 2 * g - 1, 2 * g, 2 * g + 1, 16}):
        k = tiles * bk
        a = torch.randn((256, k), device="cuda", dtype=dtype)
        b = torch.randn((256, k) if column_b else (k, 256), device="cuda", dtype=dtype)
        if column_b:
            b = b.T
        c = torch.empty((256, 256), device="cuda", dtype=dtype)
        compiled = bf16_schedule_gemm[(1,)](
            a, b, c, k, a.stride(0), *b.stride(), BK=bk,
            use_async_copy=True, use_block_pingpong=True, num_warps=8,
            num_stages=2, block_pingpong_async_global_prefetch=g,
            matrix_instr_nonkdim=16, optimize_epilogue=False)
        torch.testing.assert_close(c, (a.float() @ b.float()).to(dtype),
                                   atol=0.125 if dtype_name == "bfloat16" else 0.02,
                                   rtol=0.02, msg=f"generic BufferLds tiles={tiles}")
        assert compiled.metadata.uses_bank_conflict_aware_lds_placement
        assert "hcu.block_pingpong.applied" in compiled.asm["ttgir"]
        copies = [line for line in compiled.asm["ttgir"].splitlines()
                  if "amdg.buffer_load_to_local " in line]
        assert copies
        assert all("stride =" not in line for line in copies)


@triton.jit(do_not_specialize=["K"])
def _runtime_k_tt_gemm(a, b, c, K, A_STRIDE: tl.constexpr):
    tl.assume(K >= 0)
    tl.assume(K % 16 == 0)
    m = tl.arange(0, 256)
    n = tl.arange(0, 256)
    k = tl.arange(0, 16)
    ap = a + m[:, None] * A_STRIDE + k[None, :]
    bp = b + k[:, None] * 256 + n[None, :]
    acc = tl.zeros((256, 256), tl.float32)
    for _ in range(0, K, 16):
        acc = tl.dot(tl.load(ap), tl.load(bp), acc)
        ap += 16
        bp += 16 * 256
    tl.store(c + m[:, None] * 256 + n[None, :], acc.to(c.dtype.element_ty))


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("mmac_layout", [1, 3])
def test_runtime_k_wave_groups_reuse_one_binary(monkeypatch, dtype_name,
                                               mmac_layout):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("HCU device required")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch not in ("gfx936", "gfx938"):
        pytest.skip("gfx936/gfx938 wave-group validation")
    if arch == "gfx936" and mmac_layout == 3:
        pytest.skip("hardware accumulator transpose requires MMAC_LAYOUT")
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", "1")
    monkeypatch.setenv("TRITON_HCU_ENABLE_DS_READ_M", "1")
    torch.manual_seed(64)
    dtype = getattr(torch, dtype_name)
    a = torch.randn((256, 5120), device="cuda", dtype=dtype)
    b = torch.randn((5120, 256), device="cuda", dtype=dtype)
    c = torch.empty((256, 256), device="cuda", dtype=dtype)
    hashes = set()
    # Zero-trip, fill/drain, wrap, remainder and full GEMMs share runtime K.
    for k in (0, 16, 48, 64, 80, 112, 128, 144, 240, 256, 272, 4064, 4096, 5120):
        h = _runtime_k_tt_gemm[(1,)](
            a, b, c, k, a.stride(0), num_warps=8, num_stages=5,
            use_async_copy=True, use_block_pingpong=True,
            block_pingpong_async_global_prefetch=4, block_pingpong_async_local_prefetch=1,
            matrix_instr_nonkdim=16, mmac_layout_force=mmac_layout,
            buffer_cache_swizzle=True, optimize_epilogue=True)
        torch.testing.assert_close(
            c, (a[:, :k].float() @ b[:k].float()).to(dtype), rtol=.02,
            atol=.125 if dtype_name == "bfloat16" else .02, msg=f"K={k}")
        hashes.add(h.hash)
    assert len(hashes) == 1, "runtime K must not specialize the compiled kernel"
    assert "hcu.planned_async" not in h.asm["ttgir"]
    assert "amdg.memory_counter_wait ds(8)" in h.asm["ttgir"]
    assert h.asm["ttgir"].count("hcu.block_pingpong.static_buffers") == 2


# Prefetch distances, slot reuse, and the ordinary BlockPingpong baseline.

@pytest.mark.parametrize('g,r', [(2, 1), (3, 1), (3, 2), (4, 1), (4, 2)])
@pytest.mark.parametrize('k', [16, 80, 256])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_distances(monkeypatch, g, r, k, dtype):
    if not torch.cuda.is_available() or 'gfx936' not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip('gfx936 required')
    monkeypatch.setenv('AMDGCN_USE_BUFFER_OPS', '1')
    monkeypatch.setenv('TRITON_HCU_ENABLE_DS_READ_M', '1')
    torch.manual_seed(42)
    a = torch.randn((256, k), device='cuda', dtype=dtype)
    b = torch.randn((k, 256), device='cuda', dtype=dtype)
    c = torch.empty((256, 256), device='cuda', dtype=dtype)
    h = direct_gemm[(1,)](a, b, c, 256, 256, k, num_warps=8,
                         K_STEP_RANGE=True,
                         num_stages=2, use_block_pingpong=True,
                         use_async_copy=True, block_pingpong_async_global_prefetch=g,
                         block_pingpong_async_local_prefetch=r, matrix_instr_nonkdim=16,
                         optimize_epilogue=False)
    torch.testing.assert_close(c, a @ b, atol=0.02, rtol=0.02)
    # The compiler serializes launch options into metadata without a backend
    # assignment; exercise explicit depths through real kernel compilation.
    assert h.metadata.block_pingpong_async_global_prefetch == g
    assert h.metadata.block_pingpong_async_local_prefetch == r
    assert h.metadata.uses_bank_conflict_aware_lds_placement
    if k == 256:
        ttgir = h.asm['ttgir']
        # Include the first loop's prefetched pair, not just the register
        # prologue. Newer DMA generations may remain pending across entry.
        initial_wait = re.search(r'ttg.async_wait \{num = (\d+) : i32\}', ttgir)
        assert initial_wait and int(initial_wait[1]) == 2 * (g - r - 1)
        assert 'hcu.block_pingpong.static_buffers' in ttgir
        # Epilogue scratch may exceed the operand rings' footprint.
        assert h.metadata.shared <= 65536
        assert ttgir.count('= ttg.local_alloc :') == 2 * g
        assert f'tt.num_stages = {g + 2} :' in ttgir
        # The initial B matrix read and A scalar reads consume slots published
        # by one shared wait/barrier.  Do not serialize them with a second
        # publication boundary between the two local loads.
        first_load = ttgir.index('ttg.local_load')
        second_load = ttgir.index('ttg.local_load', first_load + 1)
        between_initial_loads = ttgir[first_load:second_load]
        assert 'ttg.async_wait' not in between_initial_loads
        assert 'ttg.local_barrier' not in between_initial_loads


@pytest.mark.parametrize('g,r', [(2, 1), (3, 1), (3, 2), (4, 1), (4, 2)])
@pytest.mark.parametrize('column_major', [False, True])
def test_prefetch_first_wrap(monkeypatch, g, r, column_major):
    if not torch.cuda.is_available() or 'gfx936' not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip('gfx936 required')
    monkeypatch.setenv('AMDGCN_USE_BUFFER_OPS', '1')
    monkeypatch.setenv('TRITON_HCU_ENABLE_DS_READ_M', '1')
    m, n, k = 256, 512, 256
    a = torch.zeros((m, k), device='cuda', dtype=torch.float16)
    b = torch.zeros((k, n), device='cuda', dtype=torch.float16)
    if column_major:
        b = b.T.contiguous().T
    c = torch.empty((m, n), device='cuda', dtype=torch.float16)
    # Distinct impulses before/at/after wrap expose stale or skipped generations.
    for active in range(g + r + 2):
        a.zero_()
        b.zero_()
        a[:, active*16:(active+1)*16] = 1
        b[active*16:(active+1)*16, :] = active + 1
        h = direct_gemm[(2,)](a, b, c, m, n, k, num_warps=8,
                             K_STEP_RANGE=True,
                             B_COLUMN_MAJOR=column_major,
                             num_stages=2, use_block_pingpong=True,
                             use_async_copy=True, block_pingpong_async_global_prefetch=g,
                             block_pingpong_async_local_prefetch=r, matrix_instr_nonkdim=16)
        torch.testing.assert_close(c, torch.full_like(c, 16*(active+1)), atol=0, rtol=0)
        assert h.metadata.uses_bank_conflict_aware_lds_placement


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('k', [240, 256, 272])
def test_static_dma_first_preserves_dynamic_tail(monkeypatch, dtype, k):
    if not torch.cuda.is_available() or 'gfx936' not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip('gfx936 required')
    monkeypatch.setenv('AMDGCN_USE_BUFFER_OPS', '1')
    a = torch.zeros((256, k), device='cuda', dtype=dtype)
    b = torch.zeros((k, 256), device='cuda', dtype=dtype)
    c = torch.empty((256, 256), device='cuda', dtype=dtype)
    for generation in range(k // 16):
        a.zero_()
        b.zero_()
        a[:, generation*16:(generation+1)*16] = 1
        b[generation*16:(generation+1)*16, :] = generation + 1
        h = direct_gemm[(1,)](
            a, b, c, 256, 256, k, num_warps=8, num_stages=2,
            K_STEP_RANGE=True,
            use_block_pingpong=True, use_async_copy=True,
            block_pingpong_async_global_prefetch=4, block_pingpong_async_local_prefetch=1,
            matrix_instr_nonkdim=16, optimize_epilogue=False)
        torch.testing.assert_close(c, torch.full_like(c,16*(generation+1)),
                                   atol=0, rtol=0)
    # The main loop has fixed LDS slots and issues DMA first. Its indexed remainder
    # deliberately retains read-before-refill; copying the hot ordering to it
    # used to lose part of generation 9 for K=256.
    loops = h.asm['ttgir'].split('= scf.for ')[1:]
    assert len(loops) >= 2
    assert loops[0].index('amdg.buffer_load_to_local') < loops[0].index('ttg.local_load')
    assert loops[1].index('ttg.local_load') < loops[1].index('amdg.buffer_load_to_local')


@pytest.mark.parametrize('g,r', [(g, r) for g in range(2, 9) for r in range(1, g)])
def test_deep_prefetch_generations(monkeypatch, g, r,
                                   dtype=torch.float16, column_major=False):
    if not torch.cuda.is_available() or 'gfx936' not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip('gfx936 required')
    monkeypatch.setenv('AMDGCN_USE_BUFFER_OPS', '1')
    monkeypatch.setenv('TRITON_HCU_ENABLE_DS_READ_M', '1')
    m, n, k = 128, 256, 512
    torch.manual_seed(42)
    a = torch.randn((m, k), device='cuda', dtype=dtype)
    b = torch.randn((k, n), device='cuda', dtype=dtype)
    if column_major:
        b = b.T.contiguous().T
    c = torch.empty((m, n), device='cuda', dtype=dtype)
    def launch():
        return direct_gemm[(2,)](a, b, c, m, n, k, BM=128, BN=128, BK=16,
                                 K_STEP_RANGE=True,
                                 B_COLUMN_MAJOR=column_major,
                                 num_warps=8, num_stages=2,
                                 use_block_pingpong=True, use_async_copy=True,
                                 block_pingpong_async_global_prefetch=g, block_pingpong_async_local_prefetch=r,
                                 matrix_instr_nonkdim=16)
    h = launch()
    torch.testing.assert_close(c, a @ b, atol=.02, rtol=.02)
    assert h.metadata.uses_bank_conflict_aware_lds_placement
    assert 'hcu.block_pingpong.static_buffers' in h.asm['ttgir']
    assert h.asm['ttgir'].count('= ttg.local_alloc :') == 2*g
    assert h.metadata.shared <= 65536
    for active in {0, g-1, g, g+r, k//16-1}:
        a.zero_()
        b.zero_()
        a[:, active*16:(active+1)*16] = 1
        b[active*16:(active+1)*16, :] = active+1
        launch()
        torch.testing.assert_close(c, torch.full_like(c,16*(active+1)),atol=0,rtol=0)


@pytest.mark.parametrize('g,r', [(5,4), (8,7)])
def test_deep_bf16_column_major(monkeypatch, g, r):
    test_deep_prefetch_generations(monkeypatch, g, r, torch.bfloat16, True)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('g', [3, 4])
@pytest.mark.parametrize('k', [128, 240, 272, 4096])
def test_static_prefetch_slot_reuse(monkeypatch, dtype, g, k):
    if not torch.cuda.is_available() or 'gfx936' not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip('gfx936 required')
    monkeypatch.setenv('AMDGCN_USE_BUFFER_OPS', '1')
    monkeypatch.setenv('TRITON_HCU_ENABLE_DS_READ_M', '1')
    torch.manual_seed(73)
    m = n = 256
    a = torch.randn((m, k), device='cuda', dtype=dtype)
    b = torch.randn((k, n), device='cuda', dtype=dtype)
    c = torch.empty((m, n), device='cuda', dtype=dtype)

    def launch():
        return direct_gemm[(4,)](
            a, b, c, m, n, k, BM=128, BN=128, BK=16,
            K_STEP_RANGE=True,
            num_warps=4, num_stages=5, use_block_pingpong=True,
            use_async_copy=True, block_pingpong_async_global_prefetch=g,
            block_pingpong_async_local_prefetch=2, matrix_instr_nonkdim=16,
            buffer_cache_swizzle=True, optimize_epilogue=True,
            amdgpu_enable_max_ilp_scheduling_strategy=False)

    h = launch()
    torch.testing.assert_close(c, a @ b, atol=.02, rtol=.02)
    # Slot reuse keeps the explicit LDS drain, not a synthetic register use.
    assert re.search(r'amdg.memory_counter_wait ds\(0\)', h.asm['ttgir'])
    assert 'hcu.block_pingpong.fragment_retirement' not in h.asm['ttgir']
    assert 'asm sideeffect ""' not in h.asm['llir']
    for generation in sorted({0, min(g - 1, k // 16 - 1), min(g, k // 16 - 1), k // 16 - 1}):
        a.zero_()
        b.zero_()
        a[:, generation * 16:(generation + 1) * 16] = 1
        b[generation * 16:(generation + 1) * 16, :] = 1
        launch()
        torch.testing.assert_close(c, torch.full_like(c, 16), atol=0, rtol=0)


@pytest.mark.parametrize("column_b", [False, True])
@pytest.mark.parametrize("tiles", [1, 5, 16])
def test_bf16_automatic_schedule(monkeypatch, column_b, tiles):
    if not torch.cuda.is_available() or "gfx936" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("gfx936 required")
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", "1")
    monkeypatch.setenv("TRITON_HCU_ENABLE_DS_READ_M", "1")
    bk = 32 if column_b else 16
    k = tiles * bk
    torch.manual_seed(51)
    a = torch.randn((256, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((256, k) if column_b else (k, 256), device="cuda", dtype=torch.bfloat16)
    if column_b:
        b = b.T
    c = torch.empty((256, 256), device="cuda", dtype=torch.bfloat16)

    def run():
        return bf16_schedule_gemm[(1,)](
            a, b, c, k, a.stride(0), *b.stride(), BK=bk,
            num_warps=8, num_stages=2,
            use_async_copy=True, use_block_pingpong=True,
            block_pingpong_async_global_prefetch=2 if column_b else 4,
            block_pingpong_async_local_prefetch=1, matrix_instr_nonkdim=16,
            optimize_epilogue=False)

    h = run()
    torch.testing.assert_close(c, (a.float() @ b.float()).bfloat16(), atol=.02, rtol=.02)
    assert h.metadata.uses_bank_conflict_aware_lds_placement
    assert re.search(r"v_mmac\w*bf16", h.asm["amdgcn"])
    # Distinct first/wrap/final generations expose stale and lost fragments.
    for generation in sorted({0, min(4, tiles - 1), tiles - 1}):
        a.zero_()
        b.zero_()
        a[:, generation * bk:(generation + 1) * bk] = 1
        b[generation * bk:(generation + 1) * bk, :] = generation + 1
        run()
        torch.testing.assert_close(c, torch.full_like(c, bk * (generation + 1)), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bm,bn,bk", [(256, 256, 32), (128, 128, 64), (128, 256, 32),
                                    (256, 128, 64), (128, 256, 64)])
def test_default_pingpong_gpu(monkeypatch, dtype, bm, bn, bk):
    if not torch.cuda.is_available():
        pytest.skip("HCU device required")
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", "1")
    monkeypatch.setenv("TRITON_HIP_USE_ASYNC_COPY", "0")
    m = n = k = 512
    torch.manual_seed(42)
    a = torch.randn((m, k), device="cuda", dtype=dtype)
    b = torch.randn((n, k), device="cuda", dtype=dtype).T
    c = torch.empty((m, n), device="cuda", dtype=dtype)
    h = direct_gemm[(m // bm * (n // bn),)](
        a, b, c, m, n, k, BM=bm, BN=bn, BK=bk, B_COLUMN_MAJOR=True,
        num_warps=8, num_stages=2, use_block_pingpong=True,
        use_async_copy=False, matrix_instr_nonkdim=16)
    torch.testing.assert_close(c, a @ b, atol=0.02, rtol=0.02)
    assert "hcu.block_pingpong.applied" in h.asm["ttgir"]
    assert "hcu.block_pingpong.static_buffers" in h.asm["ttgir"]
    assert not h.metadata.use_async_copy
    assert h.metadata.block_pingpong_async_global_prefetch is None
    assert h.metadata.block_pingpong_async_local_prefetch == 1
