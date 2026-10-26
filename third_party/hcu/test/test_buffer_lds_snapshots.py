# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Retiming must reject intervening source writes and retain safe native copies."""
import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def snapshot_before_overwrite(a, b, c, K, BM: tl.constexpr, BN: tl.constexpr,
                              BK: tl.constexpr, OVERWRITE: tl.constexpr = True):
    m = tl.arange(0, BM)
    n = tl.arange(0, BN)
    k = tl.arange(0, BK)
    ap = a + m[:, None] * K + k[None, :]
    bp = b + n[None, :] * K + k[:, None]
    av = tl.load(ap)
    bv = tl.load(bp)
    # Changing the source after its read must not change the dot operands.
    if OVERWRITE:
        tl.store(ap, tl.full((BM, BK), 0, tl.float32))
    result = tl.dot(av, bv)
    tl.store(c + m[:, None] * BN + n[None, :], result)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_buffer_lds_preserves_read_before_write(monkeypatch, dtype):
    if not torch.cuda.is_available():
        pytest.skip('HCU device required')
    monkeypatch.setenv('AMDGCN_USE_BUFFER_OPS', '1')
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', False)
    a = torch.randn((64, 32), device='cuda', dtype=dtype)
    b = torch.randn((64, 32), device='cuda', dtype=dtype)
    c = torch.empty((64, 64), device='cuda', dtype=dtype)
    expected = a.float() @ b.float().T
    compiled = snapshot_before_overwrite[(1,)](
        a, b, c, 32, BM=64, BN=64, BK=32, num_warps=8, num_stages=1,
        use_async_copy=True, use_block_pingpong=False)
    torch.testing.assert_close(c.float(), expected, atol=.125 if dtype == torch.bfloat16 else .02, rtol=.02)
    assert torch.count_nonzero(a).item() == 0
    assert not compiled.metadata.uses_bank_conflict_aware_lds_placement


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_native_copy_is_retained_without_source_write(monkeypatch, dtype):
    if not torch.cuda.is_available():
        pytest.skip('HCU device required')
    monkeypatch.setenv('AMDGCN_USE_BUFFER_OPS', '1')
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', False)
    a = torch.randn((64, 32), device='cuda', dtype=dtype)
    b = torch.randn((64, 32), device='cuda', dtype=dtype)
    c = torch.empty((64, 64), device='cuda', dtype=dtype)
    expected = a.float() @ b.float().T
    compiled = snapshot_before_overwrite[(1,)](
        a, b, c, 32, BM=64, BN=64, BK=32, OVERWRITE=False,
        num_warps=8, num_stages=1, use_async_copy=True, use_block_pingpong=False)
    torch.testing.assert_close(c.float(), expected,
                               atol=.125 if dtype == torch.bfloat16 else .02, rtol=.02)
    assert 'amdg.buffer_load_to_local' in compiled.asm['ttgir']
    assert 'llvm.amdgcn.raw.ptr.buffer.load.async.lds' in compiled.asm['llir']
