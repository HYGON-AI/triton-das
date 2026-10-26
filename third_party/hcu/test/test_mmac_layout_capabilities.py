# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Instruction-level layout selection with a runtime masked reduction bound."""
import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _fp32_k4(A, B, C, K):
    m = tl.arange(0, 64)
    n = tl.arange(0, 64)
    k = tl.arange(0, 4)
    a = tl.load(A + m[:, None] * 4 + k[None, :], k[None, :] < K, 0)
    b = tl.load(B + n[None, :] * 4 + k[:, None], k[:, None] < K, 0)
    c = tl.dot(a, b, input_precision="ieee")
    tl.store(C + m[:, None] * 64 + n[None, :], c)


def test_fp32_k4_layout_matches_instruction():
    if not torch.cuda.is_available():
        pytest.skip("requires an HCU device")
    generator = torch.Generator().manual_seed(506)
    a = torch.randn((64, 4), generator=generator, dtype=torch.float32)
    b = torch.randn((64, 4), generator=generator, dtype=torch.float32)
    x, y = a.cuda(), b.cuda()
    out = torch.empty((64, 64), device="cuda", dtype=torch.float32)
    hashes = set()
    for k in [2, 4]:
        compiled = _fp32_k4[(1,)](
            x, y, out, k, num_warps=4, num_stages=1,
            matrix_instr_nonkdim=16, enable_fp_fusion=False)
        torch.testing.assert_close(out.cpu(), a[:, :k] @ b[:, :k].T,
                                   atol=1e-5, rtol=1e-5)
        assert "instrShape = [16, 16, 4]" in compiled.asm["ttgir"]
        assert "v_mmac" in compiled.asm["amdgcn"]
        if compiled.metadata.target.arch == "gfx938":
            assert "mmacLayout = 3" in compiled.asm["ttgir"]
        hashes.add(compiled.hash)
    assert len(hashes) == 1  # Both runtime K values reuse the same binary.
