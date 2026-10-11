# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Scale-buffer copies must finish before scaled MMAC consumes their data."""
import re

import pytest
import triton
import triton.language as tl
from triton.compiler import ASTSource


@triton.jit
def scaled_copy_kernel(A, B, SA, SB, C, MLS: tl.constexpr):
    m = tl.arange(0, 64)
    n = tl.arange(0, 64)
    k = tl.arange(0, 128)
    s = tl.arange(0, 8)
    if MLS:
        a = tl.matrix_load(A, shape=[64, 128], strides=[128, 1],
                           block_shape=[64, 128], offsets=[0, 0])
        b = tl.matrix_load(B, shape=[128, 64], strides=[1, 128],
                           block_shape=[128, 64], offsets=[0, 0])
    else:
        a = tl.load(A + m[:, None] * 128 + k[None, :])
        b = tl.load(B + k[:, None] + n[None, :] * 128)
    sa = tl.load(SA + m[:, None] * 8 + s[None, :])
    sb = tl.load(SB + n[:, None] * 8 + s[None, :])
    c = tl.dot_scaled(a, sa, "e2m1", b, sb, "e2m1")
    tl.store(C + m[:, None] * 64 + n[None, :], c)


@pytest.mark.parametrize('mls', [False, True])
def test_scaled_copy_wait_before_mmac(mls):
    target = triton.runtime.driver.active.get_current_target()
    if target is None or target.backend != 'hip' or target.arch != 'gfx946':
        pytest.skip('scaled MMAC synchronization checks require gfx946')
    source = ASTSource(scaled_copy_kernel,
                       signature={'A': '*u8', 'B': '*u8', 'SA': '*u8', 'SB': '*u8', 'C': '*fp32'},
                       constexprs={'MLS': mls})
    compiled = triton.compile(source, target=target,
                              options={'num_warps': 4, 'num_stages': 1, 'mmac_layout_force': 2})
    assembly = compiled.asm['amdgcn']
    first_mmac = assembly.index('v_mmac_scale_')
    last_copy = assembly.rindex('ds_scale_copy_ds2buf', 0, first_mmac)
    between = assembly[last_copy:first_mmac]
    wait = re.search(r's_waitcnt\b[^\n]*\blgkmcnt\(0\)', between)
    assert wait, 'Missing scale-copy completion wait before scaled MMAC:\n' + between
    assert 's_barrier' in between[wait.end():], 'Missing CTA barrier after scale-copy wait'
