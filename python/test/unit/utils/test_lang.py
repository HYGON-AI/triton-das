# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

import pytest
import torch
from contextlib import nullcontext

import triton
import triton.language as tl
from triton.utils import hint


@pytest.mark.interpreter
@pytest.mark.parametrize("attr_name", ["val.no_negative", "bad.key"])
@pytest.mark.parametrize("attr_value", [True, False])
def test_hint_val(attr_name, attr_value):
    size = 1024

    @triton.jit
    def kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr, KEY: tl.constexpr,
               VAL: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = hint(KEY, tl.load(x_ptr + offsets), VAL)
        y = x + 1
        hint(KEY, y)
        tl.store(y_ptr + offsets, y)

    x = torch.rand(size, device="cuda")
    y = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(y.numel(), meta["BLOCK_SIZE"]), )

    with pytest.raises(Exception) if attr_name == "bad.key" else nullcontext():
        pgm = kernel[grid](x, y, y.numel(), BLOCK_SIZE=1024, KEY=attr_name, VAL=attr_value)

    if attr_name == "bad.key":
        return

    ttir = pgm.asm["ttir"]
    count = 0
    expected_values = [attr_value, True]
    for line in ttir.splitlines():
        if "tt.hint.val.no_negative" in line:
            count += 1
            expected = str(expected_values[count - 1]).lower()
            assert f"tt.hint.val.no_negative = {expected}" in line, line
    assert count >= len(expected_values)
