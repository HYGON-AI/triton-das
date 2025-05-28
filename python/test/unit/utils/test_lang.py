import pytest
import torch
import os
import sys
import triton
import triton.language as tl
from triton.utils import annotate_hint
from contextlib import nullcontext
    
# test annotate_hint
@pytest.mark.interpreter
@pytest.mark.parametrize("attr_name", ["non-negative", "invalid-attrs"])
@pytest.mark.parametrize("attr_value", [True, False])
def test_annotate_hint(attr_name, attr_value):
    size = 1024
    device = 'cuda'
    
    @triton.jit
    def kernel(x_ptr,  # *Pointer* to first input vector.
                y_ptr,  # *Block Pointer* to second input vector.
                n_elements,  # Size of the vector.
                BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
                ANNOTATION_NAME: tl.constexpr,
                ANNOTATION_VALUE: tl.constexpr,
          ):
        pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.

        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        
        # Apply annotation hint
        x = annotate_hint(tl.load(x_ptr + offsets), ANNOTATION_NAME, ANNOTATION_VALUE)
        y = x + 1
        annotate_hint(y, ANNOTATION_NAME) 
        tl.store(y_ptr + offsets, y)
    
    x = torch.rand(size, device=device)
    y = torch.empty_like(x)
    n_elements = y.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    try:
        with pytest.raises(Exception) if attr_name == "invalid-attrs" else nullcontext():
            # Launch kernel
            pgm = kernel[grid](x, y, n_elements, BLOCK_SIZE=1024, ANNOTATION_NAME=attr_name, ANNOTATION_VALUE=attr_value)

        if attr_name != "invalid-attrs":
            ttir = pgm.asm['ttir']
            count = 0
            expected_values = [attr_value, True]
            
            for line in ttir.splitlines():
                if "non-negative" in line:
                    count += 1
                    expected = str(expected_values[count - 1]).lower()
                    if f'"non-negative" = {expected}' not in line:
                        assert False, f"{attr_name=}, {attr_value=}: Expected 'non-negative = {expected}' not found in line:\n{line}"

            assert count >= len(expected_values), "{attr_name=}, {attr_value=}: Not enough 'non-negative' attributes found in TTIR"
            print(f"{attr_name=}, {attr_value=}: The actual output match the expectation, test passed.")
        else:
            print(f"{attr_name=}, {attr_value=}: Expected exception caught, test passed.")
    
    except Exception as e:
        if not (attr_name == "invalid-attrs"):
            raise
        else:
            print(f"Expected exception occurred and test passed: {e}")