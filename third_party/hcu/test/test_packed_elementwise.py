"""Packed elementwise ALU: opcodes, rounding, tails and strided loads."""
import re

import pytest
import torch
import triton
import triton.language as tl

from triton._internal_testing import is_hip


PACKED_ARCHS = {
    'f16': {'gfx928', 'gfx936', 'gfx938', 'gfx92a', 'gfx946'},
    'bf16': {'gfx938', 'gfx92a', 'gfx946'},
    'f32': {'gfx936', 'gfx938', 'gfx92a', 'gfx946'},
}


@triton.jit
def packed_elementwise_kernel(A, B, C, D, n, stride: tl.constexpr,
                              OP: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + i * stride, i < n, 0)
    b = tl.load(B + i * stride, i < n, 0)
    c = tl.load(C + i * stride, i < n, 0)
    if OP == 'add':
        d = a + b
    elif OP == 'mul':
        d = a * b
    else:
        d = tl.fma(a, b, c)
    tl.store(D + i, d, i < n)


def supports_hcu_packed(dtype_name):
    return (is_hip() and triton.runtime.driver.active.get_current_target().arch
            in PACKED_ARCHS[dtype_name])


@pytest.mark.parametrize('dtype_name,dtype,tiny,storage_dtype', [
    ('f16', torch.float16, 2**-24, torch.int16),
    ('bf16', torch.bfloat16, 2**-133, torch.int16),
    ('f32', torch.float32, 2**-149, torch.int32),
])
@pytest.mark.parametrize('op', ['add', 'mul', 'fma'])
@pytest.mark.parametrize('n,stride', [(1, 1), (127, 1), (1025, 1), (1025, 3)])
def test_packed_elementwise(dtype_name, dtype, tiny, storage_dtype, op, n,
                            stride):
    if not supports_hcu_packed(dtype_name):
        pytest.skip(f'requires HCU packed {dtype_name} ALU')

    # Include signed zero, subnormals, overflow, infinities and NaNs. Different
    # neighboring lanes catch swaps; the double reference rounds once per op.
    torch.manual_seed(17)
    inputs = [torch.randn(n * stride, dtype=dtype) for _ in range(3)]
    edge = torch.tensor([0., -0., tiny, -tiny, 65504., float('inf'),
                         -float('inf'), float('nan')], dtype=dtype)
    for j, x in enumerate(inputs):
        if n >= edge.numel():
            x[0:edge.numel() * stride:stride] = edge.roll(j)
    a, b, c = [x[::stride].double() for x in inputs]
    ref = {'add': lambda: a + b, 'mul': lambda: a * b,
           'fma': lambda: a * b + c}[op]().to(dtype)
    dev = [x.cuda() for x in inputs]
    out = torch.empty(n, dtype=dtype, device='cuda')
    kernel = packed_elementwise_kernel[(triton.cdiv(n, 1024),)](
        *dev, out, n, stride, op, 1024, enable_fp_fusion=False)
    actual = out.cpu()
    nan = torch.isnan(ref)
    assert torch.equal(torch.isnan(actual), nan)
    assert torch.equal(actual[~nan].view(storage_dtype),
                       ref[~nan].view(storage_dtype))
    # F32 add/mul instruction selection is covered by the MLIR conversion
    # test. This runtime shape reliably produces packed F32 FMA only.
    if dtype_name != 'f32' or op == 'fma':
        assert re.search(r'\bv_pk_' + op + '_' + dtype_name + r'\b',
                         kernel.asm['amdgcn'])
