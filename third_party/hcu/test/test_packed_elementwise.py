"""Packed elementwise ALU: opcodes, rounding, tails and strided loads."""
import re

import pytest
import torch
import triton
import triton.language as tl

from triton._internal_testing import is_hip


PACKED_ARCHS = {
    'f16': {'gfx928', 'gfx936', 'gfx938', 'gfx92a', 'gfx946'},
    'bf16': {'gfx938', 'gfx946'},
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
    elif OP == 'sub':
        d = a - b
    elif OP == 'mul':
        d = a * b
    elif OP == 'min':
        d = tl.minimum(a, b)
    elif OP == 'max':
        d = tl.maximum(a, b)
    else:
        d = tl.fma(a, b, c)
    tl.store(D + i, d, i < n)


def supports_hcu_packed(dtype_name):
    return (is_hip() and triton.runtime.driver.active.get_current_target().arch
            in PACKED_ARCHS[dtype_name])


def ieee_minmax(op, a, b):
    # tl.minimum/tl.maximum default to the NaN-safe fminnum/fmaxnum ops. The
    # HCU min/max instructions order zeros as -0 < +0 (a zero tie resolves
    # to -0 for min unless both are +0, and to +0 for max unless both are
    # -0), so replicate that instead of relying on torch.fmin/fmax ties.
    an, bn = torch.isnan(a), torch.isnan(b)
    tie_zero = (a == 0) & (b == 0)
    has_neg = torch.signbit(a) | torch.signbit(b)
    has_pos = ~torch.signbit(a) | ~torch.signbit(b)
    if op == 'min':
        r = torch.where(a <= b, a, b)
        r = torch.where(tie_zero & has_neg, torch.full_like(a, -0.0), r)
    else:
        r = torch.where(a >= b, a, b)
        r = torch.where(tie_zero & has_pos, torch.zeros_like(a), r)
    r = torch.where(an & ~bn, b, r)
    r = torch.where(bn & ~an, a, r)
    return r


@pytest.mark.parametrize('dtype_name,dtype,tiny,storage_dtype', [
    ('f16', torch.float16, 2**-24, torch.int16),
    ('bf16', torch.bfloat16, 2**-133, torch.int16),
    ('f32', torch.float32, 2**-149, torch.int32),
])
@pytest.mark.parametrize('op', ['add', 'sub', 'mul', 'min', 'max', 'fma'])
@pytest.mark.parametrize('n,stride', [(1, 1), (127, 1), (1025, 1), (1025, 3)])
def test_packed_elementwise(dtype_name, dtype, tiny, storage_dtype, op, n,
                            stride):
    if not supports_hcu_packed(dtype_name):
        pytest.skip(f'requires HCU packed {dtype_name} ALU')

    # tl.minimum/tl.maximum on bf16 compute in f32 (see
    # _promote_bfloat16_to_float32), so the stored dtype differs.
    out_dtype = (torch.float32
                 if op in ('min', 'max') and dtype is torch.bfloat16 else
                 dtype)
    out_storage = torch.int32 if out_dtype is torch.float32 else storage_dtype

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
    if op in ('min', 'max'):
        ref = ieee_minmax(op, a, b)
    else:
        ref = {'add': lambda: a + b, 'sub': lambda: a - b,
               'mul': lambda: a * b, 'fma': lambda: a * b + c}[op]()
    ref = ref.to(out_dtype)
    dev = [x.cuda() for x in inputs]
    out = torch.empty(n, dtype=out_dtype, device='cuda')
    kernel = packed_elementwise_kernel[(triton.cdiv(n, 1024),)](
        *dev, out, n, stride, op, 1024, enable_fp_fusion=False)
    actual = out.cpu()
    nan = torch.isnan(ref)
    assert torch.equal(torch.isnan(actual), nan)
    assert torch.equal(actual[~nan].view(out_storage),
                       ref[~nan].view(out_storage))
    asm = kernel.asm['amdgcn']

    def has(op_re):
        return re.search(op_re, asm) is not None

    # Subtraction has no packed f16/bf16 opcode: the backend sign-flips the
    # packed RHS (v_xor_b32) and selects the corresponding packed add.
    if op == 'sub' and dtype_name in ('f16', 'bf16'):
        assert has(r'\bv_pk_add_' + dtype_name + r'\b')
    elif op in ('min', 'max') and dtype_name == 'f16':
        assert has(r'\bv_pk_' + op + r'_f16\b')
    elif op in ('sub', 'min', 'max') and dtype_name == 'f32':
        # The current HCU backend only has v_pk_{add,mul,fma}_f32; the
        # packed pair splits to scalar v_{sub,min,max}_f32 (HCU opcodes
        # carry the _e32 suffix). Accept a future v_pk_*_f32 as well.
        assert has(r'\bv_pk_' + op + r'_f32\b') or has(
            r'\bv_' + op + r'_f32(_e32)?\b')
    elif op in ('min', 'max') and dtype_name == 'bf16':
        # tl.minimum/tl.maximum promote bf16 to f32 in the frontend (see
        # _promote_bfloat16_to_float32), so no bf16 min/max instruction is
        # emitted.
        pass
    elif dtype_name in ('f16', 'bf16'):
        assert has(r'\bv_pk_' + op + '_' + dtype_name + r'\b')
    elif op == 'fma':
        assert has(r'\bv_pk_fma_f32\b')
