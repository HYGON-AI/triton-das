#!/usr/bin/env python
# coding=utf-8

import pytest
import argparse
import sys

import functools

import torch
import triton
import triton.language as tl

from utils import get_glu_block, ensure_contiguous, calculate_settings


@triton.jit
def _swiglu_forward(a_ptr, b_ptr, c_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr, NEED_MASK: tl.constexpr):
    """
    Computes the SwigLU activation function.
    Args:
        a_ptr: Pointer to the input tensor.
        b_ptr: Pointer to the weight tensor.
        c_ptr: Pointer to the output tensor.
        N: Number of elements in the input tensor.
        BLOCK_SIZE: Block size for Triton kernel.
        NEED_MASK: Whether to apply a mask or not.
    """
    if NEED_MASK:
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        a_ = tl.load(a_ptr + offs, mask=mask, other=0)
    else:
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        a_ = tl.load(a_ptr + offs)

    a = a_.to(tl.float32)  # sigmoid requires type float32
    sig_a = tl.sigmoid(a)
    if NEED_MASK:
        b = tl.load(b_ptr + offs, mask=mask, other=0)
    else:
        b = tl.load(b_ptr + offs)

    c_out = a * sig_a * b
    if NEED_MASK:
        tl.store(c_ptr + offs, c_out, mask=mask)
    else:
        tl.store(c_ptr + offs, c_out)


@triton.jit
def _swiglu_backward(dc_ptr, a_ptr, b_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr, NEED_MASK: tl.constexpr):
    """
    Computes the backward pass of the SwigLU activation function.
    Args:
        dc_ptr: Pointer to the gradient of the output tensor.
        a_ptr: Pointer to the input tensor.
        b_ptr: Pointer to the weight tensor.
        N: Number of elements in the input tensor.
        BLOCK_SIZE: Block size for Triton kernel.
        NEED_MASK: Whether to apply a mask or not.
    """
    if NEED_MASK:
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        a_ = tl.load(a_ptr + offs, mask=mask, other=0)
    else:
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        a_ = tl.load(a_ptr + offs)

    a = a_.to(tl.float32)  # sigmoid requires type float32
    sig_a = tl.sigmoid(a)
    silu_a = a * sig_a
    if NEED_MASK:
        b = tl.load(b_ptr + offs, mask=mask, other=0)
        dc = tl.load(dc_ptr + offs, mask=mask, other=0)
    else:
        b = tl.load(b_ptr + offs)
        dc = tl.load(dc_ptr + offs)

    db_out = dc * silu_a
    da_out = dc * (silu_a * (1 - sig_a) + sig_a) * b
    if NEED_MASK:
        tl.store(a_ptr + offs, da_out, mask=mask)
        tl.store(b_ptr + offs, db_out, mask=mask)
    else:
        tl.store(a_ptr + offs, da_out)
        tl.store(b_ptr + offs, db_out)


@ensure_contiguous
def swiglu_forward(a, b):
    """
    Applies the SwigLU activation function.
    Args:
        a: Input tensor.
        b: Input tensor.
    Returns:
        Output tensor after applying SwigLU.
    """
    assert a.shape == b.shape, "a and b must have the same shape"
    ori_shape = a.shape

    n_elements = a.numel()
    BLOCK_SIZE, _ = get_glu_block(n_elements)
    align_m = n_elements // BLOCK_SIZE

    c = torch.empty_like(a)

    _swiglu_forward[(align_m, 1)](
        a,
        b,
        c,
        N=BLOCK_SIZE,
        BLOCK_SIZE=BLOCK_SIZE,
        NEED_MASK=False,
    )

    if n_elements % BLOCK_SIZE:
        _swiglu_forward[(1, 1)](
            a.flatten()[align_m * BLOCK_SIZE:],
            b.flatten()[align_m * BLOCK_SIZE:],
            c.flatten()[align_m * BLOCK_SIZE:],
            N=n_elements % BLOCK_SIZE,
            BLOCK_SIZE=triton.next_power_of_2(n_elements % BLOCK_SIZE),
            NEED_MASK=True,
        )

    return a, b, c.view(*ori_shape)


@ensure_contiguous
def swiglu_backward(a, b, dc):
    """
    Computes the gradient of the SwigLU activation function.
    Args:
        dc: Gradient of the output tensor.
        a: Input tensor.
        b: Input tensor.
    Returns:
        Gradient of the input tensor.
    """
    assert a.shape == b.shape == dc.shape, "a,b and dc must have the same shape"
    ori_shape = a.shape

    n_elements = a.numel()
    BLOCK_SIZE, _ = get_glu_block(n_elements)
    align_m = n_elements // BLOCK_SIZE

    _swiglu_backward[(align_m, 1)](
        dc,
        a,
        b,
        N=BLOCK_SIZE,
        BLOCK_SIZE=BLOCK_SIZE,
        NEED_MASK=False,
    )
    if n_elements % BLOCK_SIZE:
        _swiglu_backward[(1, 1)](
            dc.flatten()[align_m * BLOCK_SIZE:],
            a.flatten()[align_m * BLOCK_SIZE:],
            b.flatten()[align_m * BLOCK_SIZE:],
            N=n_elements % BLOCK_SIZE,
            BLOCK_SIZE=triton.next_power_of_2(n_elements % BLOCK_SIZE),
            NEED_MASK=True,
        )

    return a.view(*ori_shape), b.view(*ori_shape)


class SiLUMulFunction(torch.autograd.Function):

    @staticmethod
    @ensure_contiguous
    def forward(ctx, a, b):
        a, b, c = swiglu_forward(a, b)
        ctx.save_for_backward(a, b)
        return c

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dc):
        a, b = ctx.saved_tensors
        a, b = swiglu_backward(a, b, dc)
        return a, b


# test code
# copy from:https://github.com/linkedin/Liger-Kernel/blob/v0.5.5/src/liger_kernel/ops/swiglu.py#L16
@triton.jit
def silu(x):
    return x * tl.sigmoid(x)


@triton.jit
def _swiglu_forward_kernel(a_ptr, b_ptr, c_ptr, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    a_ptr += program_id * stride
    b_ptr += program_id * stride
    c_ptr += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # sigmoid requires type float32
    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
    c_row = silu(a_row) * b_row
    tl.store(c_ptr + col_offsets, c_row, mask=mask)


@triton.jit
def _swiglu_backward_kernel(dc_ptr, a_ptr, b_ptr, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    dc_ptr += program_id * stride
    a_ptr += program_id * stride
    b_ptr += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dc_row = tl.load(dc_ptr + col_offsets, mask=mask, other=0)
    # sigmoid requires type float32
    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)

    # recomputation to save memory
    sig_a = tl.sigmoid(a_row)
    silu_a = a_row * sig_a
    db_row = dc_row * silu_a
    da_row = dc_row * (silu_a * (1 - sig_a) + sig_a) * b_row

    tl.store(a_ptr + col_offsets, da_row, mask=mask)
    tl.store(b_ptr + col_offsets, db_row, mask=mask)


def swiglu_forward_Liger(a, b):
    ori_shape = a.shape

    n_cols = ori_shape[-1]
    a = a.view(-1, n_cols)
    b = b.view(-1, n_cols)
    c = torch.empty_like(a)
    n_rows = a.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _swiglu_forward_kernel[(n_rows, )](
        a,
        b,
        c,
        c.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return a, b, c.view(*ori_shape)


def swiglu_backward_Liger(a, b, dc):
    ori_shape = dc.shape
    n_cols = ori_shape[-1]
    dc = dc.view(-1, n_cols)
    n_rows = dc.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _swiglu_backward_kernel[(n_rows, )](
        dc,
        a,
        b,
        dc.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return a.view(*ori_shape), b.view(*ori_shape)


def swiglu_golden_forward(a, b):
    sig_a = torch.sigmoid(a)
    return a * sig_a * b


def swiglu_golden_backward(a, b, dc):
    sig_a = torch.sigmoid(a)
    da = b * (sig_a + a * sig_a * (1 - sig_a)) * dc
    db = a * sig_a * dc
    return da, db


@pytest.mark.parametrize(
    "m, n",
    [
        (2048, 4096),
        (256, 6000),
        (256, 6000),
        (512, 8192),
        (1024, 16384),
        (2048, 32768),
        # weird shapes
        (366, 5631),
        (366, 6631),
        (363, 6631),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        # atol is for small values: they have more difference, so set atol higher
        # rtol is for larger values: they are very close, so set rtol lower
        (torch.float32, 1e-0, 2e-6),
        (torch.float16, 1e-0, 2e-6),
        (torch.bfloat16, 1e-0, 2e-6),
    ],
)
def test_correctness(m, n, dtype, atol, rtol, device="cuda"):
    a = torch.randn(m, n, device=device, dtype=dtype)
    b = torch.randn(m, n, device=device, dtype=dtype)
    dc = torch.randn(m, n, device=device, dtype=dtype)

    golden_forward = swiglu_golden_forward(a, b)
    golden_da, golden_db = swiglu_golden_backward(a, b, dc)

    a0 = a.clone().requires_grad_(True)
    b0 = b.clone().requires_grad_(True)
    y1 = SiLUMulFunction.apply(a0, b0)
    y1.backward(dc)

    a1 = a.clone()
    b1 = b.clone()
    _, _, y2 = swiglu_forward_Liger(a1, b1)
    light_da, light_db = swiglu_backward_Liger(a1, b1, dc)

    torch.testing.assert_close(golden_forward, y1, atol=atol, rtol=rtol)
    torch.testing.assert_close(y2, y1, atol=atol, rtol=rtol)

    torch.testing.assert_close(golden_da, a0.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(light_da, a0.grad, atol=atol, rtol=rtol)

    torch.testing.assert_close(golden_db, b0.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(light_db, b0.grad, atol=atol, rtol=rtol)

    return


arg_to_torch_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}


#Benchmark
def run_benchmark(args):
    config = []

    for mode in args.mode.split(','):
        for dataType in args.dtype.split(','):
            if (args.M_benchmark):
                val = args.M_start
                x_vals_list = []
                while val <= args.M_end:
                    x_vals_list.append(val)
                    val += args.M_step
                mn_args = {'N': args.N_start}
                plot_name = str("swiglu-performance_" + dataType + "_N" + str(args.N_start) + "_M" + str(args.M_start) +
                                "-" + str(args.M_end) + "-" + str(args.M_step) + "_mode_" + mode)
                x_names = ['M']
            else:
                x_vals_list = [i for i in range(args.N_start, args.N_end, args.N_step)]
                mn_args = {'M': args.M_start}
                x_names = ['N']
                plot_name = str("swiglu-performance_" + dataType + "_M" + str(args.M_start) + "_N" + str(args.N_start) +
                                "-" + str(args.N_end) + "-" + str(args.N_step) + "_mode_" + mode)
            mn_args['mode'] = mode
            dtype = arg_to_torch_dtype[dataType]

            config.append(
                triton.testing.Benchmark(
                    x_names=x_names,
                    x_vals=x_vals_list,
                    line_arg='provider',
                    line_vals=['triton', 'triton_Liger', 'torch'],
                    line_names=['Triton', 'Triton_Liger', 'Torch'],
                    styles=[('red', '-'), ('blue', '-'), ('green', '-'), ('yellow', '-')],
                    ylabel="ms",
                    xlabel=f'N Size_{dataType}_{mode}',
                    plot_name=plot_name,
                    args=mn_args,
                ))

    @triton.testing.perf_report(config)
    def benchmark(M, N, provider, mode, device='cuda'):
        a = torch.randn(M, N, device=device, dtype=dtype)
        b = torch.randn(M, N, device=device, dtype=dtype)
        dc = torch.randn(M, N, device=device, dtype=dtype)

        stream = torch.cuda.Stream()
        torch.cuda.set_stream(stream)
        if provider == 'triton':
            if mode == 'fwd':
                fn = lambda: swiglu_forward(a, b)
            else:
                fn = lambda: swiglu_backward(a, b, dc)

        if provider == 'triton_Liger':
            if mode == 'fwd':
                fn = lambda: swiglu_forward_Liger(a, b)
            else:
                fn = lambda: swiglu_backward_Liger(a, b, dc)

        if provider == 'torch':
            if mode == 'fwd':
                fn = lambda: swiglu_golden_forward(a, b)
            else:
                fn = lambda: swiglu_golden_backward(a, b, dc)

        ms = triton.testing.do_bench(fn)
        return ms

    benchmark.run(save_path=".", show_plots=True, print_data=True)
    return


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark SwiGLU",
        allow_abbrev=False,
    )
    parser.add_argument('-M', "--M_start", default="600", type=int)
    parser.add_argument('-Ms', "--M_step", default="100", type=int)  #This is multiplicative step
    parser.add_argument('-Me', "--M_end", default="2600", type=int)

    parser.add_argument('-N', "--N_start", default="8000", type=int)
    parser.add_argument('-Ns', "--N_step", default="1000", type=int)
    parser.add_argument('-Ne', "--N_end", default="37000", type=int)

    parser.add_argument('-Mb', "--M_benchmark", action='store_true')
    parser.add_argument('-d', "--dtype", default="fp16,bf16,fp32")
    parser.add_argument("-mode", default='fwd,bwd', help="Pass mode: fwd, bwd or both(default).")
    return parser.parse_args()


def main():
    args = parse_args()
    run_benchmark(args)
    # test_correctness(256, 6000, torch.float32, 1e-0, 2e-6, "cuda")
    return


if __name__ == "__main__":
    sys.exit(main())
