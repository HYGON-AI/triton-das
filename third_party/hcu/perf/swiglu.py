#!/usr/bin/env python
# coding=utf-8

import pytest
import argparse
import sys

import functools

import torch
import triton
import triton.language as tl

from utils import ensure_contiguous

_swiglu_configs = [
    triton.Config({'BLOCK_SIZE': BLOCK_SIZE}, num_warps=w) \
    for BLOCK_SIZE in [4096, 8192, 16384, 32768]\
    for w in [4, 8]\
]


@triton.autotune(configs=_swiglu_configs, key=['N'])
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
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if NEED_MASK:
        mask = offs < N
        a_ = tl.load(a_ptr + offs, mask=mask, other=0)
    else:
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
    return


@triton.autotune(configs=_swiglu_configs, key=['N'])
@triton.jit
def _swiglu_backward(dc_ptr, a_ptr, b_ptr, da_ptr, db_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr,
                     NEED_MASK: tl.constexpr):
    """
    Computes the backward pass of the SwigLU activation function.
    Args:
        dc_ptr: Pointer to the gradient of the input tensor.
        a_ptr: Pointer to the input tensor.
        b_ptr: Pointer to the weight tensor.
        da_ptr: Pointer to the gradient of the a-input tensor.
        db_ptr: Pointer to the gradient of the b-input tensor.
        N: Number of elements in the input tensor.
        BLOCK_SIZE: Block size for Triton kernel.
        NEED_MASK: Whether to apply a mask or not.
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if NEED_MASK:
        mask = offs < N
        a_ = tl.load(a_ptr + offs, mask=mask, other=0)
    else:
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
        tl.store(da_ptr + offs, da_out, mask=mask)
        tl.store(db_ptr + offs, db_out, mask=mask)
    else:
        tl.store(da_ptr + offs, da_out)
        tl.store(db_ptr + offs, db_out)
    return


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
    N = a.numel()
    c = torch.empty_like(a)

    grid = lambda args: (N // args['BLOCK_SIZE'], )
    _swiglu_forward[grid](
        a,
        b,
        c,
        N=N,
        NEED_MASK=False,
    )

    best_config = _swiglu_forward.best_config
    block_size = int(best_config.__str__().split(",")[0].split("BLOCK_SIZE:")[1])

    N_REM = N % block_size
    if N_REM:
        grid = lambda args: (triton.cdiv(N_REM, args['BLOCK_SIZE']), )
        _swiglu_forward[grid](
            a.flatten()[N // block_size * block_size:],
            b.flatten()[N // block_size * block_size:],
            c.flatten()[N // block_size * block_size:],
            N=N_REM,
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
    N = a.numel()

    da = torch.empty_like(a)
    db = torch.empty_like(b)
    grid = lambda args: (N // args['BLOCK_SIZE'], )
    _swiglu_backward[grid](
        dc,
        a,
        b,
        da,
        db,
        N=N,
        NEED_MASK=False,
    )

    best_config = _swiglu_backward.best_config
    block_size = int(best_config.__str__().split(",")[0].split("BLOCK_SIZE:")[1])

    N_REM = N % block_size
    if N_REM:
        grid = lambda args: (triton.cdiv(N_REM, args['BLOCK_SIZE']), )
        _swiglu_backward[grid](
            dc.flatten()[N // block_size * block_size:],
            a.flatten()[N // block_size * block_size:],
            b.flatten()[N // block_size * block_size:],
            da.flatten()[N // block_size * block_size:],
            db.flatten()[N // block_size * block_size:],
            N=N_REM,
            NEED_MASK=True,
        )

    return da.view(*ori_shape), db.view(*ori_shape)


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
        (2600, 36000),
        # weird shapes
        (366, 5631),
        (366, 6631),
        (363, 6631),
        (2601, 36001),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        # atol is for small values: they have more difference, so set atol higher
        # rtol is for larger values: they are very close, so set rtol lower
        (torch.float32, 1e-3, 2e-6),
        (torch.float16, 1e-0, 2e-6),
        (torch.bfloat16, 1e-0, 2e-6),
    ],
)
def test_correctness(m, n, dtype, atol, rtol, use_Te, device="cuda"):
    a = torch.randn(m, n, device=device, dtype=dtype)
    b = torch.randn(m, n, device=device, dtype=dtype)
    dc = torch.randn(m, n, device=device, dtype=dtype)

    golden_forward = swiglu_golden_forward(a, b)
    golden_da, golden_db = swiglu_golden_backward(a, b, dc)

    if use_Te:
        print("Using transformer engine for testing")
        from transformer_engine.pytorch.ops.basic.activation import tex

        # input.shape: (m * n, 2)
        input = torch.stack((a.flatten(), b.flatten()), dim=1)
        # tex_forward.shape: (m * n, 1)
        tex_forward = tex.swiglu(input, quantizer=None)

        # dc.shape: (m * n, 1)
        tex_dc = dc.reshape(m * n, 1)
        tex_backward = tex.dswiglu(tex_dc, input, quantizer=None)
        tex_da, tex_db = torch.unbind(tex_backward, dim=1)

        torch.testing.assert_close(golden_forward, tex_forward.reshape(m, n), atol=atol, rtol=rtol)
        torch.testing.assert_close(golden_da, tex_da.reshape(m, n), atol=atol, rtol=rtol)
        torch.testing.assert_close(golden_db, tex_db.reshape(m, n), atol=atol, rtol=rtol)

    a0 = a.clone().requires_grad_(True)
    b0 = b.clone().requires_grad_(True)
    y1 = SiLUMulFunction.apply(a0, b0)
    y1.backward(dc)

    torch.testing.assert_close(golden_forward, y1, atol=atol, rtol=rtol)
    torch.testing.assert_close(golden_da, a0.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(golden_db, b0.grad, atol=atol, rtol=rtol)

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
            mn_args['dtype'] = arg_to_torch_dtype[dataType]

            if args.Te:
                print("Using transformer engine for benchmark")
                config.append(
                    triton.testing.Benchmark(
                        x_names=x_names,
                        x_vals=x_vals_list,
                        line_arg='provider',
                        line_vals=['transformer_engine'],
                        line_names=['Transformer_engine'],
                        styles=[('red', '-')],
                        ylabel="ms",
                        xlabel=f'N Size_{dataType}_{mode}',
                        plot_name=plot_name,
                        args=mn_args,
                    ))
            else:
                config.append(
                    triton.testing.Benchmark(
                        x_names=x_names,
                        x_vals=x_vals_list,
                        line_arg='provider',
                        line_vals=['triton', 'torch'],
                        line_names=['Triton', 'Torch'],
                        styles=[('red', '-'), ('blue', '-')],
                        ylabel="ms",
                        xlabel=f'N Size_{dataType}_{mode}',
                        plot_name=plot_name,
                        args=mn_args,
                    ))

    @triton.testing.perf_report(config)
    def benchmark(M, N, provider, mode, dtype, device='cuda'):
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

        if provider == 'torch':
            if mode == 'fwd':
                fn = lambda: swiglu_golden_forward(a, b)
            else:
                fn = lambda: swiglu_golden_backward(a, b, dc)

        if provider == 'transformer_engine':
            from transformer_engine.pytorch.ops.basic.activation import tex

            # input.shape: (M * N, 2)
            tex_input = torch.stack((a.flatten(), b.flatten()), dim=1)
            if mode == 'fwd':
                fn = lambda: tex.geglu(tex_input, quantizer=None)
            else:
                tex_dc = dc.reshape(M * N, 1)
                fn = lambda: tex.dgeglu(tex_dc, tex_input, quantizer=None)

        ms = triton.testing.do_bench(fn)
        return ms

    benchmark.run(save_path=".", show_plots=True, print_data=True)
    return 0


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
    parser.add_argument('--Te', action='store_true', help='Use transformer engine for benchmark')
    parser.add_argument('-d', "--dtype", default="fp16,bf16,fp32")
    parser.add_argument("-mode", default='fwd,bwd', help="Pass mode: fwd, bwd or both(default).")
    return parser.parse_args()


def main():
    args = parse_args()
    run_benchmark(args)
    # test_correctness(600, 36000, torch.float16, atol=1e-2, rtol=1e-2, use_Te=args.Te)
    return 0


if __name__ == "__main__":
    sys.exit(main())
