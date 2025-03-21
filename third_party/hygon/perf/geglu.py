#!/usr/bin/env python
# coding=utf-8

import pytest
import argparse
import sys

import operator
import functools
import importlib

import torch
import triton
import triton.language as tl

from typing import Callable
from packaging.version import Version


def compare_version(package: str, operator: Callable, target: str):
    try:
        pkg = importlib.import_module(package)
    except ImportError:
        return False
    pkg_version = Version(pkg.__version__)
    return operator(pkg_version, Version(target))


if compare_version("triton", operator.ge, "3.0.0"):
    from triton.language.extra.libdevice import tanh
else:
    from triton.language.math import tanh


def ensure_contiguous(fn):

    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):

        def maybe_to_contiguous(x):
            return x.contiguous() if isinstance(x, torch.Tensor) else x

        args = [maybe_to_contiguous(arg) for arg in args]
        kwargs = {k: maybe_to_contiguous(v) for k, v in kwargs.items()}
        return fn(ctx, *args, **kwargs)

    return wrapper


def get_block_size(n):
    if n >= 65536:
        return 4096

    tmp = triton.next_power_of_2(n)
    return tmp >> 4


@triton.jit
def _geglu_forward(a_ptr, b_ptr, c_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr, NEED_MASK: tl.constexpr):

    if NEED_MASK:
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0)
    else:
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        a = tl.load(a_ptr + offs)

    # tanh approximation form of GELU is computed with:
    # 0.5 * a * (1 + tanh(sqrt(2 / pi) * (a + 0.044715 * a^3)))
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2 / pi)
    a_cubed = a * a * a
    tanh_arg = sqrt_2_over_pi * (a + 0.044715 * a_cubed)
    tanh_result = tanh(tanh_arg)
    geglu_a = 0.5 * a * (1 + tanh_result)
    if NEED_MASK:
        b = tl.load(b_ptr + offs, mask=mask, other=0)
    else:
        b = tl.load(b_ptr + offs)

    c_out = geglu_a * b
    if NEED_MASK:
        tl.store(c_ptr + offs, c_out, mask=mask)
    else:
        tl.store(c_ptr + offs, c_out)


@triton.jit
def _geglu_backward(dc_ptr, a_ptr, b_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr, NEED_MASK: tl.constexpr):

    if NEED_MASK:
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0)
    else:
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        a = tl.load(a_ptr + offs)

    # recomputation to save memory
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2 / pi)
    a_cubed = a * a * a
    tanh_arg = sqrt_2_over_pi * (a + 0.044715 * a_cubed)
    tanh_result = tanh(tanh_arg)
    geglu_a = 0.5 * a * (1 + tanh_result)

    # Gradient w.r.t. a can be computed with:
    # b * (0.5 * (1 + tanh(z)) + 0.5 * a * (1 - tanh(z)^2) * (sqrt(2/pi) * (1 + 3 * 0.044715 * a^2)))
    # where z = sqrt(2/pi) * (a + 0.044715 * a^3)
    term1 = 0.5 * (1 + tanh_result)
    tanh_sq = tanh_result * tanh_result
    term2 = 0.5 * a * (1 - tanh_sq) * (sqrt_2_over_pi * (1 + 3 * 0.044715 * a * a))

    if NEED_MASK:
        b = tl.load(b_ptr + offs, mask=mask, other=0)
        dc = tl.load(dc_ptr + offs, mask=mask, other=0)
    else:
        b = tl.load(b_ptr + offs)
        dc = tl.load(dc_ptr + offs)

    db_out = dc * geglu_a
    da_out = dc * b * (term1 + term2)

    if NEED_MASK:
        tl.store(a_ptr + offs, da_out, mask=mask)
        tl.store(b_ptr + offs, db_out, mask=mask)
    else:
        tl.store(a_ptr + offs, da_out)
        tl.store(b_ptr + offs, db_out)


@ensure_contiguous
def geglu_forward(a, b):

    assert a.shape == b.shape, "a and b must have the same shape"
    ori_shape = a.shape

    n_elements = a.numel()
    BLOCK_SIZE = get_block_size(n_elements)
    align_m = n_elements // BLOCK_SIZE

    c = torch.empty_like(a)

    _geglu_forward[(align_m, 1)](
        a,
        b,
        c,
        N=BLOCK_SIZE,
        BLOCK_SIZE=BLOCK_SIZE,
        NEED_MASK=False,
    )

    if n_elements % BLOCK_SIZE:
        _geglu_forward[(1, 1)](
            a.flatten()[align_m * BLOCK_SIZE:],
            b.flatten()[align_m * BLOCK_SIZE:],
            c.flatten()[align_m * BLOCK_SIZE:],
            N=n_elements % BLOCK_SIZE,
            BLOCK_SIZE=triton.next_power_of_2(n_elements % BLOCK_SIZE),
            NEED_MASK=True,
        )

    return a, b, c.view(*ori_shape)


@ensure_contiguous
def geglu_backward(a, b, dc):
    assert a.shape == b.shape == dc.shape, "a,b and dc must have the same shape"
    ori_shape = a.shape

    n_elements = a.numel()
    BLOCK_SIZE = get_block_size(n_elements)
    align_m = n_elements // BLOCK_SIZE

    _geglu_backward[(align_m, 1)](
        dc,
        a,
        b,
        N=BLOCK_SIZE,
        BLOCK_SIZE=BLOCK_SIZE,
        NEED_MASK=False,
    )
    if n_elements % BLOCK_SIZE:
        _geglu_backward[(1, 1)](
            dc.flatten()[align_m * BLOCK_SIZE:],
            a.flatten()[align_m * BLOCK_SIZE:],
            b.flatten()[align_m * BLOCK_SIZE:],
            N=n_elements % BLOCK_SIZE,
            BLOCK_SIZE=triton.next_power_of_2(n_elements % BLOCK_SIZE),
            NEED_MASK=True,
        )

    return a.view(*ori_shape), b.view(*ori_shape)


class GELUMulFunction(torch.autograd.Function):

    @staticmethod
    @ensure_contiguous
    def forward(ctx, a, b):
        a, b, c = geglu_forward(a, b)
        ctx.save_for_backward(a, b)
        return c

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dc):
        a, b = ctx.saved_tensors
        a, b = geglu_backward(a, b, dc)
        return a, b


# test code
def is_hip() -> bool:
    return torch.version.hip is not None


def calculate_settings(n):
    # reference: https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/utils.py#L43

    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds the recommended Triton blocksize = {MAX_FUSED_SIZE}.")

    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32 if not is_hip() else 16
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps


# copy from:https://github.com/linkedin/Liger-Kernel/blob/v0.5.5/src/liger_kernel/ops/geglu.py#L26
@triton.jit
def _geglu_tanh_forward_kernel(a, b, c, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    a += program_id * stride
    b += program_id * stride
    c += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    a_row = tl.load(a + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b + col_offsets, mask=mask, other=0)

    # tanh approximation form of GELU is computed with:
    # 0.5 * a * (1 + tanh(sqrt(2 / pi) * (a + 0.044715 * a^3)))
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2 / pi)
    a_cubed = a_row * a_row * a_row
    tanh_arg = sqrt_2_over_pi * (a_row + 0.044715 * a_cubed)
    tanh_result = tanh(tanh_arg)
    geglu_a = 0.5 * a_row * (1 + tanh_result)
    c_row = geglu_a * b_row
    tl.store(c + col_offsets, c_row, mask=mask)


@triton.jit
def _geglu_tanh_backward_kernel(dc, a, b, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    dc += program_id * stride
    a += program_id * stride
    b += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dc_row = tl.load(dc + col_offsets, mask=mask, other=0)
    a_row = tl.load(a + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b + col_offsets, mask=mask, other=0)

    # recomputation to save memory
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2 / pi)
    a_cubed = a_row * a_row * a_row
    tanh_arg = sqrt_2_over_pi * (a_row + 0.044715 * a_cubed)
    tanh_result = tanh(tanh_arg)
    geglu_a = 0.5 * a_row * (1 + tanh_result)

    db_row = dc_row * geglu_a

    # Gradient w.r.t. a can be computed with:
    # b * (0.5 * (1 + tanh(z)) + 0.5 * a * (1 - tanh(z)^2) * (sqrt(2/pi) * (1 + 3 * 0.044715 * a^2)))
    # where z = sqrt(2/pi) * (a + 0.044715 * a^3)
    term1 = 0.5 * (1 + tanh_result)
    tanh_sq = tanh_result * tanh_result
    term2 = 0.5 * a_row * (1 - tanh_sq) * (sqrt_2_over_pi * (1 + 3 * 0.044715 * a_row * a_row))
    da_row = dc_row * b_row * (term1 + term2)

    tl.store(a + col_offsets, da_row, mask=mask)
    tl.store(b + col_offsets, db_row, mask=mask)


def geglu_forward_Liger(a, b):
    ori_shape = a.shape

    n_cols = ori_shape[-1]
    a = a.view(-1, n_cols)
    b = b.view(-1, n_cols)
    c = torch.empty_like(a)
    n_rows = a.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _geglu_tanh_forward_kernel[(n_rows, )](
        a,
        b,
        c,
        c.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return a, b, c.view(*ori_shape)


def geglu_backward_Liger(a, b, dc):
    ori_shape = dc.shape
    n_cols = ori_shape[-1]
    dc = dc.view(-1, n_cols)
    n_rows = dc.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _geglu_tanh_backward_kernel[(n_rows, )](
        dc,
        a,
        b,
        dc.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return a.view(*ori_shape), b.view(*ori_shape)


class LigerGELUMulFunction(torch.autograd.Function):

    @staticmethod
    @ensure_contiguous
    def forward(ctx, a, b):
        a, b, c = geglu_forward(a, b)
        ctx.save_for_backward(a, b)
        return c

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dc):
        a, b = ctx.saved_tensors
        a, b = geglu_backward(a, b, dc)
        return a, b


# copy from: https://github.com/huggingface/transformers/blob/v4.50.3/src/transformers/activations.py#L126
def geglu_golden_forward(a, b):

    def gelu_accurate(x):
        return 0.5 * x * (1 + torch.tanh(0.7978845608028654 * (x + 0.044715 * torch.pow(x, 3))))

    return gelu_accurate(a) * b


def geglu_golden_backward(a, b, dc):
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2 / pi)
    X = torch.tanh(sqrt_2_over_pi * (a + 0.044715 * a * a * a))
    da = 0.5 * b * ((1 + X) + a * (1 - X * X) * (sqrt_2_over_pi * (1 + 3 * 0.044715 * a * a))) * dc
    db = 0.5 * a * (1 + X) * dc
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

    golden_forward = geglu_golden_forward(a, b)
    golden_da, golden_db = geglu_golden_backward(a, b, dc)

    a0 = a.clone().requires_grad_(True)
    b0 = b.clone().requires_grad_(True)
    y1 = GELUMulFunction.apply(a0, b0)
    y1.backward(dc)

    a1 = a.clone().requires_grad_(True)
    b1 = b.clone().requires_grad_(True)
    y2 = LigerGELUMulFunction.apply(a1, b1)
    y2.backward(dc)

    torch.testing.assert_close(golden_forward, y1, atol=atol, rtol=rtol)
    torch.testing.assert_close(y2, y1, atol=atol, rtol=rtol)

    torch.testing.assert_close(golden_da, a0.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(a1.grad, a0.grad, atol=atol, rtol=rtol)

    torch.testing.assert_close(golden_db, b0.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(b1.grad, b0.grad, atol=atol, rtol=rtol)

    return


arg_to_torch_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}


#Benchmark
def run_benchmark(args):
    config = []

    for mode in args.mode.split(','):
        if (args.M_benchmark):
            val = args.M_start
            x_vals_list = []
            while val <= args.M_end:
                x_vals_list.append(val)
                val += args.M_step
            mn_args = {'N': args.N_start}
            plot_name = str("gegle-performance_" + args.dtype + "_N" + str(args.N_start) + "_M" + str(args.M_start) +
                            "-" + str(args.M_end) + "-" + str(args.M_step) + "_mode_" + mode)
            x_names = ['M']
        else:
            x_vals_list = [i for i in range(args.N_start, args.N_end, args.N_step)]
            mn_args = {'M': args.M_start}
            x_names = ['N']
            plot_name = str("gegle-performance_" + args.dtype + "_M" + str(args.M_start) + "_N" + str(args.N_start) +
                            "-" + str(args.N_end) + "-" + str(args.N_step) + "_mode_" + mode)
        mn_args['mode'] = mode
        dtype = arg_to_torch_dtype[args.dtype]

        print(plot_name)
        config.append(
            triton.testing.Benchmark(
                x_names=x_names,
                x_vals=x_vals_list,
                line_arg='provider',
                line_vals=['triton', 'triton_Liger', 'torch_accurate'],
                line_names=['Triton', 'Triton_Liger', 'Torch_accurate'],
                styles=[('red', '-'), ('blue', '-'), ('green', '-'), ('yellow', '-')],
                ylabel="ms",
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
                fn = lambda: geglu_forward(a, b)
            else:
                fn = lambda: geglu_backward(a, b, dc)

        if provider == 'triton_Liger':
            if mode == 'fwd':
                fn = lambda: geglu_forward_Liger(a, b)
            else:
                fn = lambda: geglu_backward_Liger(a, b, dc)

        if provider == 'torch_accurate':
            if mode == 'fwd':
                fn = lambda: geglu_golden_forward(a, b)
            else:
                fn = lambda: geglu_golden_backward(a, b, dc)

        ms = triton.testing.do_bench(fn)
        return ms

    benchmark.run(save_path=".", show_plots=True, print_data=True)
    return


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark GeGLU",
        allow_abbrev=False,
    )
    parser.add_argument('-M', "--M_start", default="600", type=int)
    parser.add_argument('-Ms', "--M_step", default="100", type=int)  #This is multiplicative step
    parser.add_argument('-Me', "--M_end", default="2600", type=int)

    parser.add_argument('-N', "--N_start", default="8000", type=int)
    parser.add_argument('-Ns', "--N_step", default="1000", type=int)
    parser.add_argument('-Ne', "--N_end", default="32000", type=int)

    parser.add_argument('-Mb', "--M_benchmark", action='store_true')
    parser.add_argument('-d', "--dtype", default="fp16")
    parser.add_argument("-mode", default='fwd,bwd', help="Pass mode: fwd, bwd or both(default).")
    return parser.parse_args()


def main():
    args = parse_args()
    run_benchmark(args)
    # test_correctness(256, 6000, torch.float32, 1e-0, 2e-6, "cuda")
    return


if __name__ == "__main__":
    sys.exit(main())
