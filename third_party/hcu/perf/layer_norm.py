"""
Layer Normalization
"""
import argparse
import sys
import pytest

import torch
import triton
import triton.language as tl
import os
import json
import math
from itertools import product

def calculate_settings(n):
    # reference: https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/utils.py#L43

    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds "
            f"the recommended Triton blocksize = {MAX_FUSED_SIZE}."
        )

    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def get_cuda_autotune_config():
    return [
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=16, num_stages=1),
    ]


def get_hip_autotune_config():
    return [
        triton.Config({'waves_per_eu': we}, num_warps=wa, num_stages=1) for we, wa in product([1, 2, 4], [4, 8, 16])
    ]


def get_autotune_config():
    if is_cuda():
        return get_cuda_autotune_config()
    else:
        return get_hip_autotune_config()


@triton.autotune(configs=get_autotune_config(), key=['n_rows', 'n_cols'], use_cuda_graph=True)
@triton.jit
def layernorm_kernel(x_ptr, y_ptr, w_ptr, b_ptr, mean_ptr, rstd_ptr, x_row_stride, y_row_stride, n_rows, n_cols, eps,
                     BLOCK_SIZE: tl.constexpr):
    #program id
    row = tl.program_id(0)
    x_ptr_start = x_ptr + (row * x_row_stride)
    y_ptr_start = y_ptr + (row * y_row_stride)

    #calculate mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    x_block = tl.load(x_ptr_start + col_offsets, mask=col_offsets < n_cols, other=0.).to(tl.float32)
    _mean += x_block
    mean = tl.sum(_mean, axis=0) / n_cols

    #variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    x_temp = tl.where(col_offsets < n_cols, x_block - mean, 0.)
    _var += x_temp * x_temp
    var = tl.sum(_var, axis=0) / n_cols
    rstd = tl.rsqrt(var + eps)

    # Write mean / rstd
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)

    #Normalize and store
    w_block = tl.load(w_ptr + col_offsets, mask=col_offsets < n_cols, other=0.0)
    b_block = tl.load(b_ptr + col_offsets, mask=col_offsets < n_cols, other=0.0)
    y_block = x_temp * rstd * w_block + b_block
    tl.store(y_ptr_start + col_offsets, y_block, mask=col_offsets < n_cols)


@triton.autotune(configs=get_autotune_config(), key=['n_rows', 'n_cols'], use_cuda_graph=True)
@triton.jit
def _layer_norm_backward_kernel(
    X_ptr,  # pointer to input, shape (n_rows, n_cols)
    W_ptr,  # pointer to weights, shape (n_cols,)
    Mean_ptr,  # pointer to mean, shape (n_rows,)
    RSTD_ptr,  # pointer to rstd, shape (n_rows,)
    DX_ptr,  # pointer to input grad, shape (n_rows, n_cols)
    DW_ptr,  # pointer to weights grad, shape (n_cols,)
    DB_ptr,  # pointer to bias grad, shape (n_cols,)
    DY_ptr,  # pointer to output grad, shape (n_rows, n_cols)
    stride_x,  # stride of each row in input
    stride_dx,  # stride of each row in input grad
    stride_dw,  # stride of each row in weights grad
    stride_db,  # stride of each row in bias grad
    stride_dy,  # stride of each row in output grad
    n_rows,
    n_cols,
    rows_per_program: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    dtype: tl.constexpr,
):
    """
    References:
    https://arxiv.org/abs/1607.06450
    https://github.com/karpathy/llm.c/blob/master/doc/layernorm/layernorm.md
    https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/triton/layer_norm.py
    """
    row_block_id = tl.program_id(0)
    row_start = row_block_id * rows_per_program
    row_end = min((row_block_id + 1) * rows_per_program, n_rows)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    dw_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    db_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    X_ptr += row_start * stride_x
    Mean_ptr += row_start
    RSTD_ptr += row_start
    DX_ptr += row_start * stride_dx
    DY_ptr += row_start * stride_dy

    for _ in range(row_start, row_end):
        x = tl.load(X_ptr + cols, mask=mask, other=0.0)
        w = tl.load(W_ptr + cols, mask=mask, other=0.0)
        dy = tl.load(DY_ptr + cols, mask=mask, other=0.0)
        mean = tl.load(Mean_ptr)
        rstd = tl.load(RSTD_ptr)

        x_hat = (x - mean) * rstd
        wdy = w * dy
        c1 = tl.sum(x_hat * wdy, axis=0) / n_cols
        c2 = tl.sum(wdy, axis=0) / n_cols
        dx = (wdy - (x_hat * c1 + c2)) * rstd
        tl.store(DX_ptr + cols, dx.to(dtype), mask=mask)

        dw_row += dy * x_hat
        db_row += dy

        X_ptr += stride_x
        Mean_ptr += 1
        RSTD_ptr += 1
        DX_ptr += stride_dx
        DY_ptr += stride_dy

    tl.store(DW_ptr + row_block_id * stride_dw + cols, dw_row.to(dtype), mask=mask)
    tl.store(DB_ptr + row_block_id * stride_db + cols, db_row.to(dtype), mask=mask)


class LayerNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps=1e-5):         
        y = torch.empty_like(x)
        M, N = x.shape
        mean = torch.empty((M, ), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M, ), dtype=torch.float32, device=x.device)
        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        
        if N > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        
        # heuristics for number of warps
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        layernorm_kernel[(M, )](x, y, weight, bias, mean, rstd, x.stride(0), y.stride(0), M, N, eps, BLOCK_SIZE)
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps

        return y
    
    @staticmethod
    def backward(ctx, dy):
        X, W, B, Mean, RSTD = ctx.saved_tensors
        shape = dy.shape
        dim = shape[-1]
        dY = dy.view(-1, dim)
        n_rows, n_cols = dY.shape
        DX = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
        sm_count = torch.cuda.get_device_properties(X.device).multi_processor_count
        # GROUP_SIZE_M = 64
        # if n_cols <= 8192: GROUP_SIZE_M = 96
        # if n_cols <= 4096: GROUP_SIZE_M = 128
        # if n_cols <= 1024: GROUP_SIZE_M = 256
        # sm_count = GROUP_SIZE_M
        _DW = torch.empty((sm_count, n_cols), dtype=W.dtype, device=W.device)
        _DB = torch.empty((sm_count, n_cols), dtype=W.dtype, device=W.device)

        BLOCK_SIZE, num_warps = calculate_settings(n_cols)
        if n_cols > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

        rows_per_program = math.ceil(n_rows / sm_count)
        grid = (sm_count,)
        triton_dtype = tl.float32 if X.dtype == torch.float32 else tl.bfloat16

        _layer_norm_backward_kernel[grid](
            X,
            W,
            Mean,
            RSTD,
            DX,
            _DW,
            _DB,
            dY,
            X.stride(0),
            DX.stride(0),
            _DW.stride(0),
            _DB.stride(0),
            dY.stride(0),
            n_rows,
            n_cols,
            rows_per_program,
            BLOCK_SIZE=BLOCK_SIZE,
            dtype=triton_dtype,
        )

        DW = _DW.sum(dim=0).to(W.dtype)
        DB = _DB.sum(dim=0).to(W.dtype)

        DX = DX.view(*shape)
        return DX, None, DW, DB, None
        


layernorm = LayerNorm.apply


def torch_layernorm(x, w_shape, w, b):
    M, N = x.shape
    y_torch = torch.nn.functional.layer_norm(x, w_shape, w, b, eps=1e-5)
    return y_torch


def run_layernorm(M, N):
    print(f"Running Layernorm on shape ({M},{N})")
    torch.manual_seed(0)
    x = torch.randn(M, N, device='cuda')
    w_shape = (N, )
    w = torch.rand(w_shape, device='cuda')
    b = torch.rand(w_shape, device='cuda')
    y_triton = layernorm(x, w_shape, w, b)

    return y_triton

@pytest.mark.parametrize('M', [4096])
@pytest.mark.parametrize('N', [512 * i for i in range(2, 32)])
def test_layernorm(M, N, eps=1e-5):
    torch.manual_seed(0)
    x = -2.3 + 0.5 * torch.randn(M, N, device='cuda')
    w_shape = (N, )
    w = torch.rand(w_shape, device='cuda', requires_grad=True)
    b = torch.rand(w_shape, device='cuda', requires_grad=True)
    dy = 0.1 * torch.randn_like(x)

    x.requires_grad_(True)

    # forward pass
    y_triton = layernorm(x, w_shape, w, b, eps)
    y_ref = torch.nn.functional.layer_norm(x, w_shape, w, b, eps)

    # backward pass (triton)
    y_triton.backward(dy, retain_graph=True)
    dx_tri, dw_tri, db_tri = [_.grad.clone() for _ in [x, w, b]]
    x.grad, w.grad, b.grad = None, None, None

    #backward pass (torch)
    y_ref.backward(dy, retain_graph=True)
    dx_ref, dw_ref, db_ref = [_.grad.clone() for _ in [x, w, b]]
    
    print (dx_tri, dx_ref)
    print (dw_tri, dw_ref)
    print (db_tri, db_ref)

    assert torch.allclose(y_triton, y_ref, atol=1e-03, rtol=0)
    assert torch.allclose(dx_tri, dx_ref, atol=1e-03, rtol=0)
    assert torch.allclose(db_tri, db_ref, atol=1e-03, rtol=0)
    assert torch.allclose(dw_tri, dw_ref, atol=1e-03, rtol=0)
    print ("triton vs torch Comparison successful")

#Benchmark
arg_to_torch_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}


def get_benchmark_shapes(args):
    # M, N
    benchmark_shapes = {}
    if args.M_benchmark:
        N = args.N_start
        assert args.M_start > 0 and args.M_end, "m start and end values need to be positive!"
        m_range = int(math.log2(args.M_end / args.M_start)) + 1
        shapes = [(args.M_start * (2**i), N) for i in range(0, m_range)]
        benchmark_shapes['M'] = shapes
    else:
        M = args.M_start
        shapes = [(M, n) for n in range(args.N_start, args.N_end, args.N_step)]
        benchmark_shapes['N'] = shapes

    return benchmark_shapes


def get_model_shapes(args):
    model_shapes = {}
    # If user did not provide an absolute path, resolve relative path from script directory
    if not os.path.isabs(args.model_configs):
        config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.model_configs)
    else:
        config_file = args.model_configs

    with open(config_file, 'r') as f:
        configs = json.load(f)

    if args.model != "all": 
        model_name = args.model
        # Check if the model exists
        if model_name not in configs:
            raise ValueError(f"Model '{model_name}' not found in {config_file}")
        # Handle a specific model
        series_config = configs[model_name]
        for model_size, config in series_config.items():
            seq_len = args.sl if args.sl else config["max_ctx_len"]
            M, N = seq_len, config["hidden_size"]
            model_id = model_name + '_' + model_size
            model_shapes[model_id] = [(M, N)]
    else:
        # Handle all models
        for model_name, series_config in configs.items():
            for model_size, config in series_config.items():
                seq_len = args.sl if args.sl else config["max_ctx_len"]
                M, N = seq_len, config["hidden_size"]
                model_id = model_name + '_' + model_size
                model_shapes[model_id] = [(M, N)]

    return model_shapes


def get_shapes(args):
    return get_model_shapes(args) if args.model else get_benchmark_shapes(args)


def run_benchmark(args):
    benchmark_shapes = get_shapes(args)
    config = []
    x_names = ['M', 'N']
    other_args = {}
    for name in benchmark_shapes.keys():
        if args.model:
            plot_name = str("layernorm-performance_for_model_" + name)
        else:
            plot_name = str("layernorm-performance_" + args.dtype + name)

        dtype = arg_to_torch_dtype[args.dtype]
        if args.mode == 'fwd' or args.mode == 'both':
            fwd_plot_name = plot_name + "_forward_pass(GBS)"
            other_args['mode'] = 'forward' 
            config.append(
                triton.testing.Benchmark(
                    x_names=x_names,
                    x_vals=benchmark_shapes[name],
                    line_arg='provider',
                    line_vals=['triton', 'torch'],
                    line_names=[
                        "Triton",
                        "Torch",
                    ],
                    styles=[('blue', '-'), ('green', '-')],
                    ylabel="GB/s",
                    plot_name=fwd_plot_name,
                    args=other_args,
                ))

        if args.mode == 'bwd' or args.mode == 'both':
            bwd_plot_name = plot_name + "_backward_pass(GBS)"
            other_args['mode'] = 'backward'
            config.append(
                triton.testing.Benchmark(
                    x_names=x_names,
                    x_vals=benchmark_shapes[name],
                    line_arg='provider',
                    line_vals=['triton', 'torch'],
                    line_names=[
                        "Triton",
                        "Torch",
                    ],
                    styles=[('blue', '-'), ('green', '-')],
                    ylabel="GB/s",
                    plot_name=bwd_plot_name,
                    args=other_args,
                ))

    @triton.testing.perf_report(config)
    def benchmark(M, N, provider, mode='forward'):
        x = torch.randn(M, N, device='cuda', dtype=dtype)
        w_shape = (N, )
        w = torch.rand(w_shape, device='cuda', dtype=dtype)
        b = torch.rand(w_shape, device='cuda', dtype=dtype)
        dy = .1 * torch.randn_like(x)
        stream = torch.cuda.Stream()
        torch.cuda.set_stream(stream)

        def y_fwd():
            if provider == 'triton':
                return layernorm(x, w_shape, w, b) 
            if provider == 'torch':
                return torch_layernorm(x, w_shape, w, b)

        if mode == 'forward':
            gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
            ms = triton.testing.do_bench(y_fwd)
        elif mode == 'backward':
            x.requires_grad_(True)
            w.requires_grad_(True)
            b.requires_grad_(True)

            y = y_fwd()
            gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)  # noqa: F811, E704
            ms = triton.testing.do_bench(lambda: y.backward(dy, retain_graph=True), grad_to_none=[x])
        else:
            raise ValueError(f"mode {mode} is not supported!")

        return gbps(ms)

    benchmark.run(save_path=".", show_plots=True, print_data=True)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark Layernorm",
        allow_abbrev=False,
    )

    parser.add_argument('-M', "--M_start", default="4096", type=int)
    parser.add_argument('-Ms', "--M_step", default="0", type=int)
    parser.add_argument('-Me', "--M_end", default="4096", type=int)
    parser.add_argument('-Mb', "--M_benchmark", action='store_true', default=False,
                        help='whether do benchmark on M or N, default N') 
    parser.add_argument('-N', "--N_start", default="1024", type=int)
    parser.add_argument('-Ns', "--N_step", default="512", type=int)
    parser.add_argument('-Ne', "--N_end", default="16382", type=int)
    parser.add_argument('-d', "--dtype", default="fp32")
    parser.add_argument('-nb', "--no_benchmark", action='store_true', default=False,
                        help='no benchmark, just test ops functional')
    parser.add_argument('-mode', "--mode", default='fwd', type=str,
                        help='run forward or backward or both passes, default is forward')

    # model related inputs
    parser.add_argument('-model_configs', type=str, default="model_configs.json", help="Model config json file.")

    def get_available_models(config_file='model_configs.json'):
        import json
        """Load model names from the configuration file."""
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_file)
        with open(config_path, 'r') as f:
            configs = json.load(f)
        return list(configs.keys())

    available_models = get_available_models()  # Dynamically load model names
    model_help = ("Model name to benchmark. Select from: [" + ", ".join(available_models) +
                  "]. Use 'all' to benchmark all models or leave blank for the default benchmark script.")
    parser.add_argument('-model', type=str, default=None, help=model_help)
    parser.add_argument(
        '-sl', type=int, default=0,
        help="Sequence length used together with model. Defaults to max_seq_len from model config if not provided.")

    return parser.parse_args()
  

def main():
    args = parse_args()
    print (args)
    if args.no_benchmark:
        test_layernorm(args.M_start, args.N_start)
    else:
        run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())