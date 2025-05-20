import os
import json
import torch

import triton
import triton.language as tl
import pytest
from triton.runtime.cache import default_cache_dir


def do_bench(kernel_call, quantiles):
    return triton.testing.do_bench(kernel_call, quantiles=quantiles, warmup=1, rep=1)


def get_save_config_root_dir():
    return f"{default_cache_dir()}/configs"


def get_gpu_label():
    gpu_name = torch.cuda.get_device_name().replace(" ", "_").replace("/", "_")
    return gpu_name


def get_device_name(device: str):
    if device == "cuda":
        return get_gpu_label()
    else:
        raise NotImplementedError


def get_save_config_file(fname, device):
    root_dir = get_save_config_root_dir()
    config_fpath = os.path.join(root_dir, fname, get_device_name(device), "config.json")
    return config_fpath


@pytest.mark.parametrize('device', ['cuda'])
def test_save_config(device: str):
    N = 1024
    src = torch.zeros(N, device=device)

    configs = [triton.Config(kwargs={'BLOCK_SIZE': 32}), triton.Config(kwargs={'BLOCK_SIZE': 128})]

    @triton.utils.hcutune(configs=configs, key=['N', 'DUMMY'], restore_value=['src'], do_bench=do_bench)
    @triton.jit
    def _kernel(src, N, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offsets, mask=offsets < N) + 1
        tl.store(src + offsets, x, mask=offsets < N)

    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']), )
    _kernel[grid](src, N)
    triton.testing.assert_close(src, torch.ones_like(src))
    assert len(_kernel.cache) == 1

    # check configs
    config_fpath = get_save_config_file("_kernel", device)
    assert os.path.exists(config_fpath)
    with open(config_fpath, "r") as fp:
        data = json.load(fp)
    for k in ["arg_names", "keys", "configs"]:
        assert k in data
    assert len(data['configs']) == 1
    assert list(data['configs'].keys())[0] == "(1024, 'torch.float32')"
    assert (value := list(data['configs'].values())[0])['BLOCK_SIZE'] == 32 or \
            value['BLOCK_SIZE'] == 128


@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('pass_kwargs_to_kernel', [False, True])
@pytest.mark.parametrize('use_cuda_graph', [False, True])
def test_save_configs(pass_kwargs_to_kernel: bool, device: str, use_cuda_graph: bool):
    N = [32, 64, 128]
    M = [i for i in range(1, 10, 3)]

    configs = [triton.Config(kwargs={'BLOCK_SIZE': 32}), triton.Config(kwargs={'BLOCK_SIZE': 128})]

    @triton.utils.hcutune(configs=configs, key=['M', 'DUMMY', 'N'], restore_value=['src'], do_bench=do_bench,
                          use_cuda_graph=use_cuda_graph)
    @triton.jit
    def _kernel2(src, M, N, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offsets, mask=offsets < N) + 1
        tl.store(src + offsets, x, mask=offsets < N)

    fn = _kernel2
    for m in M:
        for n in N:
            src = torch.zeros(n, device=device)
            grid = lambda META: (triton.cdiv(n, META['BLOCK_SIZE']), )
            if pass_kwargs_to_kernel:
                # the key word args could be in arbitrary order.
                fn[grid](N=n, M=m, src=src)
            else:
                fn[grid](src, m, n)
    assert len(fn.cache) == len(N) * len(M)

    # check configs
    config_fpath = get_save_config_file("_kernel2", device)
    with open(config_fpath, "r") as fp:
        data = json.load(fp)
    assert len(data['configs']) == len(N) * len(M)


@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('pass_kwargs_to_kernel', [False, True])
def test_restore_config(pass_kwargs_to_kernel, device):
    N = 1024
    src = torch.zeros(N, device=device)

    @triton.jit
    def _kernel(src, N, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offsets, mask=offsets < N) + 1
        tl.store(src + offsets, x, mask=offsets < N)

    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']), )

    if pass_kwargs_to_kernel:
        # the key word args could be in arbitrary order.
        args = ()
        kwargs = {'N': N, 'src': src}
    else:
        args = (src, N)
        kwargs = {}

    config_fpath = get_save_config_file("_kernel", device)
    assert os.path.exists(config_fpath)
    config = triton.utils.get_optimal_config("_kernel", *args, **kwargs)
    _kernel[grid](*args, **kwargs, **config)
    triton.testing.assert_close(src, torch.ones_like(src))
