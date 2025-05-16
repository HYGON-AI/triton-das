import os
import json
import tempfile
import torch

import triton
import triton.language as tl
import pytest
from triton.runtime.cache import get_home_dir


def do_bench(kernel_call, quantiles):
    return triton.testing.do_bench(kernel_call, quantiles=quantiles, warmup=1, rep=1)


def get_save_config_dir(dir):
    if not dir:
        dir = f"{get_home_dir()}/.triton/json"
    dir = f"{dir}/_kernel/configs"
    return dir


@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('pass_kwargs_to_kernel', [False, True])
@pytest.mark.parametrize('save_config_dir', [None, tempfile.mkdtemp()])
def test_save_config(pass_kwargs_to_kernel: bool, device: str, save_config_dir: str):
    N = 1024
    src = torch.zeros(N, device=device)

    configs = [triton.Config(kwargs={'BLOCK_SIZE': 32}), triton.Config(kwargs={'BLOCK_SIZE': 128})]

    @triton.utils.hcutune(configs=configs, key=['N'], restore_value=['src'], do_bench=do_bench,
                          save_config_dir=save_config_dir)
    @triton.jit
    def _kernel(src, N, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offsets, mask=offsets < N) + 1
        tl.store(src + offsets, x, mask=offsets < N)

    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']), )
    if pass_kwargs_to_kernel:
        # the key word args could be in arbitrary order.
        _kernel[grid](N=N, src=src)
    else:
        _kernel[grid](src, N)
    triton.testing.assert_close(src, torch.ones_like(src))
    assert len(_kernel.cache) == 1

    # check configs
    save_config_dir = get_save_config_dir(save_config_dir)
    assert os.path.exists(save_config_dir)
    all_items = os.listdir(save_config_dir)
    assert len(all_items) == 1
    configfile = all_items[0]
    assert "device_name=" in configfile and "dtype=" in configfile \
            and configfile.endswith(".json")
    with open(f"{save_config_dir}/{configfile}", "r") as fp:
        configs = json.load(fp)
    assert len(configs) == 1


@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('pass_kwargs_to_kernel', [False, True])
@pytest.mark.parametrize('save_config_dir', [tempfile.mkdtemp()])
def test_save_configs(pass_kwargs_to_kernel: bool, device: str, save_config_dir: str):
    N = [64, 128, 256]
    M = [i for i in range(1, 10, 3)]

    configs = [triton.Config(kwargs={'BLOCK_SIZE': 32}), triton.Config(kwargs={'BLOCK_SIZE': 128})]

    @triton.utils.hcutune(configs=configs, key=['M', 'N'], restore_value=['src'], do_bench=do_bench,
                          save_config_dir=save_config_dir)
    @triton.jit
    def _kernel(src, M, N, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offsets, mask=offsets < N) + 1
        tl.store(src + offsets, x, mask=offsets < N)

    for m in M:
        for n in N:
            src = torch.zeros(n, device=device)
            grid = lambda META: (triton.cdiv(n, META['BLOCK_SIZE']), )
            if pass_kwargs_to_kernel:
                # the key word args could be in arbitrary order.
                _kernel[grid](N=n, M=m, src=src)
            else:
                _kernel[grid](src, m, n)
    assert len(_kernel.cache) == len(N) * len(M)

    save_config_dir = get_save_config_dir(save_config_dir)
    all_items = os.listdir(save_config_dir)
    assert len(all_items) == 1
    with open(f"{save_config_dir}/{all_items[0]}", "r") as fp:
        configs = json.load(fp)
    assert len(configs) == len(_kernel.cache)


@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('save_config_dir', [tempfile.mkdtemp()])
def test_custom_create_fn_key(device: str, save_config_dir: str):
    N = [64, 128, 256]
    M = [i for i in range(1, 10, 3)]

    configs = [triton.Config(kwargs={'BLOCK_SIZE': 32}), triton.Config(kwargs={'BLOCK_SIZE': 128})]

    def custom_create_fn_key(nargs, keys):
        fn = f"M={nargs['M']},device_name=gpu,dtype=float32.json"
        kval = [nargs[o] for o in keys if o != 'M']
        key = str(tuple(kval)) if len(kval) > 1 else str(kval[0])
        return fn, key

    from functools import partial
    from triton.utils.hcutuner import default_save_config
    @triton.utils.hcutune(configs=configs, key=['M', 'N'], restore_value=['src'], do_bench=do_bench,
                          save_config_dir=save_config_dir,
                          do_save_config=partial(default_save_config,
                                                 create_filename_and_key=custom_create_fn_key))
    @triton.jit
    def _kernel(src, M, N, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offsets, mask=offsets < N) + 1
        tl.store(src + offsets, x, mask=offsets < N)

    for m in M:
        for n in N:
            src = torch.zeros(n, device=device)
            grid = lambda META: (triton.cdiv(n, META['BLOCK_SIZE']), )
            _kernel[grid](src, m, n)
    assert len(_kernel.cache) == len(N) * len(M)

    save_config_dir = get_save_config_dir(save_config_dir)
    all_items = os.listdir(save_config_dir)
    assert len(all_items) == len(M)
    num_configs = []
    for f in all_items:
        with open(f"{save_config_dir}/{f}", "r") as fp:
            num_configs.append(len(json.load(fp)))
    assert sum(num_configs) == len(_kernel.cache)
