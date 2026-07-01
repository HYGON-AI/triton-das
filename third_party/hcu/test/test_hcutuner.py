import os
import json
import torch
import tempfile
from typing import Union
import pytest
import subprocess

os.environ["TRITON_HCUTUNE"] = "1"
import triton
import triton.language as tl
from triton.backends.hcu.autotuner import (
    get_config_key, get_gpu_label, get_cache_dir, get_graph_cache_dir,
    get_config_cache_dir, graphtune, ConfigLoader, PruneConfigLoader,
    merge_caches, GraphConfigLoader
)
from triton.backends.hcu.testing import (
    split_x_vals, set_device
)


def do_bench(kernel_call, quantiles):
    return triton.testing.do_bench(kernel_call, quantiles=quantiles, warmup=1, rep=1)


def get_device_name(device: str):
    if device == "cuda":
        return get_gpu_label()
    else:
        raise NotImplementedError


def get_save_config_files(fname, device):
    root_dir = get_config_cache_dir()
    sub_dir = os.path.join(root_dir, fname, get_device_name(device))
    res = []
    for dirpath, _, filenames in os.walk(sub_dir):
        for filename in filenames:
            if filename != "config.json":
                continue
            res.append(os.path.join(dirpath, filename))
    return res


def get_config_cache_files(fn, *args, **kwargs):
    res = []
    cache_manager = fn.get_config_cache_manager()
    for dirpath, _, filenames in os.walk(cache_manager.key_dir):
        for filename in filenames:
            if filename != "config.json":
                continue
            res.append(os.path.join(dirpath, filename))
    return res


def get_graph_config_files(graph_name, device):
    root_dir = get_graph_cache_dir()
    sub_dir = os.path.join(root_dir, graph_name, get_device_name(device))
    res = []
    for dirpath, _, filenames in os.walk(sub_dir):
        for filename in filenames:
            if filename != "config.json":
                continue
            res.append(os.path.join(dirpath, filename))
    return res


def get_optimal_config(op_name: str, key: Union[list, dict, tuple, str], device_name: str = None):
    device_name = device_name if device_name else get_gpu_label()
    config_loader = ConfigLoader()
    tuned = config_loader.get_tuned_cache(op_name, device_name)
    if tuned:
        res = tuned.get_optimal_config(key)
        if res:
            return res
        else:
            print(
                f"[hcutuner] WARNING: Not found optimal config for {op_name} !!! cache dir: {get_cache_dir()},\t"
                f"device name: {device_name},\tconfig key: {key}"
            )
            return None
    print(
        f"[hcutuner] WARNING: Not found config cache for ({op_name}, {device_name})."
    )
    return None


def get_config_cache(op_name: str, device_name: str = None):
    device_name = device_name if device_name else get_gpu_label()
    config_loader = ConfigLoader()
    tuned = config_loader.get_tuned_cache(op_name, device_name)
    if not tuned:
        print(
            f"[hcutuner] WARNING: Not found config cache for ({op_name}, {device_name})."
        )
        return (None, None)
    return (tuned.cache['configs'], tuned.cache['paths'])


@pytest.mark.parametrize('device', ['cuda'])
def test_save_config(device: str):
    N = 1024
    src = torch.zeros(N, device=device)

    configs = [triton.Config(kwargs={'BLOCK_SIZE': 32}), triton.Config(kwargs={'BLOCK_SIZE': 128})]

    @triton.autotune(configs=configs, key=['N', 'DUMMY'], restore_value=['src'], do_bench=do_bench)
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
    config_fpaths = get_save_config_files("_kernel", device)
    for config_fpath in config_fpaths:
        assert os.path.exists(config_fpath)
        with open(config_fpath, "r") as fp:
            data = json.load(fp)
        for k in ["key", "configs", "timings"]:
            assert k in data
        assert len(data['configs']) == len(data['timings']) == 1
        assert list(data['configs'].keys())[0] == "(1024, 'torch.float32')"
        assert list(data['timings'].keys())[0] == "(1024, 'torch.float32')"
        assert (value := list(data['configs'].values())[0])['BLOCK_SIZE'] == 32 or \
                value['BLOCK_SIZE'] == 128


@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('pass_kwargs_to_kernel', [False, True])
@pytest.mark.parametrize('use_cuda_graph', [False, True])
def test_save_configs(pass_kwargs_to_kernel: bool, device: str, use_cuda_graph: bool):
    N = [32, 64, 128]
    M = [i for i in range(1, 10, 3)]

    configs = [triton.Config(kwargs={'BLOCK_SIZE': 32}), triton.Config(kwargs={'BLOCK_SIZE': 128})]

    @triton.autotune(configs=configs, key=['M', 'DUMMY', 'N'], restore_value=['src'], do_bench=do_bench,
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
    config_fpaths = get_save_config_files("_kernel2", device)
    with open(config_fpaths[0], "r") as fp:
        data = json.load(fp)
    assert len(data['configs']) == len(data['timings']) == len(N) * len(M)


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

    config_loader = ConfigLoader()
    key_dict = {
        'N': N,
        'src': src.dtype,
    }
    config = get_optimal_config("_kernel", key_dict)
    _kernel[grid](*args, **kwargs, **config)
    triton.testing.assert_close(src, torch.ones_like(src))
    config2 = get_optimal_config("_kernel", [N, src.dtype])
    assert config == config2


@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('world_size', [4])
def test_mp_save_config(device, world_size):
    code = """
import os
import sys
import torch
os.environ["TRITON_HCUTUNE"] = "1"
import triton
import triton.language as tl

def do_bench(kernel_call, quantiles):
    return triton.testing.do_bench(kernel_call, quantiles=quantiles, warmup=1, rep=1)

N = int(sys.argv[1])
device = sys.argv[2]

configs = [triton.Config(kwargs={'BLOCK_SIZE': 32}), triton.Config(kwargs={'BLOCK_SIZE': 128})]

@triton.autotune(configs=configs, key=['N'], restore_value=['src'], do_bench=do_bench)
@triton.jit
def _kernel_mp(src, N, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(src + offsets, mask=offsets < N) + 1
    tl.store(src + offsets, x, mask=offsets < N)

def main():
    src = torch.randn(N, device=device)
    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']), )
    _kernel_mp[grid](src, N)

if __name__ == "__main__":
    main()
    """

    temp_filename = None
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code.strip())
        temp_filename = f.name

    N = [i * 128 for i in range(1, world_size + 1)]
    processes = []
    for i in range(world_size):
        cmd = ["python3", temp_filename, str(N[i]), device]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append(p)

    for p in processes:
        p.wait()
        assert p.returncode == 0

    config_fpaths = get_save_config_files("_kernel_mp", device)
    assert len(config_fpaths) == world_size
    # run id check
    run_ids = [f.split('/')[-2] for f in config_fpaths]
    assert len(run_ids) == len(set(run_ids))
    # key check
    keys = []
    for f in config_fpaths:
        with open(f, "r") as fp:
            data = json.load(fp)
            keys += list(data['configs'].keys())
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('world_size', [4])
def test_mp_restore_config(device, world_size):
    N = 256
    src = torch.zeros(N, device=device)

    @triton.jit
    def _kernel_mp(src, N, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offsets, mask=offsets < N) + 1
        tl.store(src + offsets, x, mask=offsets < N)

    ConfigLoader()
    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']), )
    key_dict = {
        'src': src,
        'N': N,
    }
    config = get_optimal_config("_kernel_mp", key_dict)
    _kernel_mp[grid](src, N, **config)
    triton.testing.assert_close(src, torch.ones_like(src))


@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('always_tuning', [True, False])
def test_restore_cache(monkeypatch, device: str, always_tuning: bool):
    monkeypatch.setenv("TRITON_CACHE_DIR", tempfile.mkdtemp())
    monkeypatch.setenv("TRITON_HCUTUNE_ALWAYS_TUNING", "1" if always_tuning else "0")
    configs = [triton.Config(kwargs={'BLOCK_SIZE': 32}), triton.Config(kwargs={'BLOCK_SIZE': 128})]
    @triton.autotune(configs=configs, key=['M', 'N'], restore_value=['src'], do_bench=do_bench)
    @triton.jit
    def _kernel_cache(src, M, N, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        N = N * M // M # dummy M
        x = tl.load(src + offsets, mask=offsets < N) + 1
        tl.store(src + offsets, x, mask=offsets < N)

    fn = _kernel_cache
    SIZE = 4
    M = [i * 128 for i in range(1, SIZE + 1)]
    N = [i * 128 for i in range(1, SIZE + 1)]
    src = [torch.zeros(n, device=device) for n in N]

    for _src, m, n in zip(src, M, N):
        grid = lambda META: (triton.cdiv(n, META['BLOCK_SIZE']), )
        fn[grid](_src, m, n)
        triton.testing.assert_close(_src, torch.ones_like(_src))

    files = get_config_cache_files(fn, src[0], M[0], N[0])
    assert len(files) == 1
    fn.restore_tuned_cache()
    assert len(fn.cache) == len(fn._configs_timings) == SIZE
    for k, _ in fn.cache.items():
        assert k in fn._configs_timings

    # cache hit
    keys = [str(o) for o in fn.cache.keys()]
    for _src, m, n in zip(src, M, N):
        _, key = get_config_key(fn.arg_names, fn.keys, _src, m, n)
        assert str(key) in keys


@pytest.mark.parametrize('device', ['cuda'])
def test_cache_key_changes(monkeypatch, device: str):
    monkeypatch.setenv("TRITON_CACHE_DIR", tempfile.mkdtemp())
    M, N = 128, 128
    src = torch.zeros(N, device=device)

    # change configs: {'BLOCK_SIZE': 32} -> {'BLOCK_SIZE': 64}
    configs = [triton.Config(kwargs={'BLOCK_SIZE': 64}), triton.Config(kwargs={'BLOCK_SIZE': 128})]
    @triton.autotune(configs=configs, key=['M', 'N'], restore_value=['src'], do_bench=do_bench)
    @triton.jit
    def _kernel_cache(src, M, N, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        N = N * M // M # dummy M
        x = tl.load(src + offsets, mask=offsets < N) + 1
        tl.store(src + offsets, x, mask=offsets < N)

    # Not found the expected config cache of _kernel_cache
    assert len(get_config_cache_files(_kernel_cache, src, M, N)) == 0

    # change autotune params: use_cuda_graph=True
    configs = [triton.Config(kwargs={'BLOCK_SIZE': 32}), triton.Config(kwargs={'BLOCK_SIZE': 128})]
    @triton.autotune(configs=configs, key=['M', 'N'], restore_value=['src'], do_bench=do_bench,
                          use_cuda_graph=True)
    @triton.jit
    def _kernel_cache(src, M, N, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        N = N * M // M # dummy M
        x = tl.load(src + offsets, mask=offsets < N) + 1
        tl.store(src + offsets, x, mask=offsets < N)

    assert len(get_config_cache_files(_kernel_cache, src, M, N)) == 0

    # change autotune key: ['M', 'N'] -> ['N', 'M']
    configs = [triton.Config(kwargs={'BLOCK_SIZE': 32}), triton.Config(kwargs={'BLOCK_SIZE': 128})]
    @triton.autotune(configs=configs, key=['N', 'M'], restore_value=['src'], do_bench=do_bench)
    @triton.jit
    def _kernel_cache(src, M, N, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        N = N * M // M # dummy M
        x = tl.load(src + offsets, mask=offsets < N) + 1
        tl.store(src + offsets, x, mask=offsets < N)

    assert len(get_config_cache_files(_kernel_cache, src, M, N)) == 0

    # change code
    configs = [triton.Config(kwargs={'BLOCK_SIZE': 32}), triton.Config(kwargs={'BLOCK_SIZE': 128})]
    @triton.autotune(configs=configs, key=['M', 'N'], restore_value=['src'], do_bench=do_bench)
    @triton.jit
    def _kernel_cache(src, M, N, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offsets, mask=offsets < N) + 1
        tl.store(src + offsets, x, mask=offsets < N)

    assert len(get_config_cache_files(_kernel_cache, src, M, N)) == 0


@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('world_size', [4])
def test_mp_cache_config(monkeypatch, device: str, world_size):
    monkeypatch.setenv("TRITON_CACHE_DIR", tempfile.mkdtemp())
    code = """
import os
import sys
import torch
os.environ["TRITON_HCUTUNE"] = "1"
import triton
import triton.language as tl

def do_bench(kernel_call, quantiles):
    return triton.testing.do_bench(kernel_call, quantiles=quantiles, warmup=1, rep=1)

N = int(sys.argv[1])
device = sys.argv[2]

configs = [triton.Config(kwargs={'BLOCK_SIZE': 32}), triton.Config(kwargs={'BLOCK_SIZE': 128})]

@triton.autotune(configs=configs, key=['N'], restore_value=['src'], do_bench=do_bench)
@triton.jit
def _kernel_mp_cache(src, N, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(src + offsets, mask=offsets < N) + 1
    tl.store(src + offsets, x, mask=offsets < N)

def main():
    src = torch.zeros(N, device=device)
    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']), )
    _kernel_mp_cache[grid](src, N)
    triton.testing.assert_close(src, torch.ones_like(src))

if __name__ == "__main__":
    main()
    """

    # multi-process tuning
    temp_filename = None
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code.strip())
        temp_filename = f.name

    N = [i * 64 for i in range(1, world_size + 1)]
    src = [torch.zeros(n, device=device) for n in N]
    processes = []
    for i in range(world_size):
        cmd = ["python3", temp_filename, str(N[i]), device]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append(p)

    for p in processes:
        p.wait()
        assert p.returncode == 0

    configs = [triton.Config(kwargs={'BLOCK_SIZE': 32}), triton.Config(kwargs={'BLOCK_SIZE': 128})]

    @triton.autotune(configs=configs, key=['N'], restore_value=['src'], do_bench=do_bench)
    @triton.jit
    def _kernel_mp_cache(src, N, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offsets, mask=offsets < N) + 1
        tl.store(src + offsets, x, mask=offsets < N)

    files = get_config_cache_files(_kernel_mp_cache, src[0], N[0])
    assert len(files) == world_size


def test_heuristics_decorator(device: str = "cuda"):
    N = 1024
    src = torch.zeros(N, device=device)

    configs = [triton.Config(kwargs={'BLOCK_SIZE': 32}), triton.Config(kwargs={'BLOCK_SIZE': 128})]

    @triton.autotune(configs=configs, key=['N', 'DUMMY'], restore_value=['src'], do_bench=do_bench)
    @triton.heuristics(values={
        "N_DIVISIBLE_BY_16": lambda args: args["N"] % 16 == 0,
    })
    @triton.jit
    def _kernel(src, N, N_DIVISIBLE_BY_16: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offsets, mask=offsets < N) + 1
        tl.store(src + offsets, x, mask=offsets < N)

    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']), )
    _kernel[grid](src, N)
    triton.testing.assert_close(src, torch.ones_like(src))


# def test_hcutune_configured(device: str = "cuda"):
#     N = 1024
#     src = torch.zeros(N, device=device)
#     configs = [triton.Config(kwargs={'BLOCK_SIZE': 32}), triton.Config(kwargs={'BLOCK_SIZE': 128})]

#     @triton.jit
#     def _kernel_hcutune_configured(src, N, BLOCK_SIZE: tl.constexpr):
#         offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
#         x = tl.load(src + offsets, mask=offsets < N) + 1
#         tl.store(src + offsets, x, mask=offsets < N)

#     grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']), )
#     fn = triton.autotune(configs=configs, key=['N'], restore_value=['src'],
#                               do_bench=do_bench)(_kernel_hcutune_configured)
#     fn[grid](src, N)
#     global_config_loader.load_all()

#     fn_infer = triton.autotune_configured()(_kernel_hcutune_configured)
#     src = torch.zeros(N, device=device)
#     fn_infer[grid](src, N)
#     triton.testing.assert_close(src, torch.ones_like(src))

#     # test get_config_fn
#     src = torch.zeros(N, device=device)
#     get_config_fn = lambda META: {'BLOCK_SIZE': 32, 'num_warps': 4, 'num_ctas': 1, 'num_stages': 2} if META['N'] == N else {}
#     fn_infer = triton.autotune_configured(get_config_fn=get_config_fn)(_kernel_hcutune_configured)
#     fn_infer[grid](src, N)
#     triton.testing.assert_close(src, torch.ones_like(src))

#     # test get_key_fn
#     def get_key_fn(data, META):
#         assert len(data.keys()) == 1
#         key = list(data.keys())[0]
#         return key

#     src = torch.zeros(N, device=device)
#     fn_infer = triton.autotune_configured(get_key_fn=get_key_fn)(_kernel_hcutune_configured)
#     fn_infer[grid](src, 2 * N)
#     triton.testing.assert_close(src, torch.ones_like(src))


# def test_hcutune_configured_tuning(device: str = "cuda"):
#     N = 1024
#     src = torch.zeros(N, device=device)
#     configs = [triton.Config(kwargs={'BLOCK_SIZE': 32, 'BRANCH': 1}), triton.Config(kwargs={'BLOCK_SIZE': 128, 'BRANCH': 1})]

#     @triton.autotune(configs=configs, key=['N'], restore_value=['src'], do_bench=do_bench)
#     @triton.autotune_configured(get_config_fn=lambda META: {'BLOCK_SIZE': 32, 'BRANCH': 0, 'num_warps': 4, 'num_ctas': 1, 'num_stages': 2} if META['N'] == N else {})
#     @triton.jit
#     def _kernel_hcutune_configured_tuning(src, N, BLOCK_SIZE: tl.constexpr, BRANCH: tl.constexpr):
#         offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
#         x = tl.load(src + offsets, mask=offsets < N) + 1
#         if BRANCH:
#             tl.store(src + offsets, x, mask=offsets < N)

#     grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']), )
#     _kernel_hcutune_configured_tuning[grid](src, N)
#     triton.testing.assert_close(src, torch.ones_like(src))


# def test_hcutune_configured_warmup(monkeypatch, device: str = "cuda"):
#     monkeypatch.setenv("TRITON_HCUTUNE_COMPILE_ONLY", "1")

#     N = 1024
#     src = torch.zeros(N, device=device)
#     configs = [triton.Config(kwargs={'BLOCK_SIZE': 32, 'BRANCH': 1}), triton.Config(kwargs={'BLOCK_SIZE': 128, 'BRANCH': 1})]

#     @triton.autotune(configs=configs, key=['N'], restore_value=['src'], do_bench=do_bench)
#     @triton.autotune_configured(get_config_fn=lambda META: {'BLOCK_SIZE': 32, 'BRANCH': 0, 'num_warps': 4, 'num_ctas': 1, 'num_stages': 2} if META['N'] == N else {})
#     @triton.jit
#     def _kernel_hcutune_configured_tuning(src, N, BLOCK_SIZE: tl.constexpr, BRANCH: tl.constexpr):
#         offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
#         x = tl.load(src + offsets, mask=offsets < N) + 1
#         if BRANCH:
#             tl.store(src + offsets, x, mask=offsets < N)

#     grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']), )
#     _kernel_hcutune_configured_tuning[grid](src, N)


# @pytest.mark.parametrize('perf_mode', ['1', '0', '1'])
# def test_hcutune_perf(monkeypatch, perf_mode, device: str = "cuda"):
#     if perf_mode == '1':
#         global_config_loader.load_all()
#     monkeypatch.setenv("TRITON_HCUTUNE_PERF_MODE", perf_mode)

#     N = 1024
#     src = torch.zeros(N, device=device)
#     configs = [triton.Config(kwargs={'BLOCK_SIZE': 32}), triton.Config(kwargs={'BLOCK_SIZE': 128})]

#     @triton.autotune(configs=configs, key=['N'], restore_value=['src'], do_bench=do_bench)
#     @triton.jit
#     def _kernel_hcutune_perf(src, N, BLOCK_SIZE: tl.constexpr):
#         offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
#         x = tl.load(src + offsets, mask=offsets < N) + 1
#         tl.store(src + offsets, x, mask=offsets < N)

#     grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']), )
#     _kernel_hcutune_perf[grid](src, N)
#     triton.testing.assert_close(src, torch.ones_like(src))


# def test_run_saved_kernel(device: str = "cuda"):
#     N = 1024
#     src = torch.zeros(N, device=device)
#     configs = [triton.Config(kwargs={'BLOCK_SIZE': 32}), triton.Config(kwargs={'BLOCK_SIZE': 128})]

#     @triton.autotune(configs=configs, key=['N'], restore_value=['src'], do_bench=do_bench)
#     @triton.jit
#     def _kernel_run_saved_kernel(src, N, BLOCK_SIZE: tl.constexpr):
#         offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
#         x = tl.load(src + offsets, mask=offsets < N) + 1
#         tl.store(src + offsets, x, mask=offsets < N)

#     grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']), )
#     _kernel_run_saved_kernel[grid](src, N)

#     # run with run_saved_kernel()
#     fn = _kernel_run_saved_kernel.fn
#     src = torch.zeros(N, device=device)

#     _, key = get_config_key(_kernel_run_saved_kernel.arg_names,
#                             _kernel_run_saved_kernel.keys, src, N)
#     key = str(key)

#     global_config_loader.load_all()
#     configs, paths = get_config_cache(fn.__name__)
#     config = configs[key]
#     path = paths[key]
#     triton.utils.run_saved_kernel(fn, path, src, N, grid=grid, **config)
#     triton.testing.assert_close(src, torch.ones_like(src))


# def test_run_saved_kernel_heuristics(device: str = "cuda"):
#     N = 256
#     src = torch.zeros(N, device=device)
#     configs = [triton.Config(kwargs={'BLOCK_SIZE': 32}), triton.Config(kwargs={'BLOCK_SIZE': 128})]

#     @triton.autotune(configs=configs, key=['N'], restore_value=['src'], do_bench=do_bench)
#     @triton.heuristics(values={
#         '_N': lambda META: META['N'],
#     })
#     @triton.jit
#     def _kernel_run_saved_kernel_heuristics(src, N, BLOCK_SIZE: tl.constexpr, _N: tl.constexpr):
#         offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
#         x = tl.load(src + offsets, mask=offsets < _N) + 1
#         tl.store(src + offsets, x, mask=offsets < _N)

#     grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']), )
#     _kernel_run_saved_kernel_heuristics[grid](src, N)

#     # run with run_saved_kernel()
#     fn = _kernel_run_saved_kernel_heuristics.fn
#     src = torch.zeros(N, device=device)

#     _, key = get_config_key(_kernel_run_saved_kernel_heuristics.arg_names,
#                             _kernel_run_saved_kernel_heuristics.keys, src, N)
#     key = str(key)

#     global_config_loader.load_all()
#     configs, paths = get_config_cache(fn.fn.__name__)
#     config = configs[key]
#     path = paths[key]
#     triton.utils.run_saved_kernel(fn, path, src, N, grid=grid, **config)
#     triton.testing.assert_close(src, torch.ones_like(src))


# def test_kernel_cache_manager(monkeypatch):
#     tmp_dir = tempfile.TemporaryDirectory()
#     cache_dir = tmp_dir.name
#     file_path = f"{cache_dir}/MIKTLFSV5U26DS2XBHMKYZXG4HSDEDO5ZOUOJQGRIPHRQHGK3ZVQ/__grp__test.json"
#     os.makedirs(os.path.dirname(file_path), exist_ok=True)
#     data = {"child_paths": { "test.ttir": "/some/path/in/genEnv/MIKTLFSV5U26DS2XBHMKYZXG4HSDEDO5ZOUOJQGRIPHRQHGK3ZVQ/test.ttir", }}
#     with open(file_path, "w") as f:
#         json.dump(data, f)
#     ttir_path = f"{os.path.dirname(file_path)}/test.ttir"
#     open(ttir_path, "w").write("empty")

#     monkeypatch.setenv("TRITON_CACHE_DIR", cache_dir)
#     km = KernelCacheManager('MIKTLFSV5U26DS2XBHMKYZXG4HSDEDO5ZOUOJQGRIPHRQHGK3ZVQ')
#     metadata_group = km.get_group("test.json") or {}
#     metadata_path = metadata_group.get("test.ttir")
#     assert metadata_path is not None
#     tmp_dir.cleanup()


def test_graphtune(device: str = "cuda"):

    configs = [triton.Config(kwargs={}, num_warps=4), triton.Config(kwargs={}, num_warps=8)]

    @triton.autotune(configs=configs, key=['n_elements', 'BLOCK_SIZE'], restore_value=['out_ptr'], do_bench=do_bench)
    @triton.jit
    def _add_kernel(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        y = tl.load(in_ptr1 + offsets, mask=mask)
        output = x + y
        tl.store(out_ptr + offsets, output, mask=mask)

    @triton.autotune(configs=configs, key=['n_elements', 'BLOCK_SIZE'], restore_value=['out_ptr'], do_bench=do_bench)
    @triton.jit
    def _sub_kernel(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        y = tl.load(in_ptr1 + offsets, mask=mask)
        output = x - y
        tl.store(out_ptr + offsets, output, mask=mask)

    @graphtune
    def add_sub(x, y, block_size):
        out = torch.zeros_like(x)
        n = x.numel()
        grid = lambda META: (triton.cdiv(n, META['BLOCK_SIZE']), )
        _add_kernel[grid](x, y, out, n, block_size)
        _sub_kernel[grid](out, y, out, n, block_size)
        return out

    n = 64
    x = torch.randn(n, device=device)
    y = torch.randn(n, device=device)
    for block_size in [16, 32]:
        res = add_sub(x, y, block_size)
    triton.testing.assert_close(x, res)

    config_fpaths = get_graph_config_files("add_sub", device)
    for config_fpath in config_fpaths:
        assert os.path.exists(config_fpath)
        with open(config_fpath, "r") as fp:
            data = json.load(fp)
        assert len(data) == 2 # num_graph
        assert len(list(data.values())[0].keys()) == 2 # num_kernel
        _data = list(list(data.values())[0].values())[0]
        for k in ["key", "config", "timings", "path"]:
            assert k in _data


def test_split_vals(monkeypatch):
    # not enough: num_vals = 3, world_size = 4
    vals = list(range(3))
    monkeypatch.setenv("TRITON_HCUTUNE_WORLD_SIZE", "4")
    monkeypatch.setenv("TRITON_HCUTUNE_LOCAL_RANK", "0")
    res = split_x_vals(vals)
    assert res == [0]
    monkeypatch.setenv("TRITON_HCUTUNE_LOCAL_RANK", "3")
    res = split_x_vals(vals)
    assert res == []

    # with remainder: num_vals = 5, world_size = 4
    vals = list(range(5))
    monkeypatch.setenv("TRITON_HCUTUNE_WORLD_SIZE", "4")
    monkeypatch.setenv("TRITON_HCUTUNE_LOCAL_RANK", "0")
    res = split_x_vals(vals)
    assert res == [0, 1]
    monkeypatch.setenv("TRITON_HCUTUNE_LOCAL_RANK", "1")
    res = split_x_vals(vals)
    assert res == [2]
    monkeypatch.setenv("TRITON_HCUTUNE_LOCAL_RANK", "2")
    res = split_x_vals(vals)
    assert res == [3]
    monkeypatch.setenv("TRITON_HCUTUNE_LOCAL_RANK", "3")
    res = split_x_vals(vals)
    assert res == [4]

    # alignment: num_vals = 8, world_size = 4
    vals = list(range(8))
    monkeypatch.setenv("TRITON_HCUTUNE_WORLD_SIZE", "4")
    monkeypatch.setenv("TRITON_HCUTUNE_LOCAL_RANK", "0")
    res = split_x_vals(vals)
    assert res == [0, 1]
    monkeypatch.setenv("TRITON_HCUTUNE_LOCAL_RANK", "3")
    res = split_x_vals(vals)
    assert res == [6, 7]


def test_set_devices(monkeypatch):
    # not enough: num_vals = 3, world_size = 4
    vals = list(range(3))
    monkeypatch.setenv("TRITON_HCUTUNE_WORLD_SIZE", "4")
    monkeypatch.setenv("TRITON_HCUTUNE_VISIBLE_DEVICES", "0")
    for i in range(4):
        monkeypatch.setenv("TRITON_HCUTUNE_LOCAL_RANK", str(i))
        set_device()
        assert os.getenv("CUDA_VISIBLE_DEVICES") == "0"

    # with remainder: num_vals = 5, world_size = 4
    vals = list(range(5))
    monkeypatch.setenv("TRITON_HCUTUNE_WORLD_SIZE", "4")
    monkeypatch.setenv("TRITON_HCUTUNE_VISIBLE_DEVICES", "0,1")
    for i in range(4):
        monkeypatch.setenv("TRITON_HCUTUNE_LOCAL_RANK", str(i))
        set_device()
        if i % 2 == 0:
            assert os.getenv("CUDA_VISIBLE_DEVICES") == "0"
        else:
            assert os.getenv("CUDA_VISIBLE_DEVICES") == "1"

    # alignment: num_vals = 8, world_size = 4
    vals = list(range(8))
    monkeypatch.setenv("TRITON_HCUTUNE_WORLD_SIZE", "4")
    monkeypatch.setenv("TRITON_HCUTUNE_VISIBLE_DEVICES", "3,2,1,0")
    monkeypatch.setenv("TRITON_HCUTUNE_LOCAL_RANK", "0")
    set_device()
    assert os.getenv("CUDA_VISIBLE_DEVICES") == "3"
    monkeypatch.setenv("TRITON_HCUTUNE_LOCAL_RANK", "1")
    set_device()
    assert os.getenv("CUDA_VISIBLE_DEVICES") == "2"
    monkeypatch.setenv("TRITON_HCUTUNE_LOCAL_RANK", "2")
    set_device()
    assert os.getenv("CUDA_VISIBLE_DEVICES") == "1"
    monkeypatch.setenv("TRITON_HCUTUNE_LOCAL_RANK", "3")
    set_device()
    assert os.getenv("CUDA_VISIBLE_DEVICES") == "0"


def test_prune_config_loader():
    # not enough: num_vals = 3, world_size = 4
    vals = list(range(3))
    res = list(PruneConfigLoader(vals, 4, 0))
    assert res == [0]
    res = list(PruneConfigLoader(vals, 4, 1))
    assert res == [1]
    res = list(PruneConfigLoader(vals, 4, 2))
    assert res == [2]
    res = list(PruneConfigLoader(vals, 4, 3))
    assert res == []

    # with remainder: num_vals = 5, world_size = 4
    vals = list(range(5))
    res = list(PruneConfigLoader(vals, 4, 0))
    assert res == [0, 1]
    res = list(PruneConfigLoader(vals, 4, 1))
    assert res == [2]
    res = list(PruneConfigLoader(vals, 4, 2))
    assert res == [3]
    res = list(PruneConfigLoader(vals, 4, 3))
    assert res == [4]

    # alignment: num_vals = 8, world_size = 4
    vals = list(range(8))
    res = list(PruneConfigLoader(vals, 4, 0))
    assert res == [0, 1]
    res = list(PruneConfigLoader(vals, 4, 1))
    assert res == [2, 3]
    res = list(PruneConfigLoader(vals, 4, 2))
    assert res == [4, 5]
    res = list(PruneConfigLoader(vals, 4, 3))
    assert res == [6, 7]


code_configs_sharding = """
import os
import sys
import torch
os.environ["TRITON_HCUTUNE"] = "1"
import triton
import triton.language as tl

def do_bench(kernel_call, quantiles):
    return triton.testing.do_bench(kernel_call, quantiles=quantiles, warmup=1, rep=1)

configs = [triton.Config(kwargs={'BLOCK_SIZE_M': 32}), triton.Config(kwargs={'BLOCK_SIZE_M': 128})]

@triton.autotune(configs=configs, key=['M'], warmup=1, rep=1, do_bench=do_bench)
@triton.jit
def _kernel_configs_sharding(dst, src, stride_m: tl.constexpr, M, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr):
    offsets_m = tl.program_id(0) * stride_m + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = tl.arange(0, BLOCK_SIZE_N)
    x = tl.load(src + offsets_m[:, None] * BLOCK_SIZE_N + offsets_n[None, :])
    tl.store(dst + offsets_m[:, None] * BLOCK_SIZE_N + offsets_n[None, :], x)

device = 'cuda'

M = [256, 512, 1024, 1536, 2048, 3072, 4096]
N = 16
CASES = [(m, N) for m in M]

configs = [
    triton.testing.Benchmark(
        x_names=['M', 'N'],
        x_vals=CASES,
        line_arg='provider',
        line_vals=['triton'],
        line_names=['Triton'],
        styles=[('red', '-')],
        ylabel='TOPS',
        xlabel='M',
        plot_name="test",
        args={'device': 'cuda'},
    )
]

def dist_launch(dst, src, M, N):
     grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE_M']), )
     _kernel_configs_sharding[grid](dst, src, N, M, N)


@triton.testing.perf_report(configs)
def benchmark(M, N, provider, device='cuda'):
    def _bench():
        return dist_launch(dst, src, M, N)

    src = torch.randn(M * N, device=device)
    dst = torch.empty(M * N, device=device)
    triton.testing.do_bench(_bench)


def main():
    benchmark.run()


if __name__ == "__main__":
    main()
    """

code_x_vals_sharding = """
import os
import sys
import torch
os.environ["TRITON_HCUTUNE"] = "1"
import triton
import triton.language as tl

def do_bench(kernel_call, quantiles):
    return triton.testing.do_bench(kernel_call, quantiles=quantiles, warmup=1, rep=1)

configs = [triton.Config(kwargs={'BLOCK_SIZE_M': 32}), triton.Config(kwargs={'BLOCK_SIZE_M': 128})]

@triton.autotune(configs=configs, key=['M'], warmup=1, rep=1, do_bench=do_bench)
@triton.jit
def _kernel_x_vals_sharding(dst, src, stride_m: tl.constexpr, M, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr):
    offsets_m = tl.program_id(0) * stride_m + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = tl.arange(0, BLOCK_SIZE_N)
    x = tl.load(src + offsets_m[:, None] * BLOCK_SIZE_N + offsets_n[None, :])
    tl.store(dst + offsets_m[:, None] * BLOCK_SIZE_N + offsets_n[None, :], x)

device = 'cuda'

M = [256, 512, 1024, 1536, 2048, 3072, 4096]
N = 16
CASES = [(m, N) for m in M]

configs = [
    triton.testing.Benchmark(
        x_names=['M', 'N'],
        x_vals=CASES,
        line_arg='provider',
        line_vals=['triton'],
        line_names=['Triton'],
        styles=[('red', '-')],
        ylabel='TOPS',
        xlabel='M',
        plot_name="test",
        args={'device': 'cuda'},
    )
]

def dist_launch(dst, src, M, N):
     grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE_M']), )
     _kernel_x_vals_sharding[grid](dst, src, N, M, N)


@triton.testing.perf_report(configs)
def benchmark(M, N, provider, device='cuda'):
    def _bench():
        return dist_launch(dst, src, M, N)

    src = torch.randn(M * N, device=device)
    dst = torch.empty(M * N, device=device)
    triton.testing.do_bench(_bench)


def main():
    benchmark.run()


if __name__ == "__main__":
    main()
    """

@pytest.mark.parametrize('world_size', [4])
@pytest.mark.parametrize('sharding', [""]) # configs sharding, x_vals/cases sharding
def test_dist_launch(monkeypatch, world_size, sharding):
    monkeypatch.setenv("TRITON_CACHE_DIR", tempfile.mkdtemp())
    # multi-process tuning

    if sharding == "--configs-sharding":
        filename = "_kernel_configs_sharding" 
        code = code_configs_sharding
    else:
        filename = "_kernel_x_vals_sharding"
        code = code_x_vals_sharding

    temp_filename = None
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code.strip())
        temp_filename = f.name

    device = 'cuda'

    cmd = ["python3", "-m", "triton.backends.hcu.hcutune_cli", "--nproc", str(world_size),
           "--devices", "0,0,0,0"]
    if sharding != "":
        cmd.append(sharding)
    cmd.append(temp_filename)
    subprocess.run(cmd, stdout=subprocess.PIPE)

    # check dist result
    def do_bench(kernel_call, quantiles):
        return triton.testing.do_bench(kernel_call, quantiles=quantiles, warmup=1, rep=1)

    configs = [triton.Config(kwargs={'BLOCK_SIZE_M': 32}), triton.Config(kwargs={'BLOCK_SIZE_M': 128})]

    @triton.autotune(configs=configs, key=['M'], warmup=1, rep=1, do_bench=do_bench)
    @triton.jit
    def _kernel_configs_sharding(dst, src, stride_m: tl.constexpr, M, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr):
        offsets_m = tl.program_id(0) * stride_m + tl.arange(0, BLOCK_SIZE_M)
        offsets_n = tl.arange(0, BLOCK_SIZE_N)
        x = tl.load(src + offsets_m[:, None] * BLOCK_SIZE_N + offsets_n[None, :])
        tl.store(dst + offsets_m[:, None] * BLOCK_SIZE_N + offsets_n[None, :], x)

    @triton.autotune(configs=configs, key=['M'], warmup=1, rep=1, do_bench=do_bench)
    @triton.jit
    def _kernel_x_vals_sharding(dst, src, stride_m: tl.constexpr, M, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr):
        offsets_m = tl.program_id(0) * stride_m + tl.arange(0, BLOCK_SIZE_M)
        offsets_n = tl.arange(0, BLOCK_SIZE_N)
        x = tl.load(src + offsets_m[:, None] * BLOCK_SIZE_N + offsets_n[None, :])
        tl.store(dst + offsets_m[:, None] * BLOCK_SIZE_N + offsets_n[None, :], x)

    M, N = 256, 16
    src = torch.randn(M * N, device=device)
    dst = torch.empty(M * N, device=device)

    if filename == "_kernel_configs_sharding":
        files = get_config_cache_files(_kernel_configs_sharding, dst, src, N, M, N)
        assert len(files) == len(configs)
    else:
        files = get_config_cache_files(_kernel_x_vals_sharding, dst, src, N, M, N)
        assert len(files) == world_size

    config_loader = ConfigLoader()
    config_cache = config_loader.get_tuned_cache(filename, get_gpu_label()).cache
    assert config_cache['key'] == ['M', 'dst', 'src']
    # M = [256, 512, 1024, 1536, 2048, 3072, 4096], len(M) == 7
    assert len(config_cache['configs']) == len(config_cache['timings']) == 7


def test_min_timings_config():
    cache1 = {
        "key": ["N", "M"],
        "configs": {
            "(16, 32)": {"BLOCK_SIZE_M": 32},
        },
        "timings": {
            "(16, 32)": [0.00012, 0.00008, 0.00013]
        },
        "paths": {
            "(16, 32)": "EX3NUTH7J2B5ED47YFJMCOZKIILU3KO4N62FCCLNX2O2EOM7YJNQ",
        }
    }

    cache2 = {
        "key": ["N", "M"],
        "configs": {
            "(16, 32)": {"BLOCK_SIZE_M": 64},
        },
        "timings": {
            "(16, 32)": [0.00010, 0.00018, 0.00012]
        },
        "paths": {
            "(16, 32)": "FIKMNTKZ54W2RDVJXGK5KM73U7GOGBVFHRMGWMN3BLO7JCHMZQQQ",
        }
    }

    cache3 = {
        "key": ["N", "M"],
        "configs": {
            "(16, 32)": {"BLOCK_SIZE_M": 128},
        },
        "timings": {
            "(16, 32)": [0.00013, 0.00028, 0.00013]
        },
        "paths": {
            "(16, 32)": "EX3NUTH7J2B5ED47YFJMCOZKIILU3KO4N62FCCLNX2O2EOM7YJNQ",
        }
    }

    cache = merge_caches([cache1, cache3, cache2])
    assert len(cache['configs']) == len(cache['timings']) == len(cache['paths']) == 1
    assert cache['configs']['(16, 32)'] == {"BLOCK_SIZE_M": 64}
    assert cache['timings']['(16, 32)'] == [0.00010, 0.00018, 0.00012]
    assert cache['paths']['(16, 32)'] == "FIKMNTKZ54W2RDVJXGK5KM73U7GOGBVFHRMGWMN3BLO7JCHMZQQQ"


code_heuristics = """
import torch
import os
os.environ["TRITON_HCUTUNE"] = "1"
import triton
import triton.language as tl

device = 'cuda'
N = 1024
src = torch.zeros(N, device=device)

def do_bench(kernel_call, quantiles):
    return triton.testing.do_bench(kernel_call, quantiles=quantiles, warmup=1, rep=1)

configs = [triton.Config(kwargs={'BLOCK_SIZE': 32}), triton.Config(kwargs={'BLOCK_SIZE': 128})]

@triton.autotune(configs=configs, key=['N', 'DUMMY'], restore_value=['src'], do_bench=do_bench)
@triton.heuristics(values={
    "N_DIVISIBLE_BY_16": lambda args: args["N"] % 16 == 0,
})
@triton.jit
def _kernel(src, N, N_DIVISIBLE_BY_16: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(src + offsets, mask=offsets < N) + 1
    tl.store(src + offsets, x, mask=offsets < N)
"""

@pytest.mark.parametrize('code', [code_x_vals_sharding, code_heuristics])
def test_compile_only(code: str):
    temp_filename = None
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code.strip())
        temp_filename = f.name

    cmd = ["python3", "-m", "triton.backends.hcu.hcutune_cli", "--nproc",
            "2", "--compile-only", temp_filename]
    res = subprocess.run(cmd, stdout=subprocess.PIPE)
    assert res.returncode == 0


@pytest.mark.parametrize('world_size', [4])
def test_dist_graphtune(monkeypatch, world_size, device: str = "cuda"):
    monkeypatch.setenv("TRITON_CACHE_DIR", tempfile.mkdtemp())
    code = '''
import torch
import os
os.environ["TRITON_HCUTUNE"] = "1"
import triton
import triton.language as tl

def do_bench(kernel_call, quantiles):
    return triton.testing.do_bench(kernel_call, quantiles=quantiles, warmup=1, rep=1)

configs = [triton.Config(kwargs={}, num_warps=4), triton.Config(kwargs={}, num_warps=8)]

@triton.autotune(configs=configs, key=['n_elements', 'BLOCK_SIZE'], restore_value=['out_ptr'], do_bench=do_bench)
@triton.jit
def _dist_add_kernel(
    in_ptr0,
    in_ptr1,
    out_ptr,
    n_elements,
    BLOCK_SIZE: "tl.constexpr",
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    y = tl.load(in_ptr1 + offsets, mask=mask)
    output = x + y
    tl.store(out_ptr + offsets, output, mask=mask)

@triton.autotune(configs=configs, key=['n_elements', 'BLOCK_SIZE'], restore_value=['out_ptr'], do_bench=do_bench)
@triton.jit
def _dist_sub_kernel(
    in_ptr0,
    in_ptr1,
    out_ptr,
    n_elements,
    BLOCK_SIZE: "tl.constexpr",
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    y = tl.load(in_ptr1 + offsets, mask=mask)
    output = x - y
    tl.store(out_ptr + offsets, output, mask=mask)

from triton.backends.hcu.autotuner import graphtune
@graphtune
def dist_add_sub(x, y, out, n, block_size):
    grid = lambda META: (triton.cdiv(n, META['BLOCK_SIZE']), )
    _dist_add_kernel[grid](x, y, out, n, block_size)
    _dist_sub_kernel[grid](out, y, out, n, block_size)

N = [64, 128, 256, 512]
BLOCK_SIZE = [16, 32, 64]
CASES = [(n, block_size) for n in N for block_size in BLOCK_SIZE]

configs = [
    triton.testing.Benchmark(
        x_names=['N', 'BLOCK_SIZE'],
        x_vals=CASES,
        line_arg='provider',
        line_vals=['triton'],
        line_names=['Triton'],
        styles=[('red', '-')],
        ylabel='TOPS',
        xlabel='M',
        plot_name="test",
        args={'device': 'cuda'},
    )
]

@triton.testing.perf_report(configs)
def benchmark(N, BLOCK_SIZE, provider, device='cuda'):
    def _bench():
        return dist_add_sub(x, y, out, N, BLOCK_SIZE)

    x = torch.randn(N, device=device)
    y = torch.randn(N, device=device)
    out = torch.zeros_like(x)
    triton.testing.do_bench(_bench)

def main():
    benchmark.run()

if __name__ == "__main__":
    main()
    '''

    temp_filename = None
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code.strip())
        temp_filename = f.name

    cmd = ["hcutune_cli", "--nproc", str(world_size), "--force"]
    cmd.append(temp_filename)
    res = subprocess.run(cmd, stdout=subprocess.PIPE)
    assert res.returncode == 0

    conf_loader = GraphConfigLoader()
    _cache = conf_loader.cache[("dist_add_sub", get_device_name(device))]
    assert len(_cache) == 48
    _graph = _cache[-1]
    assert len(_graph) == 2
    _config = list(_graph.values())[0]
    for k in ["key", "config", "timings", "path"]:
        assert k in _config
