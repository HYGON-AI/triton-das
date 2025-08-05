import os
import json
import torch
import tempfile

import triton
import triton.language as tl
import pytest
from triton.knobs import cache as cache_knob
import subprocess
from triton.utils.hcutuner import get_config_key, get_gpu_label


def do_bench(kernel_call, quantiles):
    return triton.testing.do_bench(kernel_call, quantiles=quantiles, warmup=1, rep=1)


def get_save_config_root_dir():
    return f"{cache_knob.dir}/configs"


def get_gpu_label():
    gpu_name = torch.cuda.get_device_name().replace(" ", "_").replace("/", "_")
    return gpu_name


def get_device_name(device: str):
    if device == "cuda":
        return get_gpu_label()
    else:
        raise NotImplementedError


def get_save_config_files(fname, device):
    root_dir = get_save_config_root_dir()
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
    cache_manager = fn.get_config_cache_manager(*args, **kwargs)
    for dirpath, _, filenames in os.walk(cache_manager.key_dir):
        for filename in filenames:
            if filename != "config.json":
                continue
            res.append(os.path.join(dirpath, filename))
    return res


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
    config_fpaths = get_save_config_files("_kernel", device)
    for config_fpath in config_fpaths:
        assert os.path.exists(config_fpath)
        with open(config_fpath, "r") as fp:
            data = json.load(fp)
        for k in ["key", "configs", "timings"]:
            assert k in data
        assert len(data['configs']) == len(data['timings']) == 1
        assert list(data['configs'].keys())[0] == "1024"
        assert list(data['timings'].keys())[0] == "1024"
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

    triton.utils.global_config_loader.load_all()
    key_dict = {
        'N': N,
    }
    config = triton.utils.get_optimal_config("_kernel", key_dict)
    _kernel[grid](*args, **kwargs, **config)
    triton.testing.assert_close(src, torch.ones_like(src))
    config2 = triton.utils.get_optimal_config("_kernel", [N])
    config3 = triton.utils.get_optimal_config("_kernel", str(N))
    assert config == config2 == config3


@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('world_size', [4])
def test_mp_save_config(device, world_size):
    code = """
import os
import sys
import torch
import triton
import triton.language as tl

def do_bench(kernel_call, quantiles):
    return triton.testing.do_bench(kernel_call, quantiles=quantiles, warmup=1, rep=1)

N = int(sys.argv[1])
device = sys.argv[2]

configs = [triton.Config(kwargs={'BLOCK_SIZE': 32}), triton.Config(kwargs={'BLOCK_SIZE': 128})]

@triton.utils.hcutune(configs=configs, key=['N'], restore_value=['src'], do_bench=do_bench)
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

    triton.utils.global_config_loader.load_all()
    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']), )
    key_dict = {
        'N': N,
    }
    config = triton.utils.get_optimal_config("_kernel_mp", key_dict)
    _kernel_mp[grid](src, N, **config)
    triton.testing.assert_close(src, torch.ones_like(src))


@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('always_tuning', [True, False])
def test_restore_cache(device: str, always_tuning: bool):
    configs = [triton.Config(kwargs={'BLOCK_SIZE': 32}), triton.Config(kwargs={'BLOCK_SIZE': 128})]
    @triton.utils.hcutune(configs=configs, key=['M', 'N'], restore_value=['src'], do_bench=do_bench,
                          always_tuning=always_tuning)
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
    fn.restore_tuned_cache(src[0], M[0], N[0])
    assert len(fn.cache) == len(fn.configs_timings) == SIZE
    for _, v in fn.cache.items():
        assert v in fn.configs_timings

    # cache hit
    keys = [str(o) for o in fn.cache.keys()]
    for _src, m, n in zip(src, M, N):
        _, key = get_config_key(fn.arg_names, fn.keys, _src, m, n)
        assert str(key) in keys


@pytest.mark.parametrize('device', ['cuda'])
def test_cache_key_changes(device: str):
    M, N = 128, 128
    src = torch.zeros(N, device=device)

    # change configs: {'BLOCK_SIZE': 32} -> {'BLOCK_SIZE': 64}
    configs = [triton.Config(kwargs={'BLOCK_SIZE': 64}), triton.Config(kwargs={'BLOCK_SIZE': 128})]
    @triton.utils.hcutune(configs=configs, key=['M', 'N'], restore_value=['src'], do_bench=do_bench)
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
    @triton.utils.hcutune(configs=configs, key=['M', 'N'], restore_value=['src'], do_bench=do_bench,
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
    @triton.utils.hcutune(configs=configs, key=['N', 'M'], restore_value=['src'], do_bench=do_bench)
    @triton.jit
    def _kernel_cache(src, M, N, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        N = N * M // M # dummy M
        x = tl.load(src + offsets, mask=offsets < N) + 1
        tl.store(src + offsets, x, mask=offsets < N)

    assert len(get_config_cache_files(_kernel_cache, src, M, N)) == 0

    # change code
    configs = [triton.Config(kwargs={'BLOCK_SIZE': 32}), triton.Config(kwargs={'BLOCK_SIZE': 128})]
    @triton.utils.hcutune(configs=configs, key=['M', 'N'], restore_value=['src'], do_bench=do_bench)
    @triton.jit
    def _kernel_cache(src, M, N, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offsets, mask=offsets < N) + 1
        tl.store(src + offsets, x, mask=offsets < N)

    assert len(get_config_cache_files(_kernel_cache, src, M, N)) == 0


@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('world_size', [4])
def test_mp_cache_config(device: str, world_size):
    code = """
import os
import sys
import torch
import triton
import triton.language as tl

def do_bench(kernel_call, quantiles):
    return triton.testing.do_bench(kernel_call, quantiles=quantiles, warmup=1, rep=1)

N = int(sys.argv[1])
device = sys.argv[2]

configs = [triton.Config(kwargs={'BLOCK_SIZE': 32}), triton.Config(kwargs={'BLOCK_SIZE': 128})]

@triton.utils.hcutune(configs=configs, key=['N'], restore_value=['src'], do_bench=do_bench)
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

    @triton.utils.hcutune(configs=configs, key=['N'], restore_value=['src'], do_bench=do_bench)
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

    @triton.utils.hcutune(configs=configs, key=['N', 'DUMMY'], restore_value=['src'], do_bench=do_bench)
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


def test_hcutune_configured(device: str = "cuda"):
    N = 1024
    src = torch.zeros(N, device=device)
    configs = [triton.Config(kwargs={'BLOCK_SIZE': 32}), triton.Config(kwargs={'BLOCK_SIZE': 128})]

    @triton.jit
    def _kernel_hcutune_configured(src, N, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offsets, mask=offsets < N) + 1
        tl.store(src + offsets, x, mask=offsets < N)

    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']), )
    fn = triton.utils.hcutune(configs=configs, key=['N'], restore_value=['src'],
                              do_bench=do_bench)(_kernel_hcutune_configured)
    fn[grid](src, N)
    triton.utils.global_config_loader.load_all()

    fn_infer = triton.utils.hcutune_configured()(_kernel_hcutune_configured)
    src = torch.zeros(N, device=device)
    fn_infer[grid](src, N)
    triton.testing.assert_close(src, torch.ones_like(src))

    # test get_config_fn
    src = torch.zeros(N, device=device)
    get_config_fn = lambda META: {'BLOCK_SIZE': 32, 'num_warps': 4, 'num_ctas': 1, 'num_stages': 2} if META['N'] == N else {}
    fn_infer = triton.utils.hcutune_configured(get_config_fn=get_config_fn)(_kernel_hcutune_configured)
    fn_infer[grid](src, N)
    triton.testing.assert_close(src, torch.ones_like(src))

    # test get_key_fn
    def get_key_fn(data, META):
        total_N = [eval(o) for o in data.keys()]
        n = min(total_N, key=lambda x: abs(x - META['N']))
        return str(n)

    src = torch.zeros(N, device=device)
    fn_infer = triton.utils.hcutune_configured(get_key_fn=get_key_fn)(_kernel_hcutune_configured)
    fn_infer[grid](src, 2 * N)
    triton.testing.assert_close(src, torch.ones_like(src))


def test_hcutune_configured_tuning(device: str = "cuda"):
    N = 1024
    src = torch.zeros(N, device=device)
    configs = [triton.Config(kwargs={'BLOCK_SIZE': 32, 'BRANCH': 1}), triton.Config(kwargs={'BLOCK_SIZE': 128, 'BRANCH': 1})]

    @triton.utils.hcutune(configs=configs, key=['N'], restore_value=['src'], do_bench=do_bench)
    @triton.utils.hcutune_configured(get_config_fn=lambda META: {'BLOCK_SIZE': 32, 'BRANCH': 0, 'num_warps': 4, 'num_ctas': 1, 'num_stages': 2} if META['N'] == N else {})
    @triton.jit
    def _kernel_hcutune_configured_tuning(src, N, BLOCK_SIZE: tl.constexpr, BRANCH: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offsets, mask=offsets < N) + 1
        if BRANCH:
            tl.store(src + offsets, x, mask=offsets < N)

    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']), )
    _kernel_hcutune_configured_tuning[grid](src, N)
    triton.testing.assert_close(src, torch.ones_like(src))


def test_hcutune_configured_warmup(monkeypatch, device: str = "cuda"):
    monkeypatch.setenv("TRITON_HCUTUNE_COMPILE_ONLY", "1")

    N = 1024
    src = torch.zeros(N, device=device)
    configs = [triton.Config(kwargs={'BLOCK_SIZE': 32, 'BRANCH': 1}), triton.Config(kwargs={'BLOCK_SIZE': 128, 'BRANCH': 1})]

    @triton.utils.hcutune(configs=configs, key=['N'], restore_value=['src'], do_bench=do_bench)
    @triton.utils.hcutune_configured(get_config_fn=lambda META: {'BLOCK_SIZE': 32, 'BRANCH': 0, 'num_warps': 4, 'num_ctas': 1, 'num_stages': 2} if META['N'] == N else {})
    @triton.jit
    def _kernel_hcutune_configured_tuning(src, N, BLOCK_SIZE: tl.constexpr, BRANCH: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offsets, mask=offsets < N) + 1
        if BRANCH:
            tl.store(src + offsets, x, mask=offsets < N)

    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']), )
    _kernel_hcutune_configured_tuning[grid](src, N)
