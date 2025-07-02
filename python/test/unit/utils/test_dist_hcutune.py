import os
import torch
import tempfile

import triton
import triton.language as tl
import pytest
import subprocess
from triton.utils.testing import split_x_vals, set_device
from triton.utils.hcutuner import PruneConfigLoader, merge_caches, get_gpu_label
from triton.utils import global_config_loader


def get_config_cache_files(fn, *args, **kwargs):
    res = []
    cache_manager = fn.get_config_cache_manager(*args, **kwargs)
    for dirpath, _, filenames in os.walk(cache_manager.key_dir):
        for filename in filenames:
            if filename != "config.json":
                continue
            res.append(os.path.join(dirpath, filename))
    return res


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


code = """
import os
import sys
import torch
import triton
import triton.language as tl

def do_bench(kernel_call, quantiles):
    return triton.testing.do_bench(kernel_call, quantiles=quantiles, warmup=1, rep=1)

configs = [triton.Config(kwargs={'BLOCK_SIZE_M': 32}), triton.Config(kwargs={'BLOCK_SIZE_M': 128})]

@triton.utils.hcutune(configs=configs, key=['M'], warmup=1, rep=1, do_bench=do_bench)
@triton.jit
def _kernel_dist_launch(dst, src, stride_m: tl.constexpr, M, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr):
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
     _kernel_dist_launch[grid](dst, src, N, M, N)


@triton.utils.dist_perf_report(configs)
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
def test_dist_launch(world_size):
    # multi-process tuning

    temp_filename = None
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code.strip())
        temp_filename = f.name

    device = 'cuda'

    cmd = ["python3", "-m", "triton.tools.launch", "--nproc",
            str(world_size), temp_filename]
    subprocess.run(cmd, stdout=subprocess.PIPE)

    # check dist result
    def do_bench(kernel_call, quantiles):
        return triton.testing.do_bench(kernel_call, quantiles=quantiles, warmup=1, rep=1)

    configs = [triton.Config(kwargs={'BLOCK_SIZE_M': 32}), triton.Config(kwargs={'BLOCK_SIZE_M': 128})]

    @triton.utils.hcutune(configs=configs, key=['M'], warmup=1, rep=1, do_bench=do_bench)
    @triton.jit
    def _kernel_dist_launch(dst, src, stride_m: tl.constexpr, M, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr):
        offsets_m = tl.program_id(0) * stride_m + tl.arange(0, BLOCK_SIZE_M)
        offsets_n = tl.arange(0, BLOCK_SIZE_N)
        x = tl.load(src + offsets_m[:, None] * BLOCK_SIZE_N + offsets_n[None, :])
        tl.store(dst + offsets_m[:, None] * BLOCK_SIZE_N + offsets_n[None, :], x)

    M, N = 256, 16
    src = torch.randn(M * N, device=device)
    dst = torch.empty(M * N, device=device)

    files = get_config_cache_files(_kernel_dist_launch, dst, src, N, M, N)
    assert len(files) == len(configs)

    global_config_loader.load_all()
    config_cache = global_config_loader.get_tuned_cache("_kernel_dist_launch", get_gpu_label()).cache
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
        }
    }

    cache2 = {
        "key": ["N", "M"],
        "configs": {
            "(16, 32)": {"BLOCK_SIZE_M": 64},
        },
        "timings": {
            "(16, 32)": [0.00010, 0.00018, 0.00012]
        }
    }

    cache3 = {
        "key": ["N", "M"],
        "configs": {
            "(16, 32)": {"BLOCK_SIZE_M": 128},
        },
        "timings": {
            "(16, 32)": [0.00013, 0.00028, 0.00013]
        }
    }

    cache = merge_caches([cache1, cache3, cache2])
    assert len(cache['configs']) == len(cache['timings']) == 1
    assert cache['configs']['(16, 32)'] == {"BLOCK_SIZE_M": 64}
    assert cache['timings']['(16, 32)'] == [0.00010, 0.00018, 0.00012]


code_heuristics = """
import torch
import triton
import triton.language as tl

device = 'cuda'
N = 1024
src = torch.zeros(N, device=device)

def do_bench(kernel_call, quantiles):
    return triton.testing.do_bench(kernel_call, quantiles=quantiles, warmup=1, rep=1)

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
"""

@pytest.mark.parametrize('code', [code, code_heuristics])
def test_compile_only(code: str):
    temp_filename = None
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code.strip())
        temp_filename = f.name

    cmd = ["python3", "-m", "triton.tools.launch", "--nproc",
            "4", "--compile-only", temp_filename]
    res = subprocess.run(cmd, stdout=subprocess.PIPE)
    assert res.returncode == 0
