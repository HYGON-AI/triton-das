import os
import torch
import tempfile

import triton
import triton.language as tl
import pytest
import subprocess
from triton.utils.testing import split_x_vals


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


@pytest.mark.parametrize('world_size', [4])
def test_dist_launch(world_size):
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
    assert len(files) == world_size
