import os
import json
import tempfile

import triton.language as tl
import subprocess
from triton.utils.hcutuner import get_gpu_label
from pathlib import Path


def count_files(path):
    dir_path = Path(path)
    return sum(1 for f in dir_path.iterdir() if f.is_file())


code = """
import os
import sys
import torch
import triton
import triton.language as tl

def do_bench(kernel_call, quantiles):
    return triton.testing.do_bench(kernel_call, quantiles=quantiles, warmup=1, rep=1)

configs = [triton.Config(kwargs={'BLOCK_SIZE_M': 32}), triton.Config(kwargs={'BLOCK_SIZE_M': 128})]

@triton.utils.hcutune(configs=configs, key=['M', 'N', 'K'], warmup=1, rep=1, do_bench=do_bench)
@triton.jit
def _kernel_opt_config_cli(src, M, N, K, BLOCK_SIZE_M: tl.constexpr):
    # dummy
    offsets_m = tl.program_id(0) * M * N * K * BLOCK_SIZE_M
    x = tl.load(src + offsets_m)

device = 'cuda'

SIZE = 2048
M = [256, 512, 1024, 1536]
N = [16, 32, 16, 8]
K = [512, 1024, 512, 2048]

src = torch.zeros(SIZE, device=device)
for m, n, k in zip(M, N, K):
    _kernel_opt_config_cli[(1,)](src, m, n, k)
    """

def test_show():
    temp_filename = None
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code.strip())
        temp_filename = f.name

    cmd = ["python", temp_filename]
    res = subprocess.run(cmd, stdout=subprocess.PIPE)
    assert res.returncode == 0

    cmd = ["opt_config_cli", "show"]
    res = subprocess.run(cmd, stdout=subprocess.PIPE)
    assert res.returncode == 0


def test_export():
    with tempfile.TemporaryDirectory() as tmp_dir:
        cmd = ["opt_config_cli", "export", "--kernel", "_kernel_opt_config_cli",
            "--device", get_gpu_label(), "--output", "test.json",
            "--output_dir", tmp_dir]
        res = subprocess.run(cmd, stdout=subprocess.PIPE)
        assert res.returncode == 0

        assert count_files(tmp_dir) == 1

        with open(f"{tmp_dir}/test.json") as f:
            data = json.load(f)
        list(data.keys())[0] == '(256, 16, 512, torch.float32)'


def test_export_hoist():
    with tempfile.TemporaryDirectory() as tmp_dir:
        cmd = ["opt_config_cli", "export", "--kernel", "_kernel_opt_config_cli",
            "--device", get_gpu_label(), "--output", "test-%G-fp32.json",
            "--output_dir", tmp_dir, "--hoist_key", "N,K", "--delimiter", ","]
        res = subprocess.run(cmd, stdout=subprocess.PIPE)
        assert res.returncode == 0

        files = os.listdir(tmp_dir)
        assert len(files) == 3
        for f in files:
            assert f in ["test-N=16,K=512-fp32.json",
                         "test-N=32,K=1024-fp32.json",
                         "test-N=8,K=2048-fp32.json",]

        with open(f"{tmp_dir}/test-N=16,K=512-fp32.json") as f:
            data = json.load(f)
        assert len(data) == 2
        assert list(data.keys())[0] == "(256, 'torch.float32')"
