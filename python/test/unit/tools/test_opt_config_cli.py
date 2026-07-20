import os
import json
import tempfile
import subprocess
from pathlib import Path

os.environ["TRITON_HCUTUNE"] = "1"
from triton.backends.hcu._utils import get_gpu_label

KERNEL = "_kernel_opt_config_cli"


def count_files(path):
    dir_path = Path(path)
    return sum(1 for f in dir_path.iterdir() if f.is_file())


def export_outdir(tmp_dir):
    return os.path.join(tmp_dir, KERNEL)


code = """
import os
os.environ["TRITON_HCUTUNE"] = "1"
import sys
import torch
import triton
import triton.language as tl

def do_bench(kernel_call, quantiles):
    return triton.testing.do_bench(kernel_call, quantiles=quantiles, warmup=1, rep=1)

configs = [triton.Config(kwargs={'BLOCK_SIZE_M': 32}), triton.Config(kwargs={'BLOCK_SIZE_M': 128})]

@triton.autotune(configs=configs, key=['M', 'N', 'K'], warmup=1, rep=1, do_bench=do_bench)
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
        out = export_outdir(tmp_dir)
        cmd = [
            "opt_config_cli", "export", "--kernel", KERNEL, "--device",
            get_gpu_label(), "--basename", "test.json", "--output-dir", tmp_dir,
            "--output-fields", "key,config,path",
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE)
        assert res.returncode == 0

        assert count_files(out) == 1

        with open(f"{out}/test.json") as f:
            data = json.load(f)
        assert len(data) == 3
        assert list(data['config'].keys()) == list(data['path'].keys())


def test_export_hoist():
    with tempfile.TemporaryDirectory() as tmp_dir:
        out = export_outdir(tmp_dir)
        cmd = [
            "opt_config_cli", "export", "--kernel", KERNEL, "--device",
            get_gpu_label(), "--basename", "test-%G-fp32.json", "--output-dir", tmp_dir,
            "--hoist-key", "N,K", "--delimiter", ",",
            "--output-fields", "key,config,path",
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE)
        assert res.returncode == 0

        files = [o for o in os.listdir(out) if o.endswith(".json")]
        assert len(files) == 3
        for f in files:
            assert f in [
                "test-N=16,K=512-fp32.json",
                "test-N=32,K=1024-fp32.json",
                "test-N=8,K=2048-fp32.json",
            ]

        with open(f"{out}/test-N=16,K=512-fp32.json") as f:
            data = json.load(f)['config']
        assert len(data) == 2
        assert list(data.keys())[0] == "(256, 'torch.float32')"


def test_export_hoist_dtype():
    with tempfile.TemporaryDirectory() as tmp_dir:
        out = export_outdir(tmp_dir)
        cmd = [
            "opt_config_cli", "export", "--kernel", KERNEL, "--device",
            get_gpu_label(), "--basename", "test-%G.json", "--output-dir", tmp_dir,
            "--hoist-key", "N,K", "--hoist-dtype",
            "--output-fields", "key,config,path",
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE)
        assert res.returncode == 0

        files = [o for o in os.listdir(out) if o.endswith(".json")]
        assert len(files) == 3
        for f in files:
            assert f in [
                "test-N=16-K=512-src=torch.float32.json",
                "test-N=32-K=1024-src=torch.float32.json",
                "test-N=8-K=2048-src=torch.float32.json",
            ]

        with open(f"{out}/test-N=16-K=512-src=torch.float32.json") as f:
            data = json.load(f)
        assert list(data['config'].keys())[0] == "256"
        assert list(data['path'].keys())[0] == "256"


def test_export_keep_key():
    with tempfile.TemporaryDirectory() as tmp_dir:
        out = export_outdir(tmp_dir)
        cmd = [
            "opt_config_cli", "export", "--kernel", KERNEL, "--device",
            get_gpu_label(), "--basename", "test.json", "--output-dir", tmp_dir,
            "--keep-key", "M", "--output-fields", "key,config,path",
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE)
        assert res.returncode == 0

        files = [o for o in os.listdir(out) if o.endswith(".json")]
        assert len(files) == 3
        for f in files:
            assert f in [
                "test-N=16-K=512-src=torch.float32.json",
                "test-N=32-K=1024-src=torch.float32.json",
                "test-N=8-K=2048-src=torch.float32.json",
            ]

        with open(f"{out}/test-N=32-K=1024-src=torch.float32.json") as f:
            data = json.load(f)
        assert len(data['config']) == len(data['path']) == 1
        assert list(data['config'].keys())[0] == "512"
        assert list(data['path'].keys())[0] == "512"


def test_export_keep_timings():
    with tempfile.TemporaryDirectory() as tmp_dir:
        out = export_outdir(tmp_dir)
        cmd = [
            "opt_config_cli", "export", "--kernel", KERNEL, "--device",
            get_gpu_label(), "--basename", "test.json", "--output-dir", tmp_dir,
            "--keep-key", "M", "--output-fields", "key,config,path,timing",
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE)
        assert res.returncode == 0

        with open(f"{out}/test-N=32-K=1024-src=torch.float32.json") as f:
            data = json.load(f)
        assert len(data['config']) == len(data['path']) == len(data['timing']) == 1
