"""End-to-end test for elementwise FMA codegen on HCU/HIP."""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from triton._internal_testing import is_hip


_HCU_TEST_DEVICE = "cuda"
_FMA_RE = re.compile(r"\bv_fma_f32\b|\bv_fmac_f32_e32\b|\bv_mac_f32\b")


@pytest.mark.skipif(reason="elementwise FMA test is HIP-specific", condition=not is_hip())
def test_torch_compile_elementwise_fma_emits_fma():
    def run_with_cache(cache_dir):
        env = os.environ.copy()
        env["TRITON_CACHE_DIR"] = cache_dir

        script = r"""
import torch

torch.manual_seed(0)

@torch.compile(fullgraph=True)
def elementwise_fma(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return a * b + c

a = torch.randn((256, 256), device="cuda", dtype=torch.float32)
b = torch.randn((256, 256), device="cuda", dtype=torch.float32)
c = torch.randn((256, 256), device="cuda", dtype=torch.float32)

out = elementwise_fma(a, b, c)
ref = a * b + c

torch.testing.assert_close(out, ref, rtol=1e-6, atol=1e-6)
torch.cuda.synchronize()
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            "subprocess failed\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

        gcn_files = sorted(Path(cache_dir).rglob("triton_poi_fused_add_mul_0.amdgcn"))
        assert gcn_files, f"no fused add_mul kernel found under TRITON_CACHE_DIR={cache_dir}"

        matched = []
        for path in gcn_files:
            text = path.read_text(encoding="utf-8")
            if _FMA_RE.search(text):
                matched.append(path)

        assert matched, (
            "no FMA/FMAC instruction found in generated add_mul kernels\n"
            f"checked: {[str(p) for p in gcn_files]}"
        )

    cache_dir = os.environ.get("TRITON_CACHE_DIR")
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        run_with_cache(cache_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="triton-fma-cache-") as cache_dir:
            run_with_cache(cache_dir)
