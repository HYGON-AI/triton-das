# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""Exercise the actual C++ configuration search without requiring a GPU or an MLIR build."""
import pathlib
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("source", ["buffer_lds_config_test", "lds_read_model_test"])
def test_buffer_lds_config(tmp_path, source):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++17 compiler required")
    root = pathlib.Path(__file__).resolve().parents[1]
    binary = tmp_path / "buffer_lds_config"
    subprocess.run([compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                    "-I", str(root / "include"),
                    str(root / f"test/{source}.cpp"), "-o", str(binary)],
                   check=True)
    result = subprocess.run([str(binary)], check=True, capture_output=True,
                            text=True, timeout=300)
    assert "PASS" in result.stdout
