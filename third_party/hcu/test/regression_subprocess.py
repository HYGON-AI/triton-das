import subprocess
import sys
from pathlib import Path

import pytest


TEST_PATH = Path(__file__).resolve().parent
SRC_HOME = TEST_PATH.parents[2]

SUBPROCESS_CASES = [
    pytest.param("vector-add.py", ("--no_benchmark",), id="vector-add"),
    pytest.param("fused-softmax.py", ("--no_benchmark",), id="fused-softmax"),
    pytest.param("matrix-multiplication.py", ("--no_benchmark",), id="matrix-multiplication"),
    pytest.param("low-memory-dropout.py", (), id="low-memory-dropout"),
    pytest.param("layer-norm.py", ("--no_benchmark",), id="layer-norm"),
    pytest.param("extern-functions.py", (), id="extern-functions"),
    pytest.param("grouped-gemm.py", ("--no_benchmark",), id="grouped-gemm"),
    pytest.param("persistent-matmul.py", ("--no_benchmark",), id="persistent-matmul"),
]


@pytest.mark.parametrize("script, args", SUBPROCESS_CASES)
def test_hcu_regression_subprocess(script, args):
    subprocess.run(
        [sys.executable, str(TEST_PATH / script), *args],
        cwd=SRC_HOME,
        check=True,
    )
