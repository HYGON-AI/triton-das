"""CPU regressions for checked buffer-to-LDS layout entry points."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("entry,case", [
    (entry, case)
    for entry in ("producer_layout", "to_ir")
    for case in ("valid", "negative", "missing_bit", "non_bijective", "wrong_rank",
                 "oversized_value", "oversized_input", "oversized_output", "transposed", "zero_element")
    if not (entry == "to_ir" and case == "wrong_rank")
])
def test_buffer_lds_layout_checked(entry, case):
    # Isolate the native binding so a regression reports SIGABRT as a test
    # failure without taking down pytest. No GPU or active driver is needed.
    script = r'''
import resource
import sys
from triton._C.libtriton import hcu
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.hcu import BufferLdsSharedLayout

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
entry, case = sys.argv[1:]
bases = [[0, 1], [0, 2], [0, 4], [0, 8], [1, 0], [2, 0],
         [32, 0], [64, 0], [128, 0], [4, 0], [8, 0], [16, 0]]
if case == "negative":
    bases[0] = [0, -1]
elif case == "missing_bit":
    bases[1] = bases[0].copy()
elif case == "non_bijective":
    bases.append(bases[0].copy())
elif case == "wrong_rank":
    bases[0] = [0, 1, 0]
elif case == "oversized_value":
    bases[0] = [0, 1 << 30]
elif case == "oversized_input":
    bases += [[0, 0]] * 31
elif case == "oversized_output":
    bases[0] = [1 << 29, 1 << 29]
if case == "transposed":
    bases = [[c, r] for r, c in bases]
valid = case in ("valid", "transposed")
try:
    if case == "wrong_rank":
        # Exercise the native entry too, bypassing Python rank validation.
        hcu.get_buffer_lds_producer_layout(
            bases, [], 16, 2, 1, 256, 32, 8, 16, 4, 4, 16, 1, False, 0)
    else:
        layout = BufferLdsSharedLayout(
            gl.SharedLinearLayout(bases), 0 if case == "zero_element" else 2,
            0 if case == "transposed" else 1, 256, 32, 8, 16, 4, 4, 16)
        if entry == "to_ir":
            from triton._C.libtriton import ir, gluon_ir
            context = ir.context()
            ir.load_dialects(context)
            hcu.load_dialects(context)
            builder = gluon_ir.GluonOpBuilder(context)
            assert isinstance(layout._to_ir(builder), ir.attribute)
        else:
            result = layout.producer_layout()
            expected = gl.DistributedLinearLayout(
                reg_bases=[[0, 1], [0, 2], [0, 4]],
                lane_bases=[[0, 8], [1, 0], [2, 0], [32, 0], [64, 0], [128, 0]],
                warp_bases=[[4, 0], [8, 0], [16, 0]],
                block_bases=[], shape=[256, 16])
            if case == "transposed":
                expected = gl.DistributedLinearLayout(
                    *([list(reversed(b)) for b in group] for group in
                      [expected.reg_bases, expected.lane_bases, expected.warp_bases, expected.block_bases]),
                    shape=[16, 256])
            assert result == expected
except ValueError:
    assert not valid
    print("VALUE_ERROR")
else:
    assert valid, "invalid layout was accepted"
    print("SUCCESS")
'''
    result = subprocess.run([sys.executable, "-c", script, entry, case], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == ("SUCCESS" if case in ("valid", "transposed") else "VALUE_ERROR")
