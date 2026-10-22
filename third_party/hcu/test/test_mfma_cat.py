# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""MFMA concatenation must preserve logical columns without communication."""
import re

import pytest
from triton._C.libtriton import hcu, passes
from buffer_lds_test_utils import run_pass


def cat_ir(half_n=128, warps=(4, 2), convert=False):
    output = f'tensor<256x{2 * half_n}xf32, #mma>'
    source = f'''#mma = #ttg.amd_mfma<{{version = 3, warpsPerCTA = [{warps[0]}, {warps[1]}], instrShape = [16, 16, 16], isTransposed = true, mmacLayout = 1, tilesPerWarp = [1, 2]}}>
#blocked = #ttg.blocked<{{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [4, 2], order = [1, 0]}}>
module attributes {{ttg.target = "hip:gfx936", "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32}} {{
  tt.func private @cat(%a: tensor<256x{half_n}xf32, #mma>, %b: tensor<256x{half_n}xf32, #mma>) -> {output} {{
    %v = tt.cat %a, %b : tensor<256x{half_n}xf32, #mma> -> {output}
    tt.return %v : {output}
  }}
}}
'''
    if convert:
        blocked = output.replace('#mma', '#blocked')
        source = source.replace(f'-> {output} {{', f'-> {blocked} {{', 1)
        source = source.replace(f'    tt.return %v : {output}',
                                f'    %w = ttg.convert_layout %v : {output} -> {blocked}\n    tt.return %w : {blocked}')
    return source


def test_mfma_cat_does_not_fold_into_blocked_layout(tmp_path):
    def optimize(pm):
        passes.common.add_canonicalizer(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.common.add_canonicalizer(pm)
    result = run_pass(tmp_path, cat_ir(convert=True), optimize)
    assert 'tt.cat' in result
    cat_line = next(line for line in result.splitlines() if 'tt.cat' in line)
    assert '#mma' in cat_line
    assert 'ttg.convert_layout' in result


@pytest.mark.parametrize("half_n", [64, 128, 256])
def test_mfma_cat_register_order(tmp_path, half_n):
    result = run_pass(tmp_path, cat_ir(half_n=half_n),
                      lambda pm: hcu.passes.ttgpuir.add_to_llvmir(pm, 'gfx936', False))
    assert 'tt.cat' not in result
    assert 'llvm.load' not in result and 'llvm.store' not in result
    # Hardware register order for N64/N128/N256 halves uses 8/16/32-register
    # chunks. The result alternates left/right chunks rather than appending
    # all left registers followed by all right registers.
    values = {value: (int(arg), int(index)) for value, arg, index in re.findall(
        r'(%\w+) = llvm.extractvalue %arg([01])\[(\d+)\]', result)}
    actual = [values[value] for value in re.findall(
        r'llvm.insertvalue (%\w+),', result) if value in values]
    chunk = {64: 8, 128: 16, 256: 32}[half_n]
    expected = [(side, index) for base in range(0, half_n // 2, chunk)
                for side in (0, 1) for index in range(base, base + chunk)]
    assert actual == expected


def test_mfma_cat_rejects_cross_lane_join(tmp_path, capfd):
    with pytest.raises(RuntimeError):
        run_pass(tmp_path, cat_ir(half_n=8),
                 lambda pm: hcu.passes.ttgpuir.add_to_llvmir(pm, 'gfx936', False))
    assert 'tt.cat' in capfd.readouterr().err
