# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""Compile-only regression for WDRA48 FP16 N-interleaved B + matrix-store."""
import re

import pytest
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


@triton.jit
def _gemm(a, b, c, M, N, K, stride_bk: tl.constexpr,
          stride_bn: tl.constexpr, CHECK: tl.constexpr):
    m = tl.program_id(0) * 256
    acc = tl.zeros((256, 256), tl.float32)
    for k in tl.range(0, tl.cdiv(K, 64), warp_specialize=True):
        av = tl.matrix_load(a, shape=[M, K], strides=[K, 1],
                            block_shape=[256, 64], offsets=[m, k * 64],
                            boundary_check=(0, 1))
        bv = tl.matrix_load(b, shape=[K, N], strides=[stride_bk, stride_bn],
                            block_shape=[64, 256], offsets=[k * 64, 0],
                            boundary_check=(0, 1))
        acc = tl.dot(av, bv, acc)
    tl.matrix_store(c, acc.to(tl.float16), shape=[M, N], strides=[N, 1],
                    block_shape=[256, 256], offsets=[m, 0], boundary_check=CHECK)


@pytest.mark.parametrize("layout", ["TT", "TN"])
@pytest.mark.parametrize("shape", [(512, 256, 128), (480, 224, 96)])
def test_wdra48_n_interleave_matrix_store(layout, shape):
    m, n, k = shape
    signature = dict(a="*fp16", b="*fp16", c="*fp16", M="i32", N="i32", K="i32",
                     stride_bk="constexpr", stride_bn="constexpr", CHECK="constexpr")
    constants = dict(stride_bk=1 if layout == "TN" else n,
                     stride_bn=k if layout == "TN" else 1,
                     CHECK=(0, 1) if m % 256 or n % 256 else ())
    compiled = triton.compile(
        ASTSource(_gemm, signature, constexprs=constants),
        target=GPUTarget("hip", "gfx946", 64),
        options=dict(num_stages=2, optimize_epilogue=True, mmac_layout_force=3,
                     wasp_enabled=True, wdra_enabled=True,
                     wasp_num_load_warps=4, wasp_num_mma_warps=8,
                     wdra_num_load_regs=16, wdra_num_mma_regs_main=244,
                     wdra_num_mma_regs_tail=244, empty_arrive_after_mmac=False,
                     enable_consumer_pingpong=True, enable_v_mmac_cluster=0,
                     buffer_cache_swizzle=True, hcu_use_gfx946_sched_model_v2=False,
                     amdgpu_enable_max_ilp_scheduling_strategy=True))
    asm = compiled.asm["amdgcn"]
    assert "matrix_load_64x16_b16" in asm
    assert "matrix_store_32x32_b16" in asm
    instruction = "ds_read_matrix_trans_format" if layout == "TN" else "ds_read_matrix_format"
    dimensions = "row:1 col:2" if layout == "TN" else "row:2 col:1"
    assert re.search(rf"{instruction}[^\n]*element:2 {dimensions} alt:1", asm)
    if layout == "TN":
        # The K16 step is 32 bytes, not a second N tile. Check the four
        # offsets for the first N32 group; PMD must additionally check values.
        for offset in (32, 4096, 4128):
            assert re.search(rf"{instruction}[^\n]*offset:{offset}\b", asm)
