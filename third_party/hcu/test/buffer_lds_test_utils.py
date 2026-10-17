# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""IR test runner shared by transport and scheduling tests.

Keep this module independent of either layer's tests so transport tests can
run with only the async-copy commit applied.
"""
import triton
import triton.language as tl
from triton._C.libtriton import ir, hcu


def run_pass(tmp_path, source, add_pass):
    path = tmp_path / "input.mlir"
    path.write_text(source)
    context = ir.context()
    ir.load_dialects(context)
    hcu.load_dialects(context)
    module = ir.parse_mlir_module(str(path), context)
    manager = ir.pass_manager(context)
    manager.enable_debug()
    add_pass(manager)
    manager.run(module, "buffer_to_lds_test")
    return str(module)


@triton.jit
def bf16_schedule_gemm(a, b, c, K, stride_am, stride_bk, stride_bn,
                       BK: tl.constexpr):
    tl.assume(K > 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    m = tl.arange(0, 256)
    n = tl.arange(0, 256)
    k = tl.arange(0, BK)
    ap = a + m[:, None] * stride_am + k[None, :]
    bp = b + k[:, None] * stride_bk + n[None, :] * stride_bn
    acc = tl.zeros((256, 256), tl.float32)
    for _ in range(0, K, BK):
        acc = tl.dot(tl.load(ap), tl.load(bp), acc)
        ap += BK
        bp += BK * stride_bk
    tl.store(c + m[:, None] * 256 + n[None, :], acc)


@triton.jit
def _runtime_k_gemm(a, b, c, K, COLUMN_B: tl.constexpr, BK: tl.constexpr):
    m = tl.arange(0, 256)
    n = tl.arange(0, 256)
    k = tl.arange(0, BK)
    ap = a + m[:, None] * K + k[None, :]
    if COLUMN_B:
        bp = b + k[:, None] + n[None, :] * K
    else:
        bp = b + k[:, None] * 256 + n[None, :]
    acc = tl.zeros((256, 256), tl.float32)
    for _ in range(0, K, BK):
        acc = tl.dot(tl.load(ap), tl.load(bp), acc)
        ap += BK
        bp += BK if COLUMN_B else BK * 256
    tl.store(c + m[:, None] * 256 + n[None, :], acc.to(c.dtype.element_ty))
