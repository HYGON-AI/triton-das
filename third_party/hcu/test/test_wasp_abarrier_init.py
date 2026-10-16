# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Compile-only regression: an init must not race another wave's arrive."""
import re

import pytest
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


@triton.jit
def init_gemm(a, b, c, K: tl.constexpr):
    rows = tl.arange(0, 128)
    ks = tl.arange(0, 16)
    ap = a + rows[:, None] * K + ks[None, :]
    bp = b + ks[:, None] * 128 + rows[None, :]
    acc = tl.zeros((128, 128), tl.float32)
    for _ in tl.range(0, K // 16, warp_specialize=True):
        av = tl.load(ap)
        bv = tl.load(bp)
        acc = tl.dot(av, bv, acc)
        ap += 16
        bp += 16 * 128
    tl.store(c + rows[:, None] * 128 + rows[None, :], acc.to(tl.float16))


def check_init_handoff(llir, producer_waves):
    """Check the optimized CFG, including the skipped-init path's rendezvous."""
    wave, = re.findall(r"(%[\w.]+) = .*call i32 .*@llvm.hcu.get.wave.id\(\)", llir)
    predicates = set(re.findall(
        rf"(%[\w.]+) = icmp eq i32 {re.escape(wave)}, 0\b", llir))
    assert predicates, "initializer is not elected by wave ID"
    blocks = dict(re.findall(
        r"^([\w.]+):[^\n]*\n(.*?)(?=^[\w.]+:|^}|\Z)", llir, re.M | re.S))
    initialized = []
    for label, body in blocks.items():
        ids = re.findall(r"call void @llvm.hcu.s.abarrier.init\(i32 (\d+),", body)
        if not ids:
            continue
        initialized.extend(ids)
        join, = re.findall(r"br label %([\w.]+)", body)
        guard = re.compile(rf"br i1 (%[\w.]+), label %{re.escape(label)}, label %{re.escape(join)}\b")
        parents = [guard.search(parent) for parent in blocks.values()]
        assert any(match and match[1] in predicates for match in parents), label
        instructions = [line.strip() for line in blocks[join].splitlines()
                        if line.strip() and not line.lstrip().startswith(";")]
        assert re.search(
            rf"call void @llvm.hcu.s.ebarrier.sync\(i32 4, i32 {producer_waves}\)",
            instructions[0]), (label, instructions[0])
        assert "abarrier.arrive" not in body
    assert len(initialized) >= 8
    assert len(initialized) == len(set(initialized))


@pytest.mark.parametrize("waves,wdra", [
    ((2, 2), False), ((4, 4), False), ((4, 4, 4), False),
    ((4, 4), True), ((4, 4, 4), True), ((4, 4, 4, 4), True),
])
def test_wasp_single_wave_abarrier_init(monkeypatch, waves, wdra):
    # No tensors/device needed: compile for an explicit target, then inspect CFG.
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", "1")
    options = dict(num_warps=waves[0], num_stages=2, use_async_copy=True,
                   wasp_partition_warps=waves, wasp_wdra=wdra,
                   optimize_epilogue=False, matrix_instr_nonkdim=16)
    if wdra:
        options["wasp_partition_regs"] = {2: (128, 256), 3: (88, 144, 140),
                                          4: (80, 96, 144, 160)}[len(waves)]
    compiled = triton.compile(
        ASTSource(init_gemm, {"a": "*fp16", "b": "*fp16", "c": "*fp16",
                              "K": "constexpr"}, constexprs={"K": 128},
                  attrs={(i,): [["tt.divisibility", 16], ["tt.pointer_range", 32]]
                         for i in range(3)}),
        target=GPUTarget("hip", "gfx946", 64), options=options)
    check_init_handoff(compiled.asm["llir"], waves[0])
    assert "buffer_load_" in compiled.asm["amdgcn"]
