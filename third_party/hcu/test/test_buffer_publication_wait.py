# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""Publication counts come from commit/slot order, not a four/eight constant."""

import re

import pytest
from triton._C.libtriton import hcu
from triton.backends.compiler import GPUTarget
from triton.backends.hcu.compiler import HIPBackend
from test_buffer_to_lds_passes import (
    run_pass, add_scheduled_buffer_to_lds, allocation_contract_fixture,
)


def test_dynamic_cache_stride_does_not_change_other_resources(tmp_path):
    prefix = allocation_contract_fixture("").split("module attributes {")[0]
    source = prefix + '''
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {ttg.target = "hip:gfx936", "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func @copies(%a: !tt.ptr<f16>, %b: !tt.ptr<f16>, %offset: tensor<256x16xi32, #blocked>, %stride: i32, %limit: i32, %dst: !ttg.memdesc<256x16xf16, #placement, #smem, mutable>) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %pitch = arith.constant 4096 : i32
    %a0 = amdg.buffer_load_to_local %a[%offset] stride = %stride into %dst {contiguity = 8 : i32} : <f16>[tensor<256x16xi32, #blocked>] -> <256x16xf16, #placement, #smem, mutable>
    scf.for %i = %zero to %limit step %one : i32 {
      %a1 = amdg.buffer_load_to_local %a[%offset] stride = %pitch into %dst {contiguity = 8 : i32} : <f16>[tensor<256x16xi32, #blocked>] -> <256x16xf16, #placement, #smem, mutable>
      %b0 = amdg.buffer_load_to_local %b[%offset] stride = %pitch into %dst {contiguity = 8 : i32} : <f16>[tensor<256x16xi32, #blocked>] -> <256x16xf16, #placement, #smem, mutable>
    } {hcu.block_pingpong.applied}
    tt.return
  }
}
'''
    result = run_pass(tmp_path, source, hcu.passes.ttgpuir.add_schedule_buffer_to_lds)
    copies = [line for line in result.splitlines() if "amdg.buffer_load_to_local" in line]
    assert len(copies) == 3
    assert "stride =" not in copies[0]
    assert "stride =" not in copies[1]
    assert "stride =" in copies[2]


def ring_ir(slots, copies, *, dtype="f16", drain=True, alias=False,
            missing_initial=False, nested=False, sliced_read=False):
    tensor = f"tensor<16x32x{dtype}, #blocked>"
    mem = f"!ttg.memdesc<16x32x{dtype}, #shared, #smem, mutable>"
    lines = [
        '#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 4], warpsPerCTA = [1, 1], order = [1, 0]}>',
        '#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 1, order = [1, 0]}>',
        '#smem = #ttg.shared_memory',
        'module attributes {ttg.target = "hip:gfx936", "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32} {',
        f'tt.func @ring(%ptr: !tt.ptr<{dtype}>, %off: tensor<16x32xi32, #blocked>, %limit: i32) -> {tensor} {{',
        '%c0 = arith.constant 0 : i32',
        '%c1 = arith.constant 1 : i32',
        '%true = arith.constant true',
        f'%zero = arith.constant dense<0.0> : {tensor}',
    ]

    def copy(slot, operand, tag):
        lines.append(
            f'%copy_{tag}_{slot}_{operand} = amdg.buffer_load_to_local %ptr[%off] into %s{slot}_{operand} '
            f'{{contiguity = 8 : i32}} : <{dtype}>[tensor<16x32xi32, #blocked>] -> <16x32x{dtype}, #shared, #smem, mutable>')

    for slot in range(slots):
        for operand in range(copies):
            lines.append(f'%s{slot}_{operand} = ttg.local_alloc : () -> {mem}')
            if not (missing_initial and slot == 2):
                copy(slot, operand, "init")
    lines += ['gpu.barrier',
              f'%result = scf.for %iv = %c0 to %limit step %c1 iter_args(%acc = %zero) -> ({tensor}) : i32 {{']
    if nested:
        lines += ['scf.if %true {', 'gpu.barrier', '}']
    acc = "%acc"
    for phase in range(slots):
        for operand in range(copies):
            source = phase if alias else (phase + 1) % slots
            name = f"r{phase}_{operand}"
            read_source = f'%s{source}_{operand}'
            if sliced_read:
                view = f'%v{phase}_{operand}'
                view_mem = f'!ttg.memdesc<16x32x{dtype}, #shared, #smem, mutable, 16x32>'
                lines.append(f'{view} = ttg.memdesc_subslice {read_source}[0, 0] : {mem} -> {view_mem}')
                read_source = view
            lines.append(f'%{name} = ttg.local_load {read_source} : '
                         f'{view_mem if sliced_read else mem} -> {tensor}')
            lines.append(f'%sum{phase}_{operand} = arith.addf {acc}, %{name} : {tensor}')
            acc = f"%sum{phase}_{operand}"
        for operand in range(copies):
            copy(phase, operand, "body")
        lines.append('ttg.local_barrier {hcu.block_pingpong.publish}')
        if drain:
            lines.append('amdg.memory_counter_wait ds(0)')
        lines.append('rocdl.s.barrier')
    lines += [f'scf.yield {acc} : {tensor}',
              '} {hcu.block_pingpong.applied, hcu.block_pingpong.static_buffers}',
              f'tt.return %result : {tensor}', '}', '}']
    return '\n'.join(lines)


@pytest.mark.parametrize("slots,copies", [(3, 1), (4, 1), (5, 1), (4, 2), (4, 3)])
@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_publication_counts_actual_future_groups(tmp_path, slots, copies, dtype):
    result = run_pass(tmp_path, ring_ir(slots, copies, dtype=dtype),
                      add_scheduled_buffer_to_lds)
    counts = [int(x) for x in re.findall(r'ttg.async_wait.*num = (\d+)', result)]
    assert counts[:slots] == [(slots - 2) * copies] * slots
    assert result.count('hcu.block_pingpong.drop_lgkmcnt') == slots
    assert 'hcu.block_pingpong.publish' not in result
    assert 'pending_copy_groups' not in result


@pytest.mark.parametrize("case", ["drain", "alias", "missing_initial", "nested"])
def test_unproved_publication_is_conservative(tmp_path, case):
    options = {"drain": False} if case == "drain" else {case: True}
    result = run_pass(tmp_path, ring_ir(4, 2, **options),
                      add_scheduled_buffer_to_lds)
    counts = [int(x) for x in re.findall(r'ttg.async_wait.*num = (\d+)', result)]
    if case in ("missing_initial", "nested"):
        assert counts[0] == 0
    if case in ("drain", "alias", "nested"):
        assert 'hcu.block_pingpong.drop_lgkmcnt' not in result


def test_static_subslice_keeps_publication_proof(tmp_path):
    result = run_pass(tmp_path, ring_ir(4, 1, sliced_read=True),
                      add_scheduled_buffer_to_lds)
    counts = [int(x) for x in re.findall(r'ttg.async_wait.*num = (\d+)', result)]
    assert counts[:4] == [2] * 4
    assert result.count('hcu.block_pingpong.drop_lgkmcnt') == 4


@pytest.mark.parametrize("copies_per_group,pending_groups", [
    (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 6), (2, 3), (2, 4),
])
def test_buffer_asyncmark_counts_groups(tmp_path, copies_per_group, pending_groups):
    lines = ['target triple = "amdgcn-amd-amdhsa"',
             'declare void @llvm.amdgcn.raw.buffer.load.async.lds(<4 x i32>, ptr addrspace(3), i32 immarg, i32, i32, i32 immarg, i32 immarg)',
             'declare void @llvm.amdgcn.asyncmark()',
             'declare void @llvm.amdgcn.wait.asyncmark(i16 immarg)',
             'declare void @llvm.amdgcn.s.barrier()',
             'define amdgpu_kernel void @probe(<4 x i32> inreg %rsrc, ptr addrspace(3) inreg %lds, ptr addrspace(1) %out) {']
    for group in range(8):
        for copy in range(copies_per_group):
            index = group * copies_per_group + copy
            lines += [f'%p{index} = getelementptr i8, ptr addrspace(3) %lds, i32 {index * 1024}',
                      f'call void @llvm.amdgcn.raw.buffer.load.async.lds(<4 x i32> %rsrc, ptr addrspace(3) %p{index}, i32 16, i32 {index * 1024}, i32 0, i32 0, i32 0)']
        lines.append('call void @llvm.amdgcn.asyncmark()')
    lines += [f'call void @llvm.amdgcn.wait.asyncmark(i16 {pending_groups})',
              'call void @llvm.amdgcn.s.barrier()',
              '%v = load volatile <2 x i32>, ptr addrspace(3) %lds, align 8',
              'store volatile <2 x i32> %v, ptr addrspace(1) %out, align 8',
              'call void @llvm.amdgcn.wait.asyncmark(i16 0)', 'ret void', '}']
    source = '\n'.join(lines)
    backend = HIPBackend(GPUTarget("hip", "gfx936", 64))
    asm = backend.make_amdgcn(source, {}, backend.parse_options({}))
    (tmp_path / "probe.ll").write_text(source)
    (tmp_path / "probe.s").write_text(asm)
    wait = asm.index(f"; wait_asyncmark({pending_groups})")
    read = asm.index("ds_read_b64", wait)
    prefix, publication = asm[:wait], asm[wait:read]
    assert len(re.findall(r"^\s*buffer_load_dwordx4\b", prefix, re.M)) == 8 * copies_per_group
    assert not re.search(r"\bvmcnt\(", prefix)
    assert re.findall(r"\bvmcnt\((\d+)\)", publication) == [str(pending_groups * copies_per_group)]
    assert re.search(r"^\s*s_barrier\s*$", publication, re.M)
