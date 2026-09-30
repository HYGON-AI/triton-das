# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""FP8/FP16 buffer-to-LDS round trips, without GEMM or arithmetic."""
import re

import pytest
import torch
import triton

from buffer_lds_test_utils import run_pass
from triton._C.libtriton import hcu


# Historical legal placements for transport testing, not frozen search winners.
# The standalone model test checks search optimality with an independent oracle.
# role, rows, rowBytes, waves, copyBytes, chunk, wrap, step, group
TRANSPORT_CONFIGS = {
    1: [
        ("K", 64, 32, 2, 16, 4, 2, 16, 2),
        ("K", 128, 32, 4, 16, 4, 4, 16, 4),
        ("K", 256, 32, 8, 16, 4, 4, 16, 8),
        ("N", 32, 64, 2, 16, 4, 2, 16, 2),
        ("N", 32, 128, 4, 16, 4, 4, 16, 4),
        ("N", 32, 256, 8, 16, 2, 8, 16, 8),
    ],
    2: [
        ("K", 64, 32, 2, 16, 4, 2, 16, 2),
        ("K", 128, 32, 4, 16, 4, 4, 16, 4),
        ("K", 256, 32, 8, 16, 4, 4, 16, 8),
        ("N", 16, 128, 2, 16, 1, 2, 64, 2),
        ("N", 16, 256, 4, 16, 1, 2, 64, 4),
        ("N", 16, 512, 8, 16, 1, 2, 64, 2),
    ],
}

DTYPES = [("f8E4M3FN", torch.float8_e4m3fn, 1),
          ("f8E5M2", torch.float8_e5m2, 1),
          ("f16", torch.float16, 2)]
COPY_CASES = [
    pytest.param(dtype, torch_dtype, element_bytes, *copy_config,
                 id=f"{dtype}-{copy_config[0]}-w{copy_config[3]}-{copy_config[1]}x{copy_config[2]}")
    for dtype, torch_dtype, element_bytes in DTYPES
    for copy_config in TRANSPORT_CONFIGS[element_bytes]
]
# Exercise narrower copies now that automatic selection compares every width.
COPY_CASES += [
    pytest.param(dtype, torch_dtype, element_bytes, *copy_config[:4], width, *copy_config[5:],
                 id=f"{dtype}-{copy_config[0]}-copy{width}")
    for dtype, torch_dtype, element_bytes in DTYPES
    for copy_config in (TRANSPORT_CONFIGS[element_bytes][0], TRANSPORT_CONFIGS[element_bytes][-1])
    for width in (4, 8)
]


def _logical_byte(byte, rows, row_bytes, waves, copy_bytes, rows_per_chunk,
                  waves_per_group, transfer_major):
    bytes_per_transfer = 64 * copy_bytes
    bytes_per_wave = rows * row_bytes // waves
    if transfer_major:
        wave = byte // bytes_per_transfer % waves
        local = (byte // (waves * bytes_per_transfer) * bytes_per_transfer +
                 byte % bytes_per_transfer)
    else:
        wave = byte // bytes_per_wave
        local = byte % bytes_per_wave
    group_waves = waves_per_group or waves
    local_row = local // row_bytes
    row = (wave // group_waves * (rows // waves * group_waves) +
           (local_row // rows_per_chunk * group_waves + wave % group_waves) *
           rows_per_chunk + local_row % rows_per_chunk)
    return row * row_bytes + local % row_bytes


def _encodings(element_bytes, contiguous_dim, rows, row_bytes, waves,
               copy_bytes, rows_per_chunk, waves_per_group, transfer_major):
    registers, lanes, warps, offsets = [], [], [], []
    tile_bytes = rows * row_bytes
    bytes_per_transfer = 64 * copy_bytes
    bytes_per_wave = tile_bytes // waves
    bit = element_bytes
    while bit < tile_bytes:
        logical = _logical_byte(bit, rows, row_bytes, waves, copy_bytes,
                                rows_per_chunk, waves_per_group,
                                transfer_major) // element_bytes
        basis = [logical // (row_bytes // element_bytes),
                 logical % (row_bytes // element_bytes)]
        if contiguous_dim == 0:
            basis.reverse()
        offsets.append(basis)
        if bit < copy_bytes:
            registers.append(basis)
        elif bit < bytes_per_transfer:
            lanes.append(basis)
        elif transfer_major:
            (warps if bit < bytes_per_transfer * waves else registers).append(basis)
        else:
            (registers if bit < bytes_per_wave else warps).append(basis)
        bit *= 2
    return registers, lanes, warps, offsets


@pytest.mark.parametrize("contiguous_dim", [0, 1])
@pytest.mark.parametrize("transfer_major", [False, True])
@pytest.mark.parametrize(
    "dtype,torch_dtype,element_bytes,role,rows,row_bytes,waves,copy_bytes,"
    "rows_per_chunk,wrap_count,wrap_step,waves_per_group", COPY_CASES)
def test_async_copy_roundtrip(
        tmp_path, dtype, torch_dtype, element_bytes, role, contiguous_dim, rows,
        row_bytes, waves, copy_bytes, rows_per_chunk, wrap_count, wrap_step,
        waves_per_group, transfer_major, consumer=False, consumer_k_width=4,
        required_arch="gfx936", physical_readback=False):
    if (not torch.cuda.is_available() or
            required_arch not in torch.cuda.get_device_properties(0).gcnArchName):
        pytest.skip("required HCU target is unavailable")
    registers, lanes, warps, offsets = _encodings(
        element_bytes, contiguous_dim, rows, row_bytes, waves, copy_bytes,
        rows_per_chunk, waves_per_group, transfer_major)
    row_elements = row_bytes // element_bytes
    shape = ((row_elements, rows) if contiguous_dim == 0 else
             (rows, row_elements))
    shape_text = "x".join(map(str, shape))
    major, minor = ("cb", "rb") if contiguous_dim == 0 else ("rb", "cb")
    global_stride = shape[0] if contiguous_dim == 0 else shape[1]
    buffer_lds_extras = (
        f", wavesPerRowGroup = {waves_per_group}" if waves_per_group else "")
    if transfer_major:
        buffer_lds_extras += ", transferMajor = true"
    consumer_encoding = "#producer"
    consumer_declaration = ""
    write_conversion = ""
    write_offset = "%offset"
    if consumer:
        op_idx = 1 - contiguous_dim if role == "K" else contiguous_dim
        warp_shape = [waves, 1] if op_idx == 0 else [1, waves]
        tiles = [2, 1] if op_idx == 0 else [1, 2]
        consumer_encoding = "#consumer"
        consumer_declaration = f'''
#mma = #ttg.amd_mfma<{{version = 3, warpsPerCTA = {warp_shape}, instrShape = [16, 16, {4 * consumer_k_width}], isTransposed = false, mmacLayout = 1, tilesPerWarp = {tiles}}}>
#consumer = #ttg.dot_op<{{opIdx = {op_idx}, parent = #mma, kWidth = {consumer_k_width}}}>
'''
        # Keep the loaded fragment in consumer encoding through the store.
        # Converting it straight back lets canonicalization fold the consumer
        # local_load away, making this silently test producer readback again.
        write_conversion = f"%consumer_offset = ttg.convert_layout %offset : tensor<{shape_text}xi32, #producer> -> tensor<{shape_text}xi32, #consumer>"
        write_offset = "%consumer_offset"
    read_source = "%dst"
    read_type = f"!ttg.memdesc<{shape_text}x{dtype}, #buffer_lds, #smem, mutable>"
    physical_reinterpret = ""
    if physical_readback:
        physical_type = read_type.replace("#buffer_lds", "#shared")
        physical_reinterpret = (
            f"%physical = ttg.memdesc_reinterpret %dst : {read_type} -> "
            f"{physical_type}")
        read_source = "%physical"
        read_type = physical_type
    source = f'''
{consumer_declaration}
#producer = #ttg.linear<{{register = {registers}, lane = {lanes}, warp = {warps}, block = []}}>
#shared = #ttg.shared_linear<{{offset = {offsets}}}, alignment = 16>
#buffer_lds = #ttg.hcu_buffer_lds_shared<{{linearComponent = #shared, elementBytes = {element_bytes}, contiguousDim = {contiguous_dim}, rows = {rows}, rowBytes = {row_bytes}, waves = {waves}, copyBytesPerLane = {copy_bytes}, rowsPerChunk = {rows_per_chunk}, wrapCount = {wrap_count}, wrapStepBytes = {wrap_step}{buffer_lds_extras}}}>
#smem = #ttg.shared_memory
#r = #ttg.slice<{{dim = 1, parent = #producer}}>
#c = #ttg.slice<{{dim = 0, parent = #producer}}>
module attributes {{"ttg.target" = "hip:{required_arch}", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = {waves} : i32, "ttg.threads-per-warp" = 64 : i32}} {{
  tt.func public @copy(%src: !tt.ptr<{dtype}> {{tt.divisibility = 16 : i32}}, %out: !tt.ptr<{dtype}> {{tt.divisibility = 16 : i32}}) {{
    %r = tt.make_range {{start = 0 : i32, end = {shape[0]} : i32}} : tensor<{shape[0]}xi32, #r>
    %c = tt.make_range {{start = 0 : i32, end = {shape[1]} : i32}} : tensor<{shape[1]}xi32, #c>
    %re = tt.expand_dims %r {{axis = 1 : i32}} : tensor<{shape[0]}xi32, #r> -> tensor<{shape[0]}x1xi32, #producer>
    %ce = tt.expand_dims %c {{axis = 0 : i32}} : tensor<{shape[1]}xi32, #c> -> tensor<1x{shape[1]}xi32, #producer>
    %rb = tt.broadcast %re : tensor<{shape[0]}x1xi32, #producer> -> tensor<{shape_text}xi32, #producer>
    %cb = tt.broadcast %ce : tensor<1x{shape[1]}xi32, #producer> -> tensor<{shape_text}xi32, #producer>
    %stride = arith.constant dense<{global_stride}> : tensor<{shape_text}xi32, #producer>
    %major = arith.muli %{major}, %stride : tensor<{shape_text}xi32, #producer>
    %offset = arith.addi %major, %{minor} : tensor<{shape_text}xi32, #producer>
    %dst = ttg.local_alloc : () -> !ttg.memdesc<{shape_text}x{dtype}, #buffer_lds, #smem, mutable>
    %v = amdg.buffer_load %src[%offset] : tensor<{shape_text}x{dtype}, #producer>
    ttg.local_store %v, %dst : tensor<{shape_text}x{dtype}, #producer> -> !ttg.memdesc<{shape_text}x{dtype}, #buffer_lds, #smem, mutable>
    {physical_reinterpret}
    %read = ttg.local_load {read_source} : {read_type} -> tensor<{shape_text}x{dtype}, {consumer_encoding}>
    {write_conversion}
    amdg.buffer_store %read, %out[{write_offset}] : tensor<{shape_text}x{dtype}, {consumer_encoding}>
    tt.return
  }}
}}
'''
    converted = run_pass(tmp_path, source, lambda pm:
                         hcu.passes.ttgpuir.add_convert_buffer_lds_copies(pm))
    assert "amdg.buffer_load_to_local" in converted
    assert f"elementBytes = {element_bytes}" in converted
    path = tmp_path / "copy.ttgir"
    path.write_text(converted)
    kernel = triton.compile(
        str(path), options={"use_async_copy": True, "num_warps": waves})
    assert "llvm.amdgcn.raw.ptr.buffer.load.async.lds" in kernel.asm["llir"]
    assert "llvm.amdgcn.asyncmark" in kernel.asm["llir"]
    assert "llvm.amdgcn.wait.asyncmark(i16 0)" in kernel.asm["llir"]
    assert re.search(r"buffer_load_\w+.*\blds\b", kernel.asm["amdgcn"])
    assert "v_mmac" not in kernel.asm["amdgcn"]
    if consumer and role == "N" and element_bytes == 2:
        assert "ds_read_m32x16_b16" in kernel.asm["amdgcn"]
    if consumer and role == "N" and element_bytes == 1 and consumer_k_width == 8:
        assert "ds_read_m32x32_b8" in kernel.asm["amdgcn"]

    num_elements = rows * row_elements
    generator = torch.Generator(device="cuda").manual_seed(
        1009 + rows + 3 * row_bytes + waves + element_bytes)
    storage_dtype = torch.uint8 if element_bytes == 1 else torch.uint16
    high = 1 << (8 * element_bytes)
    bits = torch.randint(0, high, (num_elements,), device="cuda",
                         dtype=storage_dtype, generator=generator)
    prefix = torch.arange(min(high, num_elements), device="cuda",
                          dtype=torch.int32).to(storage_dtype)
    bits[:prefix.numel()] = prefix
    src = bits.view(torch_dtype)
    dst = torch.empty_like(src)
    kernel[(1, 1, 1)](src, dst)
    expected = bits
    if physical_readback:
        # Independently reconstruct the physical LDS image. M0 rotates each
        # issued wave transfer, not an LDS 128B phase or the wave's full tile.
        physical_to_logical = [0] * num_elements
        period = 64 * copy_bytes
        total_per_wave = rows * row_bytes // waves
        for unwrapped in range(0, rows * row_bytes, element_bytes):
            logical = _logical_byte(
                unwrapped, rows, row_bytes, waves, copy_bytes,
                rows_per_chunk, waves_per_group, transfer_major)
            wave = (unwrapped // period % waves if transfer_major else
                    unwrapped // total_per_wave)
            rotation = wave % wrap_count * wrap_step
            physical = (unwrapped // period * period +
                        (unwrapped % period + rotation) % period)
            physical_to_logical[physical // element_bytes] = logical // element_bytes
        indices = torch.tensor(physical_to_logical, device="cuda", dtype=torch.long)
        expected = bits.to(torch.int32)[indices].to(storage_dtype)
    torch.testing.assert_close(dst.view(storage_dtype), expected, atol=0, rtol=0)


@pytest.mark.parametrize("contiguous_dim", [0, 1])
@pytest.mark.parametrize(
    "dtype,torch_dtype,element_bytes,copy_bytes,wrap_step",
    [
        pytest.param("f16", torch.float16, 2, 4, 4, id="f16-copy4-fine1"),
        pytest.param("f16", torch.float16, 2, 8, 12, id="f16-copy8-fine3"),
        pytest.param("f16", torch.float16, 2, 16, 20,
                     id="f16-copy16-coarse1-fine1"),
        pytest.param("f8E4M3FN", torch.float8_e4m3fn, 1, 4, 4,
                     id="fp8-copy4-fine1"),
        pytest.param("f8E4M3FN", torch.float8_e4m3fn, 1, 8, 12,
                     id="fp8-copy8-fine3"),
        pytest.param("f8E4M3FN", torch.float8_e4m3fn, 1, 16, 20,
                     id="fp8-copy16-coarse1-fine1"),
    ])
def test_nmz_split_wrap_physical_mapping(
        tmp_path, dtype, torch_dtype, element_bytes, copy_bytes, wrap_step,
        contiguous_dim):
    """Validate NMZ's split M0 wrap field against the physical LDS image.

    Four waves exercise rotations 0, step, 2*step and 3*step.  In particular,
    a 20-byte step requires both the coarse 4-dword and fine 1-dword fields;
    producer/consumer agreement alone would not detect encoding those bits in
    the wrong M0 positions, so this test reads the physical allocation back.
    """
    test_async_copy_roundtrip(
        tmp_path, dtype, torch_dtype, element_bytes, "K", contiguous_dim,
        rows=4, row_bytes=1024, waves=4, copy_bytes=copy_bytes,
        rows_per_chunk=1, wrap_count=4, wrap_step=wrap_step,
        waves_per_group=4, transfer_major=False,
        required_arch="gfx938", physical_readback=True)


@pytest.mark.parametrize("contiguous_dim", [0, 1])
@pytest.mark.parametrize(
    "dtype,torch_dtype",
    [("f8E4M3FN", torch.float8_e4m3fn),
     ("f8E5M2", torch.float8_e5m2)])
def test_nmz_async_copy_matrix_read(tmp_path, dtype, torch_dtype,
                                    contiguous_dim):
    """Exercise NMZ BufferLds through the native b8 matrix-read consumer."""
    role, rows, row_bytes, waves, copy_bytes, chunk, wrap, step, group = \
        TRANSPORT_CONFIGS[1][3]
    assert role == "N"
    test_async_copy_roundtrip(
        tmp_path, dtype, torch_dtype, 1, role, contiguous_dim, rows, row_bytes,
        waves, copy_bytes, chunk, wrap, step, group, transfer_major=False,
        consumer=True, consumer_k_width=8, required_arch="gfx938")


@pytest.mark.parametrize("dtype,torch_dtype,element_bytes,k_width",
                         [(*d, 4) for d in DTYPES] +
                         [(*d, 8) for d in DTYPES if d[2] == 1])
@pytest.mark.parametrize("role", ["K", "N"])
@pytest.mark.parametrize("copy_bytes", [4, 8, 16])
@pytest.mark.parametrize("contiguous_dim", [0, 1])
def test_async_copy_consumer_fragment(tmp_path, dtype, torch_dtype, element_bytes,
                                      role, copy_bytes, contiguous_dim, k_width):
    # Read and store through the real MMAC consumer encoding for bit-exact
    # global verification. This remains a copy test: no FP8 GEMM.
    copy_config = next(p for p in TRANSPORT_CONFIGS[element_bytes] if p[0] == role)
    _, rows, row_bytes, waves, _, chunk, wrap, step, group = copy_config
    test_async_copy_roundtrip(
        tmp_path, dtype, torch_dtype, element_bytes, role, contiguous_dim, rows,
        row_bytes, waves, copy_bytes, chunk, wrap, step, group,
        transfer_major=False, consumer=True, consumer_k_width=k_width)
