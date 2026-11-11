import pytest
import torch
import triton
import triton.language as tl

from triton._internal_testing import is_hip_cdna, is_hip_hcu


@triton.jit
def _convert_loop(source, scales, output, tiles, SCALED: tl.constexpr,
                  BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    values = tl.load(source + offsets)
    for tile in range(tiles):
        next_values = tl.load(source + (tile + 1) * BLOCK + offsets,
                              mask=tile + 1 < tiles, other=0.0)
        if SCALED:
            converted = values.to(tl.float32) * tl.load(scales + tile)
        else:
            converted = values
        tl.store(output + tile * BLOCK + offsets,
                 converted.to(output.dtype.element_ty))
        values = next_values


@pytest.mark.skipif(not (is_hip_hcu() and is_hip_cdna()), reason="requires a supported HCU CDNA target")
@pytest.mark.parametrize("source_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("destination_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("block", [64, 128, 256, 512])
@pytest.mark.parametrize("scaled", [False, True])
def test_fp8_loop_carriers(source_dtype, destination_dtype, block, scaled, monkeypatch, tmp_path):
    from triton._C.libtriton import hcu

    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path))
    scales = torch.tensor([0.1, 0.371, 1.00390625, -0.73, 1e-20, 1e20, 0.0, -0.0])
    source = torch.arange(256, dtype=torch.int32).to(torch.uint8)
    source = source.repeat(scales.numel() * block // source.numel()).view(source_dtype)
    expected = source.float().reshape(-1, block)
    if scaled:
        expected = expected * scales[:, None]
    expected = expected.to(destination_dtype).reshape(-1)
    device_source = source.cuda()
    device_scales = scales.cuda()
    results = []
    for packing_enabled in (True, False):
        with monkeypatch.context() as patch:
            if not packing_enabled:
                patch.setattr(hcu.passes.ttgpuir, "add_pack_bit8_loop_carriers", lambda manager: None)
            output = torch.empty(source.shape, dtype=destination_dtype, device="cuda")
            kernel = triton.jit(_convert_loop.fn)
            kernel[(1,)](device_source, device_scales, output, scales.numel(), scaled, block, num_warps=1)
            actual = output.cpu()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
            results.append(actual)
    storage_dtype = torch.int32 if destination_dtype == torch.float32 else torch.int16
    assert torch.equal(results[0].view(storage_dtype), results[1].view(storage_dtype))
