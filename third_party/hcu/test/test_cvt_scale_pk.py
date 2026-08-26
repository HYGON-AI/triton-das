#!/usr/bin/env python3

"""OCP fp8/bf8 convert smoke for HCU (gfx938 CVT_FP8F32 / gfx946 CVT_SCALE_PK).

Small tensors only (PMD-friendly). Asserts exact bit/value results for a fixed
input set rather than print-only.
"""

import pytest
import torch
import triton
import triton.language as tl

from triton._internal_testing import is_hip


def arch_name():
    props = triton.runtime.driver.active.utils.get_device_properties(0)
    return props["arch"].split(":")[0]


def is_hip_gfx938():
    return is_hip() and arch_name() == "gfx938"


def is_hip_gfx946():
    return is_hip() and arch_name() == "gfx946"


def is_hip_hcu_ocp_fp8():
    """HCU targets that expose OCP float8 (e4nv/e5) conversion."""
    return is_hip_gfx938() or is_hip_gfx946()


@pytest.fixture
def device():
    return "cuda"


@triton.jit
def convert_kernel(src_ptr, dst_ptr, n: tl.constexpr, rounding: tl.constexpr):
    offs = tl.arange(0, n)
    x = tl.load(src_ptr + offs)
    if rounding == "":
        y = x.to(dst_ptr.dtype.element_ty)
    else:
        y = x.to(dst_ptr.dtype.element_ty, fp_downcast_rounding=rounding)
    tl.store(dst_ptr + offs, y)


def torch_storage(tl_ty):
    bits = tl_ty.primitive_bitwidth
    if bits == 8:
        return torch.uint8
    if bits == 16:
        if tl_ty == tl.float16:
            return torch.float16
        if tl_ty == tl.bfloat16:
            return torch.bfloat16
        return torch.int16
    if bits == 32:
        return torch.float32
    raise ValueError(f"unsupported bitwidth {bits}")


def run_convert(src_cpu, src_tl, dst_tl, rounding=""):
    n = src_cpu.numel()
    assert n <= 64, "keep tiny for PMD/model"
    assert src_cpu.device.type == "cpu"

    src_gpu = src_cpu.to("cuda")
    dst_gpu = torch.empty(n, dtype=torch_storage(dst_tl), device="cuda")
    convert_kernel[(1,)](
        triton.reinterpret(src_gpu, src_tl),
        triton.reinterpret(dst_gpu, dst_tl),
        n,
        rounding,
    )
    torch.cuda.synchronize()
    return dst_gpu.to("cpu")


def u8_list(t):
    return [int(x) & 0xFF for x in t.view(torch.uint8)]


def f_list(t):
    return [float(x) for x in t]


# Fixed inputs / golden results (validated on gfx946 PMD; same OCP encodings
# expected on gfx938 HW pk / SW paths for these exact values).
SRC_F16 = torch.tensor(
    [0.0, 1.0, -1.0, 0.5, 2.0, -0.5, 0.25, 4.0], dtype=torch.float16
)
SRC_BF16 = torch.tensor(
    [0.0, 1.0, -1.0, 0.5, 2.0, -0.5, 0.25, 4.0], dtype=torch.bfloat16
)
SRC_F32 = torch.tensor(
    [0.0, 1.0, -1.0, 0.5, 2.0, -0.5, 0.25, 4.0], dtype=torch.float32
)
SRC_F32_N2 = torch.tensor([0.0, 1.0], dtype=torch.float32)

# OCP e5m2 / e4m3 raw bytes
SRC_E5 = torch.tensor([0x00, 0x3C, 0x38, 0x40, 0xBC, 0x01, 0x7B, 0x80], dtype=torch.uint8)
SRC_E4 = torch.tensor([0x00, 0x38, 0x30, 0x40, 0xB8, 0x01, 0x7E, 0x80], dtype=torch.uint8)

GOLD_DOWN_E4 = [0x00, 0x38, 0xB8, 0x30, 0x40, 0xB0, 0x28, 0x48]
GOLD_DOWN_E5 = [0x00, 0x3C, 0xBC, 0x38, 0x40, 0xB8, 0x34, 0x44]
GOLD_UP_E5_F = [0.0, 1.0, 0.5, 2.0, -1.0, 1.52587890625e-05, 57344.0, -0.0]
GOLD_UP_E4_F = [0.0, 1.0, 0.5, 2.0, -1.0, 0.001953125, 448.0, -0.0]


pytestmark = pytest.mark.skipif(
    not is_hip_hcu_ocp_fp8(),
    reason="OCP fp8 convert smoke only on gfx938/gfx946 HCU backends",
)


@pytest.mark.parametrize(
    "src,src_tl,dst_tl,rounding,golden",
    [
        (SRC_F16, tl.float16, tl.float8e4nv, "rtne", GOLD_DOWN_E4),
        (SRC_F16, tl.float16, tl.float8e5, "rtne", GOLD_DOWN_E5),
        (SRC_BF16, tl.bfloat16, tl.float8e4nv, "rtne", GOLD_DOWN_E4),
        (SRC_BF16, tl.bfloat16, tl.float8e5, "rtne", GOLD_DOWN_E5),
        (SRC_F32, tl.float32, tl.float8e4nv, "rtne", GOLD_DOWN_E4),
        (SRC_F32, tl.float32, tl.float8e5, "rtne", GOLD_DOWN_E5),
        (SRC_F32_N2, tl.float32, tl.float8e4nv, "rtne", [0x00, 0x38]),
        (SRC_F32_N2, tl.float32, tl.float8e5, "rtne", [0x00, 0x3C]),
    ],
    ids=[
        "fp16->fp8e4nv",
        "fp16->fp8e5",
        "bf16->fp8e4nv",
        "bf16->fp8e5",
        "fp32->fp8e4nv(n=8)",
        "fp32->fp8e5(n=8)",
        "fp32->fp8e4nv(n=2)",
        "fp32->fp8e5(n=2)",
    ],
)
def test_ocp_fp8_downcast(src, src_tl, dst_tl, rounding, golden):
    # n=8 is 4x 2-wide packs on gfx946 (numElements==2); not the dead 4-wide OR path.
    out = run_convert(src.clone(), src_tl, dst_tl, rounding=rounding)
    assert u8_list(out) == golden


@pytest.mark.parametrize(
    "src,src_tl,dst_tl,golden",
    [
        (SRC_E5, tl.float8e5, tl.float16, GOLD_UP_E5_F),
        (SRC_E4, tl.float8e4nv, tl.float16, GOLD_UP_E4_F),
        (SRC_E5, tl.float8e5, tl.bfloat16, GOLD_UP_E5_F),
        (SRC_E4, tl.float8e4nv, tl.bfloat16, GOLD_UP_E4_F),
        (SRC_E4, tl.float8e4nv, tl.float32, GOLD_UP_E4_F),
        (SRC_E5, tl.float8e5, tl.float32, GOLD_UP_E5_F),
    ],
    ids=[
        "fp8e5->fp16",
        "fp8e4nv->fp16",
        "fp8e5->bf16",
        "fp8e4nv->bf16",
        "fp8e4nv->fp32",
        "fp8e5->fp32",
    ],
)
def test_ocp_fp8_upcast(src, src_tl, dst_tl, golden):
    out = run_convert(src.clone(), src_tl, dst_tl, rounding="")
    assert f_list(out) == golden
