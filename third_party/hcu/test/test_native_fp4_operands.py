"""Native packed/padded FP4 operand regression tests for gfx946."""

import re

import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _native_dot(A, B, C, K: tl.constexpr, A_FORMAT: tl.constexpr,
                B_FORMAT: tl.constexpr):
    a_pack: tl.constexpr = 2 if A_FORMAT == "e2m1" else 1
    b_pack: tl.constexpr = 2 if B_FORMAT == "e2m1" else 1
    rows = tl.arange(0, 16)
    ak = tl.arange(0, K // a_pack)
    bk = tl.arange(0, K // b_pack)
    a = tl.load(A + rows[:, None] * (K // a_pack) + ak[None, :])
    b = tl.load(B + rows[None, :] * (K // b_pack) + bk[:, None])
    c = tl.dot_scaled(a, None, A_FORMAT, b, None, B_FORMAT)
    tl.store(C + rows[:, None] * 16 + rows[None, :], c)


def _operand(fmt, k, generator):
    if fmt == "e4m3":
        x = (torch.randn((16, k), generator=generator) * .5).to(torch.float8_e4m3fn)
        return x.cuda(), x.float()
    packed = torch.randint(0, 256, (16, k // 2), generator=generator,
                           dtype=torch.uint8)
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2)
    values = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                           -0., -.5, -1., -1.5, -2., -3., -4., -6.])
    return packed.cuda(), values[codes.long()]


@pytest.mark.parametrize("a_format,b_format", [
    ("e4m3", "e2m1"), ("e2m1", "e4m3"), ("e2m1", "e2m1"),
])
@pytest.mark.parametrize("k", [32, 64, 128])
def test_native_fp4_operand_representation(a_format, b_format, k):
    if not torch.cuda.is_available():
        pytest.skip("requires gfx946")
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "hip" or target.arch.split(":")[0] not in ("gfx946", "gfx948"):
        pytest.skip("requires gfx946 native MMAC")

    generator = torch.Generator().manual_seed(43)
    a, a_ref = _operand(a_format, k, generator)
    b, b_ref = _operand(b_format, k, generator)
    out = torch.empty((16, 16), device="cuda", dtype=torch.float32)
    compiled = _native_dot[(1,)](a, b, out, k, a_format, b_format,
                                num_warps=4, num_stages=1,
                                enable_fp_fusion=False)
    torch.testing.assert_close(out.cpu(), a_ref @ b_ref.T, atol=1e-5, rtol=1e-5)

    # Cover packed FP4 on either side and K32/K64 instruction selection, so
    # logical K cannot be inferred from a single kWidth value or doubled twice.
    instructions = re.findall(r"\bv_mmac\S+", compiled.asm["amdgcn"])
    assert instructions
    if a_format == b_format == "e2m1":
        allowed = {"v_mmac_f32_16x16x32_fp4_fp4", "v_mmac_f32_16x16x64_fp4"}
    else:
        lhs = "fp4" if a_format == "e2m1" else "fp8"
        rhs = "fp4" if b_format == "e2m1" else "fp8"
        allowed = {f"v_mmac_f32_16x16x32_{lhs}_{rhs}"}
    assert set(instructions) <= allowed


@pytest.mark.parametrize("a_format,b_format", [
    ("e4m3", "e2m1"), ("e2m1", "e4m3"), ("e2m1", "e2m1"),
])
@pytest.mark.parametrize("k", [32, 64, 128])
def test_native_fp4_instruction_selection(a_format, b_format, k):
    # Compile without a device: mixed F8/F4 must not select packed-FP4 K64.
    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget
    from triton.backends.hcu.compiler import HIPBackend
    from triton.compiler import ASTSource

    backend = HIPBackend(GPUTarget("hip", "gfx946", 64))
    options = backend.parse_options(dict(num_warps=4, num_stages=1,
                                         enable_fp_fusion=False))
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    source = ASTSource(
        _native_dot,
        dict(A="*fp8e4nv" if a_format == "e4m3" else "*u8",
             B="*fp8e4nv" if b_format == "e4m3" else "*u8", C="*fp32"),
        constexprs=dict(K=k, A_FORMAT=a_format, B_FORMAT=b_format),
        attrs={(i,): [["tt.divisibility", 16], ["tt.pointer_range", 32]]
               for i in range(3)})
    module = source.make_ir(options, backend.get_codegen_implementation(options),
                            backend.get_module_map(), context)
    metadata = {}
    module = backend.make_ttir(module, metadata, options)
    module = backend.make_ttgir(module, metadata, options)
    instruction_k = 64 if a_format == b_format == "e2m1" and k >= 64 else 32
    assert f"instrShape = [16, 16, {instruction_k}]" in str(module)
    # Exercise the complete native operand lowering as well as selection.
    backend.make_llir(module, metadata, options)
