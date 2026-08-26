"""tt.hint: frontend stamps + HCU LLIR materialization self-checks."""
import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_hip
from triton.utils import hint


def _kernel_define_and_attrs(llir):
    """Return kernel `define` line and its `attributes #N = {...}` line."""
    lines = llir.splitlines()
    define_line = None
    attr_ref = None
    for line in lines:
        if not line.lstrip().startswith("define"):
            continue
        define_line = line
        for tok in line.replace("{", " ").split():
            if tok.startswith("#") and tok[1:].isdigit():
                attr_ref = tok
                break
        break
    if define_line is None:
        raise AssertionError("LLVM define not found in llir")
    attr_line = ""
    if attr_ref is not None:
        prefix = f"attributes {attr_ref} "
        for line in lines:
            if line.startswith(prefix):
                attr_line = line
                break
    if not attr_line:
        raise AssertionError(
            f"attributes line for {attr_ref} not found; define={define_line!r}")
    return define_line, attr_line


def _compile_hint_kernel():
    @triton.jit
    def kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
        pid = hint("val.no_negative", tl.program_id(0))
        hint("func.args_ptr.no_alias", True)
        hint("func.attr.optsize", True)
        hint("mod.flag.my_feature", True)
        hint("mod.flag.tile_factor", 4)
        hint("mod.flag.mode", "fast")

        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask), mask=mask)

    n = 128
    x = torch.randn(n, device="cuda")
    out = torch.empty_like(x)
    compiled = kernel[(1, )](x, out, n, BLOCK=128)
    torch.testing.assert_close(out, x)
    return compiled


def test_hint_ttir_markers():
    """Frontend must stamp tt.hint.* into TTIR."""
    compiled = _compile_hint_kernel()
    ttir = compiled.asm["ttir"]
    assert "tt.hint.val.no_negative" in ttir
    assert "tt.hint.func.args_ptr.no_alias" in ttir
    assert "tt.hint.func.attr.optsize" in ttir
    assert "tt.hint.mod.flag.my_feature" in ttir
    assert "tt.hint.mod.flag.tile_factor" in ttir
    assert "tt.hint.mod.flag.mode" in ttir


@pytest.mark.skipif(not is_hip(), reason="HCU apply-func-hints-and-clean runs on HIP/HCU only")
def test_hint_hcu_llir_materialized():
    """Self-check: func.* effects must appear in LLVM IR (llir), not only TTIR."""
    compiled = _compile_hint_kernel()
    assert "llir" in compiled.asm, sorted(compiled.asm.keys())
    llir = compiled.asm["llir"]
    define_line, attr_line = _kernel_define_and_attrs(llir)

    # Consumed FuncOp / ModuleOp / val hints must not leak as tt.hint.* into LLIR.
    assert "tt.hint.func." not in llir
    assert "tt.hint.val." not in llir
    assert "tt.hint.mod.flag." not in llir

    # func.args_ptr.no_alias → noalias on pointer args of define
    assert define_line.count("noalias") >= 2, define_line

    # func.attr.optsize → LLVM function attribute on attributes #N
    assert "optsize" in attr_line, attr_line
