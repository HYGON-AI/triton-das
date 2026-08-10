"""Semantics of ``tl.fdiv`` fast vs precise division on fp32 special values.

Guards the ``tl.fdiv(ieee_rounding=False)`` fast path added in
"add tl.fdiv(ieee_rounding=False) support":

* ``tl.fdiv(x, y, ieee_rounding=False)`` (default)  -> fast path,
  on AMD/HCU lowered to ``v_rcp_f32 + v_mul_f32`` (~1 ulp on normals);
* ``tl.fdiv(x, y, ieee_rounding=True)`` and ``tl.math.div_rn`` -> precise
  IEEE round-to-nearest division;
* plain ``x / y`` stays precise (this fork routes ``/`` through the exact
  path, diverging from upstream Triton's approximate-by-default ``/``).

The tests pin the fast-path contract on normal, abnormal (extreme finite),
denormal, infinity and NaN inputs, and verify the compiled ISA on HCU.
"""

import math

import pytest
import torch

import triton
import triton.language as tl

from triton._internal_testing import is_hip_hcu


@triton.jit
def _fdiv_fast_kernel(OUT, X, Y, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + off)
    y = tl.load(Y + off)
    tl.store(OUT + off, tl.fdiv(x, y, ieee_rounding=False))


@triton.jit
def _fdiv_precise_kernel(OUT, X, Y, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + off)
    y = tl.load(Y + off)
    tl.store(OUT + off, tl.fdiv(x, y, ieee_rounding=True))


@triton.jit
def _div_rn_kernel(OUT, X, Y, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + off)
    y = tl.load(Y + off)
    tl.store(OUT + off, tl.math.div_rn(x, y))


@triton.jit
def _div_default_kernel(OUT, X, Y, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + off)
    y = tl.load(Y + off)
    tl.store(OUT + off, x / y)


def _run(kernel, x, y, BLOCK=256):
    n = x.numel()
    grid = triton.cdiv(n, BLOCK)
    total = grid * BLOCK
    xp = torch.empty(total, dtype=torch.float32, device=x.device)
    yp = torch.empty(total, dtype=torch.float32, device=x.device)
    xp[:n] = x
    xp[n:] = 1.0
    yp[:n] = y
    yp[n:] = 1.0
    out = torch.empty(total, dtype=torch.float32, device=x.device)
    kernel[(grid,)](out, xp, yp, BLOCK=BLOCK)
    return out[:n]


def _assert_same_bits(a, b, msg=""):
    """Bitwise compare incl. NaN positions and signed zeros."""
    assert a.shape == b.shape and a.dtype == b.dtype
    ai = a.view(torch.int32)
    bi = b.view(torch.int32)
    assert (ai == bi).all().item(), (
        f"{msg} bitwise mismatch\n{a}\nvs\n{b}"
    )


def _nan_mask(t):
    return torch.isnan(t)


def _assert_same_semantics(a, b, msg=""):
    """IEEE semantics: NaN positions equal, other lanes bitwise equal."""
    assert a.shape == b.shape and a.dtype == b.dtype
    nan_a, nan_b = torch.isnan(a), torch.isnan(b)
    assert (nan_a == nan_b).all().item(), f"{msg} NaN mismatch"
    ok = ~nan_a
    _assert_same_bits(a[ok], b[ok], msg)


def _assert_fast_semantics(fast, ref, msg=""):
    """Fast-path contract vs IEEE reference.

    NaN positions must match; lanes whose IEEE result is NaN/inf/±0 must be
    bitwise equal (signed zeros matter); remaining finite lanes may be off
    by the rcp approximation (~1 ulp, well under 1e-6 relative).
    """
    assert fast.shape == ref.shape and fast.dtype == ref.dtype
    nan_f, nan_r = torch.isnan(fast), torch.isnan(ref)
    assert (nan_f == nan_r).all().item(), f"{msg} NaN mismatch"
    special = nan_r | torch.isinf(ref) | (ref == 0)
    _assert_same_bits(fast[special], ref[special], f"{msg} special lanes")
    torch.testing.assert_close(
        fast[~special],
        ref[~special],
        rtol=1e-6,
        atol=0.0,
        equal_nan=True,
        msg=f"{msg} finite lanes",
    )


def _special_value_matrix(device):
    """Normal, abnormal (extreme finite), denormal, zero, inf, nan values."""
    fi = torch.finfo(torch.float32)
    vals = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.1,
        4.6875,
        0.9375,
        float(fi.max),
        -float(fi.max),
        float(fi.tiny),          # smallest positive normal
        -float(fi.tiny),
        1.0e-30,
        1.0e30,
        2.0**-149,               # smallest positive denormal
        1.0e-38,                 # denormal (just below tiny)
        float("inf"),
        float("-inf"),
        float("nan"),
        float("-nan"),
    ]
    return torch.tensor(vals, dtype=torch.float32, device=device)


def _pairs_from_matrix(v):
    n = v.numel()
    x = v.repeat_interleave(n)
    y = v.repeat(n)
    return x, y


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_fdiv_fast_normal_values(device):
    """Fast path on normals: small relative error; precise paths bit-exact."""
    torch.manual_seed(0)
    n = 1 << 18
    x = (torch.randn(n, device=device) * 10 + 1e-2).to(torch.float32)
    y = (torch.randn(n, device=device).abs() + 1e-2).to(torch.float32)

    fast = _run(_fdiv_fast_kernel, x, y)
    prec = _run(_fdiv_precise_kernel, x, y)
    rn = _run(_div_rn_kernel, x, y)
    dflt = _run(_div_default_kernel, x, y)
    ref = x / y  # torch IEEE fp32 division

    _assert_same_bits(prec, ref, "precise(tl.fdiv True)")
    _assert_same_bits(rn, ref, "div_rn")
    _assert_same_bits(dflt, ref, "plain /")

    rel = ((fast - prec).abs() / prec.abs()).max().item()
    assert rel <= 1e-6, f"fast path rel error {rel:.3e} > 1e-6"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_fdiv_fast_abnormal_extreme_values(device):
    """Extreme finite values: overflow/underflow plus the rcp-denormal hole.

    ``v_rcp_f32`` flushes *denormal results* to zero, so when ``1/|y|`` is
    below the smallest normal (``|y| > 2**126``) the reciprocal underflows
    to zero and ``max / max`` evaluates to ``0.0`` instead of ``1.0`` on
    the fast path. The precise path keeps the IEEE result.
    """
    fi = torch.finfo(torch.float32)
    cases = [
        (float(fi.max), 0.5),      # overflow -> inf
        (1.0e30, 1.0e-30),         # overflow -> inf
        (float(fi.max), float(fi.max)),  # exactly 1.0
        (1.0e-30, 1.0e30),         # underflow -> 0
        (float(fi.tiny), 2.0),     # denormal result, preserved
        (1.0, float(fi.max)),      # 1/max is denormal -> flushed to 0
        (-1.0, float(fi.max)),
        (float(fi.max), 2.0),
    ]
    x = torch.tensor([c[0] for c in cases] + [1.0] * 3, dtype=torch.float32, device=device)
    y = torch.tensor([c[1] for c in cases] + [1.0] * 3, dtype=torch.float32, device=device)
    fast = _run(_fdiv_fast_kernel, x, y, BLOCK=8)
    prec = _run(_fdiv_precise_kernel, x, y, BLOCK=8)
    ref = x / y
    _assert_same_bits(prec, ref, "precise extreme")
    # Documented fast-path hole: denormal reciprocal results are flushed.
    assert (fast[2] == 0.0).item() and (ref[2] == 1.0).item()
    assert (fast[5] == 0.0).item() and (ref[5] == 2.9387358770557188e-39).item()
    assert (fast[6] == -0.0).item() and (ref[6] == -2.9387358770557188e-39).item()
    # All other lanes keep IEEE semantics on the fast path.
    keep = torch.ones(11, dtype=torch.bool, device=device)
    keep[2] = keep[5] = keep[6] = False
    _assert_fast_semantics(fast[keep], ref[keep], "fast extreme")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_fdiv_fast_denormal_operands(device):
    """Pin the documented fast-path denormal contract.

    ``v_rcp_f32`` does not support denormal inputs (LLVM AMDGPU comment),
    so a denormal divisor is flushed to zero before the reciprocal:
    1 / 1e-38 -> inf on the fast path, while IEEE gives ~1.0e+38.
    Denormal *results* of the multiply are preserved, and denormal
    numerators still divide correctly.
    """
    x = torch.tensor(
        [1.0, 1.0, 2.0**-149, 0.0, 2.0**-149, 1.0],
        dtype=torch.float32,
        device=device,
    )
    y = torch.tensor(
        [1.0e-38, 2.0**-149, 2.0, 1.0e-38, 2.0**-149, 1.0],
        dtype=torch.float32,
        device=device,
    )
    fast = _run(_fdiv_fast_kernel, x, y, BLOCK=4)
    prec = _run(_fdiv_precise_kernel, x, y, BLOCK=4)
    ref = x / y

    # IEEE path must match torch exactly on denormal lanes too.
    _assert_same_bits(prec, ref, "precise denormal")
    # Fast path: divisor flushed to zero -> overflow to inf.
    assert torch.isinf(fast[0]).item() and fast[0] > 0
    assert not torch.isinf(ref[0]).item(), "IEEE 1/1e-38 is finite"
    # 0 / denormal: flushed divisor gives 0 * inf = NaN (IEEE gives 0).
    assert torch.isnan(fast[3]).item()
    assert ref[3].item() == 0.0
    # denormal / denormal: flushed divisor -> inf (IEEE gives 1.0).
    assert torch.isinf(fast[4]).item() and fast[4] > 0
    assert ref[4].item() == 1.0
    # Min denormal / 2 underflows to 0 on both paths (below representable).
    _assert_same_bits(fast[1:3], ref[1:3], "fast denormal rest")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_fdiv_fast_special_values(device):
    """±0, ±inf, NaN: fast path keeps IEEE special-value semantics."""
    v = _special_value_matrix(device)
    x, y = _pairs_from_matrix(v)
    fast = _run(_fdiv_fast_kernel, x, y)
    prec = _run(_fdiv_precise_kernel, x, y)
    rn = _run(_div_rn_kernel, x, y)
    dflt = _run(_div_default_kernel, x, y)
    ref = x / y

    _assert_same_bits(prec, ref, "precise specials")
    _assert_same_bits(rn, ref, "div_rn specials")
    _assert_same_bits(dflt, ref, "plain / specials")
    # Fast path must agree on NaN/inf/zero semantics; only denormal
    # lanes may diverge (covered by test_fdiv_fast_denormal_operands).
    tiny = torch.finfo(torch.float32).tiny
    denorm = ((x.abs() > 0) & (x.abs() < tiny)) | ((y.abs() > 0) & (y.abs() < tiny))
    # Fast path also flushes denormal *reciprocal results*: |y| > 2**126
    # makes 1/|y| denormal, so exclude those lanes here (see
    # test_fdiv_fast_abnormal_extreme_values).
    rcp_underflow = y.abs() > 2.0**126
    keep = ~(denorm | rcp_underflow)
    _assert_fast_semantics(fast[keep], ref[keep], "fast specials")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_fdiv_fast_exact_quotient_boundary(device):
    """4.6875 / 0.9375 == 5.0 exactly: precise must be bit-exact, fast ≤1 ulp.

    This is the regression case from triton-lang/triton#10663: an
    approximate divide can be off by 1 ulp right at an exact boundary.
    """
    x = torch.full((8,), 4.6875, dtype=torch.float32, device=device)
    y = torch.full((8,), 0.9375, dtype=torch.float32, device=device)
    fast = _run(_fdiv_fast_kernel, x, y, BLOCK=8)
    prec = _run(_fdiv_precise_kernel, x, y, BLOCK=8)
    ref = x / y
    _assert_same_bits(prec, ref, "precise exact quotient")
    assert (prec == 5.0).all().item()
    assert (fast - 5.0).abs().max().item() <= 1.0e-6


@pytest.mark.skipif(not is_hip_hcu(), reason="HCU/AMDGPU ISA check")
def test_fdiv_fast_isa(device):
    """Fast path compiles to v_rcp+v_mul; precise path keeps the IEEE chain."""

    @triton.jit
    def kernel(OUT, X, Y, BLOCK: tl.constexpr, FAST: tl.constexpr):
        off = tl.arange(0, BLOCK)
        x = tl.load(X + off)
        y = tl.load(Y + off)
        if FAST:
            z = tl.fdiv(x, y, ieee_rounding=False)
        else:
            z = tl.fdiv(x, y, ieee_rounding=True)
        tl.store(OUT + off, z)

    x = torch.randn(64, dtype=torch.float32, device=device)
    y = torch.randn(64, dtype=torch.float32, device=device).abs() + 0.1
    out = torch.empty_like(x)

    compiled = kernel.warmup(out, x, y, BLOCK=64, FAST=True, grid=(1,))
    asm = compiled.asm["amdgcn"]
    assert "v_rcp_f32" in asm, "fast path should use v_rcp_f32"
    assert "v_div_fixup_f32" not in asm, "fast path must skip IEEE div chain"

    compiled = kernel.warmup(out, x, y, BLOCK=64, FAST=False, grid=(1,))
    asm = compiled.asm["amdgcn"]
    assert "v_div_fixup_f32" in asm, "precise path keeps IEEE div chain"
