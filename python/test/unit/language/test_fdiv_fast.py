"""tl.fdiv vs tl.fdiv_fast semantics tests.

Path map on AMD/HCU:

  - plain ``/`` (``create_fdiv``, no fastmath)  -> IEEE exact chain
  - ``tl.fdiv(x, y)`` -> ``create_fdiv`` -> IEEE exact chain. The legacy
    ``ieee_rounding`` keyword is accepted for API compatibility and ignored:
    ``tl.fdiv(x, y, ieee_rounding=True)`` and
    ``tl.fdiv(x, y, ieee_rounding=False)`` behave identically.
  - ``tl.math.div_rn`` -> ``PreciseDivFOp`` -> IEEE exact
  - ``tl.fdiv_fast(x, y)`` -> ``create_fdiv_fast`` (afn) ->
    ``llvm.amdgcn.fdiv.fast`` (2.5 ulp; denormal inputs flushed to zero)

The precise paths must match torch bitwise; the fast path uses a 2.5 ulp
tolerance plus the documented denormal contract.
"""

import pytest
import torch
import triton
import triton.language as tl


@pytest.fixture
def device():
    return "cuda"


def _run(kernel, x, y, BLOCK=256):
    n = x.numel()
    grid = triton.cdiv(n, BLOCK)
    total = grid * BLOCK
    xp = torch.empty(total, dtype=x.dtype, device=x.device)
    yp = torch.empty(total, dtype=x.dtype, device=x.device)
    xp[:n] = x
    xp[n:] = 1.0
    yp[:n] = y
    yp[n:] = 1.0
    out = torch.empty(total, dtype=x.dtype, device=x.device)
    kernel[(grid,)](out, xp, yp, BLOCK=BLOCK)
    return out[:n]


def _assert_same_bits(a, b, msg=""):
    assert a.shape == b.shape and a.dtype == b.dtype
    int_dtype = torch.int64 if a.dtype == torch.float64 else torch.int32
    ai = a.view(int_dtype)
    bi = b.view(int_dtype)
    assert (ai == bi).all().item(), f"{msg} bitwise mismatch\n{a}\nvs\n{b}"


def _assert_fast_semantics(fast, ref, msg=""):
    """Fast-path contract vs IEEE reference.

    NaN positions must match; lanes whose IEEE result is NaN/inf/±0 must be
    bitwise equal (signed zeros matter); remaining finite lanes may be off by
    the fdiv.fast approximation (2.5 ulp).
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


@triton.jit
def _fdiv_kernel(OUT, X, Y, BLOCK: tl.constexpr):
    off = tl.arange(0, BLOCK)
    x = tl.load(X + off)
    y = tl.load(Y + off)
    z = tl.fdiv(x, y)
    tl.store(OUT + off, z)


@triton.jit
def _fdiv_ieee_true_kernel(OUT, X, Y, BLOCK: tl.constexpr):
    off = tl.arange(0, BLOCK)
    x = tl.load(X + off)
    y = tl.load(Y + off)
    z = tl.fdiv(x, y, ieee_rounding=True)
    tl.store(OUT + off, z)


@triton.jit
def _fdiv_ieee_false_kernel(OUT, X, Y, BLOCK: tl.constexpr):
    off = tl.arange(0, BLOCK)
    x = tl.load(X + off)
    y = tl.load(Y + off)
    z = tl.fdiv(x, y, ieee_rounding=False)
    tl.store(OUT + off, z)


@triton.jit
def _fdiv_fast_kernel(OUT, X, Y, BLOCK: tl.constexpr):
    off = tl.arange(0, BLOCK)
    x = tl.load(X + off)
    y = tl.load(Y + off)
    z = tl.fdiv_fast(x, y)  # create_fdiv_fast -> fdiv.fast
    tl.store(OUT + off, z)


@triton.jit
def _div_rn_kernel(OUT, X, Y, BLOCK: tl.constexpr):
    off = tl.arange(0, BLOCK)
    x = tl.load(X + off)
    y = tl.load(Y + off)
    z = tl.math.div_rn(x, y)
    tl.store(OUT + off, z)


@triton.jit
def _div_default_kernel(OUT, X, Y, BLOCK: tl.constexpr):
    off = tl.arange(0, BLOCK)
    x = tl.load(X + off)
    y = tl.load(Y + off)
    z = x / y  # plain `/` -> create_fdiv -> IEEE exact
    tl.store(OUT + off, z)


def _rand_inputs(device, n=256):
    torch.manual_seed(0)
    x = (torch.randn(n, device=device) * 100).abs() + 0.1
    y = (torch.randn(n, device=device) * 100).abs() + 0.1
    return x, y


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_fdiv_matches_torch(device):
    x, y = _rand_inputs(device)
    out = _run(_fdiv_kernel, x, y)
    _assert_same_bits(out, x / y, "fdiv()")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_fdiv_ieee_rounding_ignored(device):
    """The legacy ieee_rounding keyword must not change tl.fdiv semantics."""
    x, y = _rand_inputs(device)
    out_t = _run(_fdiv_ieee_true_kernel, x, y)
    out_f = _run(_fdiv_ieee_false_kernel, x, y)
    ref = x / y
    _assert_same_bits(out_t, ref, "fdiv(ieee_rounding=True)")
    _assert_same_bits(out_f, ref, "fdiv(ieee_rounding=False)")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_plain_div_matches_torch(device):
    x, y = _rand_inputs(device)
    out = _run(_div_default_kernel, x, y)
    _assert_same_bits(out, x / y, "plain /")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_div_rn_matches_torch(device):
    x, y = _rand_inputs(device)
    out = _run(_div_rn_kernel, x, y)
    _assert_same_bits(out, x / y, "div_rn")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_fdiv_fast_normal_values(device):
    x, y = _rand_inputs(device, 256)
    fast = _run(_fdiv_fast_kernel, x, y)
    ref = x / y
    _assert_fast_semantics(fast, ref, "fast normal")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_fdiv_fast_abnormal_extreme_values(device):
    """fdiv.fast carries a denormal-result guard: max/max and 1/max are
    preserved bitwise instead of being flushed to zero."""
    fi = torch.finfo(torch.float32)
    cases = [
        (float(fi.max), 0.5),
        (1.0e30, 1.0e-30),
        (float(fi.max), float(fi.max)),
        (1.0e-30, 1.0e30),
        (float(fi.tiny), 2.0),
        (1.0, float(fi.max)),
        (-1.0, float(fi.max)),
        (float(fi.max), 2.0),
    ]
    x = torch.tensor([c[0] for c in cases], dtype=torch.float32, device=device)
    y = torch.tensor([c[1] for c in cases], dtype=torch.float32, device=device)
    fast = _run(_fdiv_fast_kernel, x, y, BLOCK=8)
    prec = _run(_fdiv_kernel, x, y, BLOCK=8)
    ref = x / y
    _assert_same_bits(prec, ref, "precise extreme")
    _assert_same_bits(fast[2], ref[2], "fast max/max")
    _assert_same_bits(fast[5], ref[5], "fast 1/max")
    _assert_same_bits(fast[6], ref[6], "fast -1/max")
    keep = torch.ones(8, dtype=torch.bool, device=device)
    keep[2] = keep[5] = keep[6] = False
    _assert_fast_semantics(fast[keep], ref[keep], "fast extreme")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_fdiv_fast_denormal_operands(device):
    """fdiv.fast's v_rcp does not support denormal inputs: a denormal divisor
    is flushed to zero, giving inf/NaN. The precise path keeps IEEE results."""
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
    fast = _run(_fdiv_fast_kernel, x, y, BLOCK=8)
    prec = _run(_fdiv_kernel, x, y, BLOCK=8)
    ref = x / y
    _assert_same_bits(prec, ref, "precise denormal")
    assert torch.isinf(fast[0]).item() and fast[0] > 0
    assert torch.isnan(fast[3]).item()
    assert torch.isinf(fast[4]).item() and fast[4] > 0


def _special_value_matrix(device):
    vals = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.1,
        4.6875,
        0.9375,
        torch.finfo(torch.float32).max,
        torch.finfo(torch.float32).tiny,
        torch.tensor(float("inf"), device=device),
        torch.tensor(float("-inf"), device=device),
        torch.tensor(float("nan"), device=device),
    ]
    return torch.tensor([float(v) for v in vals], dtype=torch.float32, device=device)


def _special_value_matrix_f64(device):
    vals = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.1,
        4.6875,
        0.9375,
        torch.finfo(torch.float64).max,
        torch.finfo(torch.float64).tiny,
        torch.tensor(float("inf"), device=device),
        torch.tensor(float("-inf"), device=device),
        torch.tensor(float("nan"), device=device),
    ]
    return torch.tensor([float(v) for v in vals], dtype=torch.float64, device=device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_fdiv_special_values(device):
    """±0, ±inf, NaN: precise paths bitwise, fast path keeps NaN/inf/±0
    semantics (only denormal lanes may diverge)."""
    v = _special_value_matrix(device)
    x, y = torch.meshgrid(v, v, indexing="ij")
    x = x.reshape(-1)
    y = y.reshape(-1)
    ref = x / y
    for name, kernel, precise in (
        ("fdiv", _fdiv_kernel, True),
        ("plain", _div_default_kernel, True),
        ("div_rn", _div_rn_kernel, True),
        ("fast", _fdiv_fast_kernel, False),
    ):
        out = _run(kernel, x, y)
        if precise:
            nan_o, nan_r = torch.isnan(out), torch.isnan(ref)
            assert (nan_o == nan_r).all().item(), f"{name} NaN mismatch"
            _assert_same_bits(out[~nan_r], ref[~nan_r], f"{name} specials")
        else:
            tiny = torch.finfo(torch.float32).tiny
            denorm = ((x.abs() > 0) & (x.abs() < tiny)) | (
                (y.abs() > 0) & (y.abs() < tiny)
            )
            keep = ~denorm
            _assert_fast_semantics(out[keep], ref[keep], f"{name} specials")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_fdiv_f64_special_values(device):
    """f64 has no fast division (llvm.amdgcn.fdiv.fast is f32-only, mirroring
    NVIDIA's always-exact div.rn.d): both tl.fdiv and tl.fdiv_fast lower to
    the IEEE exact chain, so special values are bitwise identical to torch."""
    v = _special_value_matrix_f64(device)
    x, y = torch.meshgrid(v, v, indexing="ij")
    x = x.reshape(-1)
    y = y.reshape(-1)
    ref = x / y
    for name, kernel in (
        ("fdiv", _fdiv_kernel),
        ("plain", _div_default_kernel),
        ("fast", _fdiv_fast_kernel),
    ):
        out = _run(kernel, x, y)
        nan_o, nan_r = torch.isnan(out), torch.isnan(ref)
        assert (nan_o == nan_r).all().item(), f"f64 {name} NaN mismatch"
        _assert_same_bits(out[~nan_r], ref[~nan_r], f"f64 {name} specials")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_fdiv_fast_isa(device):
    """Fast path compiles to fdiv.fast; precise path keeps the IEEE chain."""

    @triton.jit
    def kernel(OUT, X, Y, BLOCK: tl.constexpr, FAST: tl.constexpr):
        off = tl.arange(0, BLOCK)
        x = tl.load(X + off)
        y = tl.load(Y + off)
        if FAST:
            z = tl.fdiv_fast(x, y)
        else:
            z = tl.fdiv(x, y)
        tl.store(OUT + off, z)

    x = torch.randn(64, dtype=torch.float32, device=device)
    y = torch.randn(64, dtype=torch.float32, device=device).abs() + 0.1
    out = torch.empty_like(x)

    compiled = kernel.warmup(out, x, y, BLOCK=64, FAST=True, grid=(1,))
    asm = compiled.asm["amdgcn"]
    assert "v_cmp_gt_f32" in asm or "v_cndmask_b32" in asm, "fast path fdiv.fast"
    assert "v_rcp_f32" in asm, "fast path should use v_rcp_f32"
    assert "v_div_fixup_f32" not in asm, "fast path must skip IEEE div chain"

    compiled = kernel.warmup(out, x, y, BLOCK=64, FAST=False, grid=(1,))
    asm = compiled.asm["amdgcn"]
    assert "v_div_fixup_f32" in asm, "precise path keeps IEEE div chain"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_plain_div_isa_exact(device):
    """Plain `/` must NOT take the fast path: it stays on the IEEE chain."""

    @triton.jit
    def kernel(OUT, X, Y, BLOCK: tl.constexpr):
        off = tl.arange(0, BLOCK)
        x = tl.load(X + off)
        y = tl.load(Y + off)
        z = x / y
        tl.store(OUT + off, z)

    x = torch.randn(64, dtype=torch.float32, device=device)
    y = torch.randn(64, dtype=torch.float32, device=device).abs() + 0.1
    out = torch.empty_like(x)
    compiled = kernel.warmup(out, x, y, BLOCK=64, grid=(1,))
    asm = compiled.asm["amdgcn"]
    assert "v_div_fixup_f32" in asm, "plain / must keep the IEEE div chain"
