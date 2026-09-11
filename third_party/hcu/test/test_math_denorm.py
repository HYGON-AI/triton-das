import pytest
import torch

import triton
import triton.language as tl


def is_hcu():
    if not torch.cuda.is_available():
        return False
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "hip" and target.arch in ("gfx928", "gfx936", "gfx938", "gfx946", "gfx92a")


pytestmark = pytest.mark.skipif(not is_hcu(), reason="requires an HCU GPU")


@triton.jit
def math_kernel(X, Y, N: tl.constexpr, OP: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    x = tl.load(X + i, i < N, other=1).to(tl.float32)
    if OP == "sqrt":
        y = tl.sqrt(x)
    elif OP == "exp2":
        y = tl.exp2(x)
    elif OP == "rsqrt":
        y = tl.rsqrt(x)
    tl.store(Y + i, y, i < N)


@pytest.mark.parametrize("flush_denorm", [None, False, True])
@pytest.mark.parametrize("op", ["sqrt", "rsqrt", "exp2"])
def test_math_ftz_option(op, flush_denorm):
    # Omitted options retain legacy lowering; explicit False selects non-FTZ.
    host = (torch.tensor([-1., 0., 1.]) if op == "exp2" else
            torch.tensor([1, 2, 0x007fffff, 0x3f800000], dtype=torch.int32).view(torch.float32))
    x = host.cuda()
    out = torch.empty_like(x)
    options = {} if flush_denorm is None else {"allow_flush_denorm": flush_denorm}
    kernel = math_kernel[(1,)](x, out, x.numel(), op, 256, **options)
    assert kernel.metadata.allow_flush_denorm_explicitly_set is (flush_denorm is not None)
    intrinsic = {"sqrt": "sqrt", "rsqrt": "rsq", "exp2": "exp2"}[op]
    if flush_denorm is not False or op == "sqrt":
        assert "llvm.amdgcn." + intrinsic + ".f32" in kernel.asm["llir"]
    if flush_denorm is False and op in ("sqrt", "rsqrt"):
        ref = host.double().sqrt()
        if op == "rsqrt":
            ref = ref.reciprocal()
        torch.testing.assert_close(out.cpu(), ref.float(), rtol=1.2e-7, atol=0)
        return
    if op == "exp2":
        if flush_denorm is False:
            assert "llvm.exp2.f32" in kernel.asm["llir"]
        expected = torch.tensor([0.5, 1., 2.])
    elif op == "sqrt":
        expected = torch.tensor([0., 0., 0., 1.])
    else:
        expected = torch.tensor([float("inf"), float("inf"), float("inf"), 1.])
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


def test_ftz_option_parsing_and_cache():
    from dataclasses import asdict
    from triton.backends.hcu.compiler import HIPBackend
    backend = HIPBackend(triton.runtime.driver.active.get_current_target())
    omitted = backend.parse_options({})
    explicit_false = backend.parse_options({"allow_flush_denorm": False})
    explicit_true = backend.parse_options({"allow_flush_denorm": True})
    assert omitted.allow_flush_denorm is False
    assert omitted.allow_flush_denorm_explicitly_set is False
    assert explicit_false.allow_flush_denorm_explicitly_set is True
    assert explicit_true.allow_flush_denorm_explicitly_set is True
    assert omitted.hash() != explicit_false.hash()
    for opts in (omitted, explicit_false, explicit_true):
        assert backend.parse_options(asdict(opts)).hash() == opts.hash()
