"""Main gfx946 XCD launch checks, shared by hardware and PMD.

Run inside the configured hardware/model environment, in a fresh working
directory and with a fresh TRITON_CACHE_DIR. Select the environment BEFORE
starting Python; the test does not load or replace the installed HIP runtime.
The missing-extension compatibility case instead uses an isolated host-only
HIP stub with the generated launcher; it needs a C compiler, not a GPU.

    # Model (four XCDs); GPU_CHIP=sb also covers the single-XCD model.
    PMD_PATH=/opt/rocm/pmd GPU_CHIP=sbx4 GPU_ARGS='[]' \
        TRITON_FAST_JIT=0 TRITON_HCUTUNE=0 python3 -m pytest -q \
        /zhenxin/triton/third_party/hcu/test/test_xcd_gfx946.py

    # Hardware: use the hardware HIP runtime/toolchain, without PMD_PATH.
    env -u PMD_PATH TRITON_FAST_JIT=0 TRITON_HCUTUNE=0 python3 -m pytest -q \
        /zhenxin/triton/third_party/hcu/test/test_xcd_gfx946.py

Repeat with TRITON_FAST_JIT=1 in a new process/cache to cover Fast JIT.
Unsetting PMD_PATH alone cannot turn a model-only installation into hardware.
No autotune candidates or policy are tested. All inputs and golden results
are computed on CPU; outputs are copied back for comparison.

These checks cover launch delivery and numerical correctness, NOT physical
XCD placement or L2 performance. The model does not implement mask filtering;
the mask launch below checks numerical correctness only, not affinity.
Optional model logs: GPU_DFLAGS="['StatLog', 'IT', 'LdsI']"; hardware counters
and physical placement require separate observation, not pid % num_xcds.
"""

import ast
import ctypes
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import triton
import triton.language as tl
from triton.backends.hcu import driver as hcu_driver
from triton.backends.hcu.driver import HCUDriver, _pack_xcd_metadata
from triton.backends.hcu.xcd import XCDLaunchMetadata
from triton.language.extra.hcu.xcd_remap import remap_xcd, remap_xcd_chunked, remap_xcd_fixed
from triton.runtime.driver import driver
from triton.runtime.jit import separate_xcd_metadata


@pytest.fixture
def gfx946():
    if not torch.cuda.is_available():
        pytest.skip("requires a gfx946 device or PMD runtime")
    if not isinstance(driver.active, HCUDriver) or driver.active.get_current_target().arch != "gfx946":
        pytest.skip("requires the HCU gfx946 backend")
    if not os.environ.get("PMD_PATH"):
        return None  # Hardware: do not infer its topology from GPU_CHIP.
    assert Path(os.environ["PMD_PATH"]).is_dir(), "PMD_PATH must name the installed model directory"
    chip = os.environ.get("GPU_CHIP")
    assert chip in ("sb", "sbx4"), "model tests require stock GPU_CHIP=sb or sbx4"
    assert driver.active.get_current_device() == 0, "model tests use device 0"
    args = ast.literal_eval(os.environ.get("GPU_ARGS", "[]"))
    assert isinstance(args, list) and all(isinstance(arg, str) for arg in args)
    assert not any(arg.split("=", 1)[0] in ("--useDispConfig", "--num-chip", "--num-xcd") for arg in args), \
        "GPU_ARGS must not override per-launch dispatch or stock topology"
    return chip


def test_metadata_encoding_and_separation():
    for metadata, expected in [
        (XCDLaunchMetadata(), (1, 0, 1, 0)),
        (XCDLaunchMetadata(die_mask=5, chunk_size=2), (1, 5, 2, 0)),
        (XCDLaunchMetadata(mode="fixed"), (2, 0, 0, 0)),
        (XCDLaunchMetadata(mode="block", block_x=2, block_y=2), (0, 0, 2, 2)),
    ]:
        assert _pack_xcd_metadata(metadata) == expected
        kwargs = {"xcd_metadata": metadata, "num_warps": 4}
        assert separate_xcd_metadata(kwargs, ["input", "output"]) is metadata
        assert kwargs == {"num_warps": 4}
    # A same-named kernel argument must not be consumed as launch metadata.
    kwargs = {"xcd_metadata": 42}
    assert separate_xcd_metadata(kwargs, ["xcd_metadata"]) is None
    assert kwargs == {"xcd_metadata": 42}
    for kwargs in ({}, {"xcd_metadata": None}):
        assert separate_xcd_metadata(kwargs, []) is None
        assert kwargs == {}


@pytest.mark.parametrize("fields,error", [
    ({"mode": "unknown"}, ValueError),
    ({"die_mask": -1}, ValueError),
    ({"chunk_size": 0}, ValueError),
    ({"chunk_size": True}, TypeError),
    ({"mode": "fixed", "die_mask": 1}, ValueError),
    ({"mode": "linear", "block_x": 2}, ValueError),
    ({"mode": "block", "chunk_size": 2}, ValueError),
    ({"die_mask": 16}, ValueError),
    ({"chunk_size": 1 << 31}, ValueError),
    ({"mode": "block", "block_y": 2048}, ValueError),
])
def test_invalid_metadata(fields, error):
    with pytest.raises(error):
        _pack_xcd_metadata(XCDLaunchMetadata(**fields))


@pytest.mark.parametrize("with_extension", [False, True], ids=["missing", "present"])
def test_launch_with_optional_xcd_runtime_extension(monkeypatch, with_extension):
    """Exercise the real launcher and header ABI with a host-only HIP stub.

    Only dynamic-library lookup and HIP execution are stubbed in this test's
    translation unit. This checks compatibility branches, not an old SDK ABI
    or real device execution. No installed runtime or production code changes.
    """
    stub = r"""
#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

_Static_assert(sizeof(hipLaunchMultiDieConfig) == 16, "MultiDie config size");
_Static_assert(_Alignof(hipLaunchMultiDieConfig) == 4, "MultiDie config alignment");
_Static_assert(offsetof(hipLaunchMultiDieConfig, blockMode.mode) == 0, "mode offset");
_Static_assert(offsetof(hipLaunchMultiDieConfig, blockMode.dieMask) == 4, "mask offset");
_Static_assert(offsetof(hipLaunchMultiDieConfig, blockMode.numMcmBlockX) == 8, "block X offset");
_Static_assert(offsetof(hipLaunchMultiDieConfig, blockMode.numMcmBlockY) == 12, "block Y offset");
_Static_assert(offsetof(hipLaunchMultiDieConfig, linearMode.chunkSize) == 8, "chunk offset");

static const char *stub_error;
static hipError_t stub_last_error(void) { return hipSuccess; }
static const char *stub_error_string(hipError_t error) { return "unexpected HIP stub call"; }
static hipError_t stub_pointer(void *out, hipPointer_attribute attr, hipDeviceptr_t ptr) {
    return hipErrorInvalidValue;
}
static hipError_t stub_launch(hipFunction_t function,
    unsigned int gx, unsigned int gy, unsigned int gz,
    unsigned int bx, unsigned int by, unsigned int bz,
    unsigned int shared, hipStream_t stream, void **params, void **extra) {
    if (gx != 2 || gy != 1 || gz != 1 || bx != 64 || by != 1 || bz != 1)
        return hipErrorInvalidValue;
    // The test passes a host address as an integer, never a device pointer.
    int *calls = (int *)(uintptr_t)*(hipDeviceptr_t *)params[0];
    ++*calls;
    return hipSuccess;
}
static hipError_t stub_multidie(hipFunction_t function,
    unsigned int gx, unsigned int gy, unsigned int gz,
    unsigned int bx, unsigned int by, unsigned int bz,
    unsigned int shared, hipStream_t stream, void **params, void **extra,
    hipLaunchMultiDieConfig *config) {
    hipError_t result = stub_launch(function, gx, gy, gz, bx, by, bz, shared, stream, params, extra);
    if (result != hipSuccess) return result;
    unsigned int *output = (unsigned int *)(uintptr_t)*(hipDeviceptr_t *)params[0];
    memcpy(output + 1, config, sizeof(*config));
    return hipSuccess;
}
_Static_assert(__builtin_types_compatible_p(__typeof__(&stub_multidie),
    __typeof__(&hipModuleLaunchKernelMultiDie)), "MultiDie function signature");
static hipError_t stub_capture(hipStream_t stream, hipStreamCaptureStatus *status) {
    *status = hipStreamCaptureStatusNone;
    return hipSuccess;
}
static hipError_t stub_get_proc(const char *name, void **ptr, int version,
    uint64_t flags, hipDriverProcAddressQueryResult *status) {
    *ptr = NULL;
    *status = (hipDriverProcAddressQueryResult)0;
    if (!strcmp(name, "hipGetLastError")) *ptr = (void *)stub_last_error;
    else if (!strcmp(name, "hipGetErrorString")) *ptr = (void *)stub_error_string;
    else if (!strcmp(name, "hipPointerGetAttribute")) *ptr = (void *)stub_pointer;
    else if (!strcmp(name, "hipModuleLaunchKernel") ||
             !strcmp(name, "hipModuleLaunchCooperativeKernel")) *ptr = (void *)stub_launch;
    return *ptr ? hipSuccess : hipErrorNotSupported;
}
static void *stub_dlopen(const char *path, int flags) { return (void *)1; }
static int stub_dlclose(void *handle) { return 0; }
static char *stub_dlerror(void) {
    const char *error = stub_error;
    stub_error = NULL;
    return (char *)error;
}
static void *stub_dlsym(void *handle, const char *name) {
    if (!strcmp(name, "hipGetProcAddress")) return (void *)stub_get_proc;
    if (STUB_HAS_XCD) {
        if (!strcmp(name, "hipModuleLaunchKernelMultiDie")) return (void *)stub_multidie;
        if (!strcmp(name, "hipStreamIsCapturing")) return (void *)stub_capture;
    }
    // Preserve dlerror semantics when an optional entry point is absent.
    stub_error = "symbol absent from test runtime";
    return NULL;
}
#define dlopen stub_dlopen
#define dlclose stub_dlclose
#define dlerror stub_dlerror
#define dlsym stub_dlsym
"""
    compile_module = hcu_driver.compile_module_from_src

    def compile_with_stub(src, **kwargs):
        return compile_module(src=f"#define STUB_HAS_XCD {int(with_extension)}\n" + stub + src, **kwargs)

    monkeypatch.setattr(hcu_driver, "_get_path_to_hip_runtime_dylib", lambda: "test-only-hip-stub")
    monkeypatch.setattr(hcu_driver, "compile_module_from_src", compile_with_stub)
    launcher = hcu_driver.HCULauncher(
        SimpleNamespace(constants={}, signature={(0,): "*i32"}),
        SimpleNamespace(warp_size=64, launch_cooperative_grid=False,
                        profile_scratch_size=0, profile_scratch_align=1),
    )
    calls = ctypes.c_int(0)
    args = (2, 1, 1, 0, 0, (1, 1, 0), None, None, None, ctypes.addressof(calls))
    launcher(*args)
    assert calls.value == 1
    launcher(*args, xcd_metadata=None)
    assert calls.value == 2

    requests = [
        (XCDLaunchMetadata(), (1, 0, 1, 0)),
        (XCDLaunchMetadata(die_mask=5, chunk_size=2), (1, 5, 2, 0)),
        (XCDLaunchMetadata(mode="block", die_mask=3, block_x=2, block_y=4), (0, 3, 2, 4)),
        (XCDLaunchMetadata(mode="fixed"), (2, 0, 0, 0)),
    ]
    for metadata, expected in requests:
        output = (ctypes.c_uint * 5)()
        xcd_args = (*args[:-1], ctypes.addressof(output))
        if with_extension:
            launcher(*xcd_args, xcd_metadata=metadata)
            assert tuple(output) == (1, *expected)
        else:
            with pytest.raises(RuntimeError, match="missing hipModuleLaunchKernelMultiDie"):
                launcher(*xcd_args, xcd_metadata=metadata)
            assert tuple(output) == (0, 0, 0, 0, 0)


@triton.jit
def _probe(input, output):
    pid = tl.program_id(0) + tl.num_programs(0) * tl.program_id(1)
    tl.store(output + pid, tl.load(input + pid) + pid)


@pytest.mark.parametrize("compiled_entry", [False, True], ids=["jit", "compiled"])
def test_launch_modes_and_binary_reuse(gfx946, compiled_entry):
    cpu = torch.arange(32, dtype=torch.int32) * 3 - 17
    golden = cpu + torch.arange(32, dtype=torch.int32)
    sentinel = torch.full((32,), -1234567, dtype=torch.int32)
    gpu, output = cpu.cuda(), sentinel.cuda()
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()
    compiled = _probe.warmup(gpu, output, grid=(8, 4), num_warps=1) if compiled_entry else None
    # Repeated metadata exercises hot launches; changes must not recompile.
    requests = [
        {"xcd_metadata": XCDLaunchMetadata(chunk_size=2)},
        {"xcd_metadata": XCDLaunchMetadata(chunk_size=2)},
        {"xcd_metadata": XCDLaunchMetadata(mode="block", block_x=2, block_y=2)},
    ]
    if gfx946 != "sb":  # Fixed on the single-XCD model is outside tested scope.
        requests.append({"xcd_metadata": XCDLaunchMetadata(mode="fixed")})
    requests += [{}, {"xcd_metadata": None}, {"xcd_metadata": XCDLaunchMetadata()},
                 {"xcd_metadata": XCDLaunchMetadata(die_mask=1)}]
    hashes = set()
    for kwargs in requests:
        output.copy_(sentinel)
        torch.cuda.synchronize()
        if compiled_entry:
            # Supply a stream different from PyTorch's current stream.
            compiled[(8, 4, 1)](gpu, output, stream=stream.cuda_stream, **kwargs)
            kernel = compiled
        else:
            with torch.cuda.stream(stream):
                kernel = _probe[(8, 4)](gpu, output, num_warps=1, **kwargs)
        stream.synchronize()
        torch.testing.assert_close(output.cpu(), golden, rtol=0, atol=0)
        hashes.add(kernel.metadata.hash)
    assert len(hashes) == 1, "per-launch metadata changed the compiled binary"
    if hasattr(_probe, "saved_kernel_cache"):
        assert len(_probe.saved_kernel_cache) == 1
    else:
        assert len(_probe.device_caches[torch.cuda.current_device()][0]) == 1

    # Empty grid must leave output untouched, including with explicit metadata.
    output.copy_(sentinel)
    torch.cuda.synchronize()
    if compiled_entry:
        compiled[(0, 4, 1)](gpu, output, stream=stream.cuda_stream, xcd_metadata=XCDLaunchMetadata())
    else:
        with torch.cuda.stream(stream):
            _probe[(0, 4)](gpu, output, num_warps=1, xcd_metadata=XCDLaunchMetadata())
    stream.synchronize()
    torch.testing.assert_close(output.cpu(), sentinel, rtol=0, atol=0)


def test_invalid_launch_metadata(gfx946):
    cpu = torch.zeros(32, dtype=torch.int32)
    gpu, output = cpu.cuda(), cpu.cuda()
    compiled = _probe.warmup(gpu, output, grid=(8, 4), num_warps=1)
    for invalid in (False, 0, {}, []):
        with pytest.raises(TypeError, match="xcd_metadata"):
            compiled[(8, 4, 1)](gpu, output, xcd_metadata=invalid)
    with pytest.raises(ValueError, match="grid"):
        compiled[(-1, 4, 1)](gpu, output, xcd_metadata=XCDLaunchMetadata())


@triton.jit(do_not_specialize=["M", "N", "K"])
def _gemm(a, b, output, M, N, K,
          REMAP_FIXED: tl.constexpr = False, NUM_XCDS: tl.constexpr = 4):
    pid_m, pid_n = tl.program_id(1), tl.program_id(0)
    if REMAP_FIXED:
        tiles_m = tl.cdiv(M, 16)
        tiles_n = tl.cdiv(N, 16)
        tile = remap_xcd_fixed(pid_m * tiles_n + pid_n, tiles_m, tiles_n, NUM_XCDS)
        pid_m, pid_n = tile // tiles_n, tile % tiles_n
    mi = pid_m * 16 + tl.arange(0, 16)
    ni = pid_n * 16 + tl.arange(0, 16)
    ki = tl.arange(0, 32)
    acc = tl.full((16, 16), 0, tl.float32)
    for step in range(tl.cdiv(K, 32)):
        k = step * 32 + ki
        av = tl.load(a + mi[:, None] * K + k[None, :], (mi[:, None] < M) & (k[None, :] < K), other=0)
        bv = tl.load(b + k[:, None] * N + ni[None, :], (k[:, None] < K) & (ni[None, :] < N), other=0)
        acc += tl.dot(av, bv)
    tl.store(output + mi[:, None] * N + ni[None, :], acc, (mi[:, None] < M) & (ni[None, :] < N))


@pytest.mark.parametrize("mode", ["default", "linear", "fixed", "block"])
def test_gemm_tail(gfx946, mode):
    if gfx946 == "sb" and mode == "fixed":
        pytest.skip("Fixed is verified only on the four-XCD model")
    # A 4x4 workgroup grid fits Fixed/Block model scope while all axes have tails.
    m, n, k = 49, 49, 35
    a_cpu = ((torch.arange(m * k).reshape(m, k) % 13 - 6).float() / 8).half()
    b_cpu = ((torch.arange(k * n).reshape(k, n) % 11 - 5).float() / 8).half()
    golden = a_cpu.float() @ b_cpu.float()
    a, b = a_cpu.cuda(), b_cpu.cuda()
    output = torch.full((m, n), float("nan"), dtype=torch.float32).cuda()
    metadata = {
        "default": None,
        "linear": XCDLaunchMetadata(chunk_size=2),
        "fixed": XCDLaunchMetadata(mode="fixed"),
        "block": XCDLaunchMetadata(mode="block", block_x=2, block_y=2),
    }[mode]
    _gemm[(4, 4)](a, b, output, m, n, k, num_warps=4, xcd_metadata=metadata)
    torch.testing.assert_close(output.cpu(), golden, rtol=1e-3, atol=1e-3)


@triton.jit
def _remap_probe(input, ids, output, N, NUM_XCDS: tl.constexpr, CHUNK_SIZE: tl.constexpr):
    raw_pid = tl.program_id(0)
    full = remap_xcd(raw_pid, N, NUM_XCDS)
    chunked = remap_xcd_chunked(raw_pid, N, NUM_XCDS, CHUNK_SIZE)
    tl.store(ids + raw_pid, full)
    tl.store(ids + N + raw_pid, chunked)
    # Guard against a broken permutation reading outside input; the CPU checks
    # below will report both the bad tile ID and the wrong gathered value.
    tl.store(output + raw_pid, tl.load(input + full, (full >= 0) & (full < N), other=-1))
    tl.store(output + N + raw_pid, tl.load(input + chunked, (chunked >= 0) & (chunked < N), other=-1))


@pytest.mark.parametrize("n,d,c", [
    (1, 4, 2), (3, 4, 2), (7, 4, 2), (8, 4, 2), (9, 4, 2),
    (31, 4, 2), (32, 4, 2), (33, 4, 2), (17, 8, 3), (17, 1, 2), (17, 4, 1),
])
def test_xcd_remap(gfx946, n, d, c):
    # Independent CPU reference: enumerate raw PIDs in desired tile order,
    # then invert that permutation. D is a logical grouping parameter here;
    # this test does not infer physical placement from the model topology.
    full_order = [pid for owner in range(d) for pid in range(owner, n, d)]
    window = d * c
    tail = n // window * window
    chunk_order = [base + owner + j * d for base in range(0, tail, window)
                   for owner in range(d) for j in range(c)] + list(range(tail, n))
    expected_ids = torch.stack([torch.argsort(torch.tensor(order)) for order in (full_order, chunk_order)])
    cpu = torch.arange(n, dtype=torch.int32) * 3 + 7
    golden = cpu[expected_ids]
    gpu = cpu.cuda()
    ids = torch.full((2, n), -1, dtype=torch.int32).cuda()
    output = torch.full((2, n), -1, dtype=torch.int32).cuda()
    _remap_probe[(n,)](gpu, ids, output, n, d, c, num_warps=1)
    torch.testing.assert_close(ids.cpu(), expected_ids.to(torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(output.cpu(), golden, rtol=0, atol=0)


@triton.jit(do_not_specialize=["M", "N"])
def _fixed_remap_probe(input, ids, output, M, N,
                       NUM_XCDS: tl.constexpr, REMAP: tl.constexpr):
    raw_pid = tl.program_id(0) + tl.num_programs(0) * tl.program_id(1)
    if REMAP:
        tile = remap_xcd_fixed(raw_pid, M, N, NUM_XCDS)
    else:
        tile = raw_pid
    tl.store(ids + raw_pid, tile)
    valid = (tile >= 0) & (tile < M * N)
    value = tl.load(input + tile, valid, other=0)
    tl.store(output + tile, value + tile, valid)


@pytest.mark.parametrize("m,n,d", [
    (1, 1, 1), (3, 5, 1),
    (1, 2, 2), (3, 8, 2), (6, 10, 2), (3, 7, 2), (5, 1, 2),
    (2, 2, 4), (2, 8, 4), (8, 2, 4), (4, 8, 4), (6, 10, 4),
    (3, 8, 4), (4, 7, 4), (3, 5, 4), (5, 3, 4), (1, 1, 4), (1, 7, 4), (7, 1, 4),
    (3, 5, -1), (3, 5, 0), (3, 5, 3), (3, 5, 8),
])
def test_xcd_remap_fixed(gfx946, m, n, d):
    # Independently slice a row-major CPU tile matrix, then interleave region
    # contents to obtain tile IDs in raw PID order. D is a logical grouping
    # parameter here, not a measurement of the runtime's physical topology.
    tiles = torch.arange(m * n, dtype=torch.int32).reshape(m, n)
    # Unsupported group counts skip remapping, just like the one-group case.
    parts_m, parts_n = {2: (1, 2), 4: (2, 2)}.get(d, (1, 1))
    core_m, core_n = m // parts_m * parts_m, n // parts_n * parts_n
    if core_m == 0 or core_n == 0:
        expected_ids = tiles.flatten()
    else:
        regions = [region.flatten() for band in tiles[:core_m, :core_n].chunk(parts_m, dim=0)
                   for region in band.chunk(parts_n, dim=1)]
        expected_ids = torch.cat((torch.stack(regions).T.reshape(-1),
                                  tiles[:core_m, core_n:].flatten(), tiles[core_m:, :].flatten()))
    assert sorted(expected_ids.tolist()) == list(range(m * n))
    cpu = torch.arange(m * n, dtype=torch.int32) * 3 + 7
    golden = cpu + torch.arange(m * n, dtype=torch.int32)
    gpu = cpu.cuda()
    # Only the four-XCD environment has a verified hardware-Fixed comparison.
    modes = (True, False) if d == 4 and gfx946 != "sb" and m % 2 == 0 and n % 2 == 0 else (True,)
    for remap in modes:
        ids = torch.full_like(cpu, -1).cuda()
        output = torch.full_like(cpu, -1).cuda()
        # Software remap pairs with Linear, NOT Fixed. The second launch is
        # an independent hardware-Fixed/no-remap numerical baseline.
        metadata = XCDLaunchMetadata(mode="linear" if remap else "fixed")
        _fixed_remap_probe[(n, m)](gpu, ids, output, m, n, d, remap,
                                   num_warps=1, xcd_metadata=metadata)
        reference_ids = expected_ids if remap else torch.arange(m * n, dtype=torch.int32)
        torch.testing.assert_close(ids.cpu(), reference_ids, rtol=0, atol=0)
        torch.testing.assert_close(output.cpu(), golden, rtol=0, atol=0)


@pytest.mark.parametrize("d", [1, 2, 4])
def test_gemm_fixed_remap_dynamic_shapes(gfx946, d):
    compiled = None
    for m, n, k in [(32, 64, 32), (33, 65, 35), (65, 33, 17), (7, 35, 9), (35, 7, 3)]:
        a_cpu = ((torch.arange(m * k).reshape(m, k) % 13 - 6).float() / 8).half()
        b_cpu = ((torch.arange(k * n).reshape(k, n) % 11 - 5).float() / 8).half()
        golden = a_cpu.float() @ b_cpu.float()
        a, b = a_cpu.cuda(), b_cpu.cuda()
        output = torch.full((m, n), float("nan"), dtype=torch.float32).cuda()
        grid = (triton.cdiv(n, 16), triton.cdiv(m, 16))
        kernel = _gemm[grid](a, b, output, m, n, k, REMAP_FIXED=True, NUM_XCDS=d,
                             num_warps=4, xcd_metadata=XCDLaunchMetadata(mode="linear"))
        torch.testing.assert_close(output.cpu(), golden, rtol=1e-3, atol=1e-3)
        if compiled is None:
            compiled = kernel
        # Fast JIT reloads the saved artifact into a new Python wrapper after
        # its first fallback. Compare the artifact, not wrapper identity.
        assert kernel.metadata_group == compiled.metadata_group
        assert kernel.kernel == compiled.kernel
        # Bypass JIT for a second launch: even the original compiled entry must
        # handle the new shape, rather than silently compiling a specialization.
        output.copy_(torch.full((m, n), float("nan"), dtype=torch.float32))
        # The direct launcher still expects the constexpr argument positions;
        # their values were fixed at compilation, not changed by this call.
        compiled[(grid[0], grid[1], 1)](a, b, output, m, n, k, True, d,
                                       xcd_metadata=XCDLaunchMetadata(mode="linear"))
        torch.testing.assert_close(output.cpu(), golden, rtol=1e-3, atol=1e-3)
