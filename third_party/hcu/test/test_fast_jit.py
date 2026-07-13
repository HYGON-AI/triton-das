import os
import pytest
import json
import torch
import tempfile

os.environ["TRITON_FAST_JIT"] = "1"
import triton
import triton.language as tl
from triton.backends.hcu.jit import (
    FastJITFunction, get_saved_kernel_cache_hash, get_saved_kernel_cache_dir
)
from triton.runtime.jit import JITFunction


@pytest.mark.parametrize('device', ['cuda'])
def test_fast_jit(device):

    @triton.jit
    def add_kernel_utils(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        y = tl.load(in_ptr1 + offsets, mask=mask)
        output = x + y
        tl.store(out_ptr + offsets, output, mask=mask)

    fn = add_kernel_utils

    x = torch.randn(4, device=device)
    y = torch.randn(4, device=device)
    out = torch.zeros_like(x)
    # compile kernel by triton.jit
    fn[(4, )](x, y, out, 4, 4)

    fpath = f"{get_saved_kernel_cache_dir()}/add_kernel_utils-{get_saved_kernel_cache_hash(fn)}.json"
    assert os.path.isfile(fpath)

    with open(fpath) as fp:
        data = json.load(fp)['cache']

    keys = list(data.keys())
    assert len(keys) == 1

    assert keys[0] == "('torch.float32', 'torch.float32', 'torch.float32', 4, 4)"

    out2 = torch.zeros_like(out)
    # launch kernel with triton.jit
    fn[(4, )](x, y, out2, 4, 4)
    triton.testing.assert_close(out2, x + y)


@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('key', [['n_elements', 'BLOCK_SIZE'],
                                 lambda META: [META['n_elements'], META['BLOCK_SIZE']]])
def test_jit_key(device, key):

    @triton.jit(key=key)
    def add_kernel_key(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        y = tl.load(in_ptr1 + offsets, mask=mask)
        output = x + y
        tl.store(out_ptr + offsets, output, mask=mask)

    fn = add_kernel_key

    x = torch.randn(4, device=device)
    y = torch.randn(4, device=device)
    out = torch.zeros_like(x)
    fn[(4, )](x, y, out, 4, 4)

    fpath = f"{get_saved_kernel_cache_dir()}/add_kernel_key-{get_saved_kernel_cache_hash(fn)}.json"
    assert os.path.isfile(fpath)

    with open(fpath) as fp:
        data = json.load(fp)['cache']

    keys = list(data.keys())
    assert len(keys) == 1

    assert keys[0] == "(4, 4)"


# @pytest.mark.parametrize('device', ['cuda'])
# def test_empty_metadata(monkeypatch, device):
#     monkeypatch.setenv("TRITON_CACHE_DIR", tempfile.mkdtemp())

#     @triton.jit
#     def add_kernel_empty_metadata(
#         in_ptr0,
#         in_ptr1,
#         out_ptr,
#         n_elements,
#         BLOCK_SIZE: "tl.constexpr",
#     ):
#         pid = tl.program_id(axis=0)
#         block_start = pid * BLOCK_SIZE
#         offsets = block_start + tl.arange(0, BLOCK_SIZE)
#         mask = offsets < n_elements
#         x = tl.load(in_ptr0 + offsets, mask=mask)
#         y = tl.load(in_ptr1 + offsets, mask=mask)
#         output = x + y
#         tl.store(out_ptr + offsets, output, mask=mask)

#     fn = add_kernel_empty_metadata

#     x = torch.randn(4, device=device)
#     y = torch.randn(4, device=device)
#     out = torch.zeros_like(x)

#     kern_key = fn.get_key(key={'in_ptr0': str(x.dtype), 'in_ptr1': str(y.dtype),
#                            'out_ptr': str(out.dtype), 'n_elements': 4, 'BLOCK_SIZE': 4})
#     fn.saved_kernel_cache = {}
#     fn.saved_kernel_cache[kern_key] = "empty_path"

#     # compile and overwrite the kernel path
#     fn[(4, )](x, y, out, 4, 4)
#     triton.testing.assert_close(out, x + y)
#     assert fn.saved_kernel_cache[kern_key] != "empty_path"


@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('fast', ['1', '0'])
def test_jit(monkeypatch, device, fast):
    monkeypatch.setenv("TRITON_FAST_JIT", fast)

    @triton.jit
    def add_kernel(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        y = tl.load(in_ptr1 + offsets, mask=mask)
        output = x + y
        tl.store(out_ptr + offsets, output, mask=mask)

    x = torch.randn(4, device=device)
    y = torch.randn(4, device=device)
    out = torch.zeros_like(x)
    add_kernel[(4, )](x, y, out, 4, 4)
    triton.testing.assert_close(out, x + y)

    out = torch.zeros_like(x)
    add_kernel[(4, )](x, y, out, 4, 4)
    triton.testing.assert_close(out, x + y)

    if fast == "1":
        assert isinstance(add_kernel, FastJITFunction)
    else:
        assert isinstance(add_kernel, JITFunction)


@pytest.mark.parametrize('device', ['cuda'])
def test_key_change(monkeypatch, device):
    keys = []

    @triton.jit
    def _add_kernel(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        y = tl.load(in_ptr1 + offsets, mask=mask)
        output = x + y
        tl.store(out_ptr + offsets, output, mask=mask)

    keys.append(_add_kernel.saved_cache_key)

    @triton.jit
    def _add_kernel_change_sig(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
        NOP: "tl.constexpr" = 0,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        y = tl.load(in_ptr1 + offsets, mask=mask)
        output = x + y
        tl.store(out_ptr + offsets, output, mask=mask)

    keys.append(_add_kernel_change_sig.saved_cache_key)

    @triton.jit
    def _add_kernel_change_code(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        y = tl.load(in_ptr1 + offsets, mask=mask)
        output = x - y
        tl.store(out_ptr + offsets, output, mask=mask)

    keys.append(_add_kernel_change_code.saved_cache_key)

    monkeypatch.setenv("AMDGCN_ENABLE_DUMP", "1")

    @triton.jit
    def _add_kernel_change_env(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        y = tl.load(in_ptr1 + offsets, mask=mask)
        output = x + y
        tl.store(out_ptr + offsets, output, mask=mask)

    keys.append(_add_kernel_change_env.saved_cache_key)

    assert len(keys) == len(set(keys))


@pytest.mark.parametrize('device', ['cuda'])
def test_fast_jit_err_json_file(device):

    @triton.jit
    def add_kernel_utils_err_json(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        y = tl.load(in_ptr1 + offsets, mask=mask)
        output = x + y
        tl.store(out_ptr + offsets, output, mask=mask)

    fn = add_kernel_utils_err_json

    fpath = f"{get_saved_kernel_cache_dir()}/add_kernel_utils_err_json-{get_saved_kernel_cache_hash(fn)}.json"
    with open(fpath, "w") as fp:
        fp.write("not JSON format")

    x = torch.randn(4, device=device)
    y = torch.randn(4, device=device)
    out = torch.zeros_like(x)
    # compile kernel by triton.jit
    fn[(4, )](x, y, out, 4, 4)
    triton.testing.assert_close(out, x + y)


@pytest.mark.parametrize('device', ['cuda'])
def test_fast_jit_auto_do_not_specialize(monkeypatch, device):
    monkeypatch.setenv("TRITON_AUTO_DNS_THRESHOLD", "0")

    @triton.jit
    def add_kernel_do_not_specialize(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        y = tl.load(in_ptr1 + offsets, mask=mask)
        output = x + y
        tl.store(out_ptr + offsets, output, mask=mask)

    fn = add_kernel_do_not_specialize

    x = torch.randn(4, device=device)
    y = torch.randn(4, device=device)
    out = torch.zeros_like(x)
    kernels = set()
    # compile kernel by triton.jit
    # TRITON_AUTO_SPECIALIZE_THRESHOLD=0, disable auto-do-not-specialize
    kernels.add(fn[(4, )](x, y, out, 4, 4).hash)
    kernels.add(fn[(4, )](x, y, out, 8, 4).hash)
    kernels.add(fn[(4, )](x, y, out, 12, 4).hash)
    kernels.add(fn[(4, )](x, y, out, 32, 4).hash)
    triton.testing.assert_close(out, x + y)
    kernels.add(fn[(4, )](x, y, out, 64, 4).hash)
    kernels.add(fn[(4, )](x, y, out, 128, 4).hash)

    fpath = f"{get_saved_kernel_cache_dir()}/add_kernel_do_not_specialize-{get_saved_kernel_cache_hash(fn)}.json"
    with open(fpath) as fp:
        assert len(json.load(fp)['dns']) == 0

    monkeypatch.setenv("TRITON_AUTO_DNS_THRESHOLD", "5")

    fn = triton.jit(add_kernel_do_not_specialize.fn)

    kernels.add(fn[(4, )](x, y, out, 4, 8).hash)
    triton.testing.assert_close(out, x + y)
    kernels.add(fn[(4, )](x, y, out, 16, 16).hash)
    kernels.add(fn[(4, )](x, y, out, 32, 32).hash)
    kernels.add(fn[(4, )](x, y, out, 64, 64).hash)

    fpath = f"{get_saved_kernel_cache_dir()}/add_kernel_do_not_specialize-{get_saved_kernel_cache_hash(fn)}.json"
    with open(fpath) as fp:
        assert len(json.load(fp)['dns']) == 0

    # TRITON_AUTO_SPECIALIZE_THRESHOLD=5, n_elements has now accumulated 5 distinct values,
    # so it do-not-specialize
    new_kernel = fn[(4, )](x, y, out, 8, 4).hash # the new kernel

    fpath = f"{get_saved_kernel_cache_dir()}/add_kernel_do_not_specialize-{get_saved_kernel_cache_hash(fn)}.json"
    with open(fpath) as fp:
        assert json.load(fp)['dns'] == ["n_elements"]

    new_kernels = set()
    new_kernels.add(fn[(4, )](x, y, out, 7, 4).hash)
    triton.testing.assert_close(out, x + y)
    new_kernels.add(fn[(4, )](x, y, out, 8, 4).hash)
    triton.testing.assert_close(out, x + y)
    new_kernels.add(fn[(4, )](x, y, out, 100, 4).hash)
    triton.testing.assert_close(out, x + y)
    assert len(new_kernels) == 1 and new_kernels.pop() == None, f"No new kernel should be generation after auto do-not-specialize!"
    assert new_kernel not in kernels, f"The new kernel should be not in the previous kernel! {new_kernel} vs. {kernels}"
