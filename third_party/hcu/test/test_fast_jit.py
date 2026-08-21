# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

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

    assert keys[0] == "(torch.float32, torch.float32, torch.float32, 4, 4)"

    out2 = torch.zeros_like(out)
    # launch kernel with triton.jit
    fn[(4, )](x, y, out2, 4, 4)
    triton.testing.assert_close(out2, x + y)


@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('kwargs', [{}, {'BLOCK_SIZE': 4}, {'BLOCK_SIZE': 4, 'num_warps': 2}])
def test_jit_key(monkeypatch, device, kwargs):
    monkeypatch.setenv("TRITON_CACHE_DIR", tempfile.mkdtemp())

    @triton.jit()
    def add_kernel_key(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr" = 4,
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
    fn[(4, )](x, y, out, 4, **kwargs)

    fpath = f"{get_saved_kernel_cache_dir()}/add_kernel_key-{get_saved_kernel_cache_hash(fn)}.json"
    assert os.path.isfile(fpath)

    with open(fpath) as fp:
        data = json.load(fp)['cache']

    keys = list(data.keys())
    assert len(keys) == 1

    if len(kwargs) > 1:
        assert keys[0] == "(torch.float32, torch.float32, torch.float32, 4, 4, 'num_warps=2')"
    else:
        assert keys[0] == "(torch.float32, torch.float32, torch.float32, 4, 4)"


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
    fn[(4, )](x, y, out, 4, 4)
    fn[(4, )](x, y, out, 8, 4)
    fn[(4, )](x, y, out, 12, 4)
    fn[(4, )](x, y, out, 32, 4)
    triton.testing.assert_close(out, x + y)
    fn[(4, )](x, y, out, 64, 4)
    fn[(4, )](x, y, out, 128, 4)
    assert len(fn._dns_set) == 0, f"TRITON_AUTO_SPECIALIZE_THRESHOLD=0 by default, so no auto do-not-specialize should be triggered!"

    monkeypatch.setenv("TRITON_AUTO_DNS_THRESHOLD", "5")

    fn = triton.jit(add_kernel_do_not_specialize.fn)
    fn[(4, )](x, y, out, 4, 8)
    fn[(4, )](x, y, out, 4, 16)
    fn[(4, )](x, y, out, 4, 32)
    fn[(4, )](x, y, out, 4, 64)
    fn[(4, )](x, y, out, 4, 128)
    assert len(fn._dns_set) == 0 and 'BLOCK_SIZE' not in fn._dns_value_history, f"constexpr BLOCK_SIZE should not be triggered auto do-not-specialize!"
    triton.testing.assert_close(out, x + y)

    # TRITON_AUTO_SPECIALIZE_THRESHOLD=5, n_elements has now accumulated 5 distinct values,
    # so it do-not-specialize
    fn[(4, )](x, y, out, 8, 16)
    fn[(4, )](x, y, out, 16, 16)
    fn[(4, )](x, y, out, 32, 32)
    kernels = set()
    kernels.add(fn[(4, )](x, y, out, 64, 64).metadata.hash) # the new kernel
    assert len(fn._dns_set) == 1 and list(fn._dns_set) == ['n_elements'], f"do-not-specialize arg != ['n_elements']!"

    new_kernels = set()
    new_kernels.add(fn[(4, )](x, y, out, 7, 64).metadata.hash)
    new_kernels.add(fn[(4, )](x, y, out, 111, 64).metadata.hash)
    new_kernels.add(fn[(4, )](x, y, out, 100, 64).metadata.hash)
    triton.testing.assert_close(out, x + y)
    assert len(kernels) == len(new_kernels) == 1, f"No new kernel should be generation after auto do-not-specialize!"
    assert kernels == new_kernels


@pytest.mark.parametrize('device', ['cuda'])
def test_fast_jit_autotune(monkeypatch, device):
    configs = [triton.Config(kwargs={'BLOCK_SIZE': 8}), triton.Config(kwargs={'BLOCK_SIZE': 16})]

    def _kernel_fast_jit_autotune(src, N, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offsets, mask=offsets < N) + 1
        tl.store(src + offsets, x, mask=offsets < N)

    Ns = 2
    fn = triton.autotune(configs=configs, key=['N'], restore_value=['src'])(
            triton.jit(_kernel_fast_jit_autotune))
    for i in range(6):
        N = Ns ** i
        src = torch.zeros(N, device=device)
        grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']), )
        fn[grid](src, N)
        triton.testing.assert_close(src, torch.ones_like(src))
    assert len(fn.fn._dns_set) == 0


    monkeypatch.setenv("TRITON_HCUTUNE", "1")
    fn = triton.autotune(configs=configs, key=['N'], restore_value=['src'])(
            triton.jit(_kernel_fast_jit_autotune))
    for i in range(6):
        N = Ns ** i
        src = torch.zeros(N, device=device)
        grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']), )
        fn[grid](src, N)
        triton.testing.assert_close(src, torch.ones_like(src))
    assert len(fn.fn._dns_set) == 0

    monkeypatch.setenv("TRITON_AUTO_DNS_THRESHOLD", "3")
    fn = triton.autotune(configs=configs, key=['N'], restore_value=['src'])(
            triton.jit(_kernel_fast_jit_autotune))
    for i in range(6):
        N = Ns ** i
        src = torch.zeros(N, device=device)
        grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']), )
        fn[grid](src, N)
        triton.testing.assert_close(src, torch.ones_like(src))
    assert len(fn.fn._dns_set) == 1 and str(fn.fn._dns_set) == "{'N'}", f"N should be triggered auto do-not-specialize!"
