# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

import os
import torch
import triton
import functools
from triton.runtime.driver import driver
from triton.runtime.jit import DependenciesFinder

from triton import __version__ as triton_version
triton_major_version = int(triton_version.split(".")[0])
triton_minor_version = int(triton_version.split(".")[1])
triton_version_float = triton_major_version + float(triton_minor_version / 10)

if triton_version_float >= 3.3:
    from triton.knobs import cache as cache_knob
else:
    from triton.runtime.cache import default_cache_dir, default_dump_dir, default_override_dir

rocm_version = None


@functools.lru_cache()
def get_triton_label():
    import importlib.metadata as md
    try:
        return md.version("triton")
    except md.PackageNotFoundError:
        return md.version("flagtree")


def get_cache_dir():
    if triton_version_float >= 3.3:
        return cache_knob.dir
    else:
        return os.getenv("TRITON_CACHE_DIR", "").strip() or default_cache_dir()


def get_dump_dir():
    if triton_version_float >= 3.3:
        return cache_knob.dump_dir
    else:
        return os.getenv("TRITON_DUMP_DIR", "").strip() or default_dump_dir()


def get_override_dir():
    if triton_version_float >= 3.3:
        return cache_knob.override_dir
    else:
        return os.getenv("TRITON_OVERRIDE_DIR", "").strip() or default_override_dir()


@functools.lru_cache
def get_gpu_label():
    target = driver.active.get_current_target()
    device = torch.cuda.current_device()
    num_cu = torch.cuda.get_device_properties(device).multi_processor_count
    return f"{target.arch}_cu{num_cu}"


def get_weak_fn_hash(fn: triton.JITFunction):
    # we are not a compiler, just an autotuner match, we don't need globals
    if triton_version_float >= 3.5:
        dependencies_finder = DependenciesFinder(name=fn.__name__, globals={}, src=fn.src, nonlocals={})
    else:
        dependencies_finder = DependenciesFinder(name=fn.__name__, globals={}, src=fn.src)
    dependencies_finder.visit(fn.parse())
    return dependencies_finder.ret


def _get_rocm_version():
    """
    Get ROCM runtime/driver version (i.e. which rocm linker is used).
    This version is often different from the rocm version pytorch uses internally.
    """
    global rocm_version
    if rocm_version is not None:
        return rocm_version
    try:
        import subprocess
        import re

        rocm_ldd_path = triton.backends.backends["hcu"].compiler.path_to_rocm_lld()
        rocm_dir = os.path.dirname(rocm_ldd_path)
        amdgpu_arch_path = os.path.abspath(os.path.join(rocm_dir, "amdgpu-arch"))

        result = subprocess.check_output(
            [amdgpu_arch_path, "--version"],
            stderr=subprocess.STDOUT,
        )
        version = re.search(
            r".*roc-(\d+\.\d+.\d+).*", result.decode("utf-8"), flags=re.MULTILINE
        )
        rocm_version = version.group(1)
    except Exception as e:
        # print(
        #     f"[triton.utils.jit] Fail to determining rocm version with: {e}\n"
        #     f"using torch.version.hip as fallback"
        # )
        rocm_version = f"torch_{torch.version.hip}"
    return rocm_version


@functools.lru_cache()
def get_runtime_label():
    assert torch.version.hip is not None
    return f"rocm_{_get_rocm_version()}"


def create_tuple(k):
    s = k[1:-1]
    entries = s.split(", ")
    ret = []
    for e in entries:
        if e[0] == "'" or e[0] == '"':
            ret.append(e[1:-1])
        else:
            try:
                ret.append(eval(e))
            except ValueError:
                ret.append(eval(str(e)))
    ret_t = tuple(ret)
    return ret_t
