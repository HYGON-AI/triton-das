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
