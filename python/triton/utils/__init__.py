from typing import Union
from .kernelentry import kernelentry
from .hcutuner import hcutune, ConfigLoader, get_gpu_label, hcutune_configured
from .lang import annotate_hint
from .testing import dist_perf_report


def get_optimal_config(op_name: str, key: Union[list, dict, tuple, str], device_name: str = None):
    device_name = device_name if device_name else get_gpu_label()
    tuned = global_config_loader.get_tuned_cache(op_name, device_name)
    if tuned:
        res = tuned.get_optimal_config(key)
        if res:
            return res
        else:
            import os
            from .. import knobs
            cache_dir = knobs.cache.dir
            print(
                f"[hcutuner] WARNING: Not found optimal config for {op_name} !!! cache dir: {cache_dir},\t"
                f"device name: {device_name},\tconfig key: {key}"
            )
            return None
    print(
        f"[hcutuner] WARNING: Not found config cache for ({op_name}, {device_name})."
    )
    return None


def get_config_cache(op_name: str, device_name: str = None):
    device_name = device_name if device_name else get_gpu_label()
    tuned = global_config_loader.get_tuned_cache(op_name, device_name)
    if not tuned:
        print(
            f"[hcutuner] WARNING: Not found config cache for ({op_name}, {device_name})."
        )
        return None
    return tuned.cache['configs']


global_config_loader = ConfigLoader()


__all__ = [
    "kernelentry",
    "hcutune",
    "get_optimal_config",
    "annotate_hint",
    "dist_perf_report",
    "get_gpu_label",
    "get_config_cache",
    "hcutune_configured",
]
