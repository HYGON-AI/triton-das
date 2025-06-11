from .kernelentry import kernelentry
from .hcutuner import hcutune, ConfigLoader, get_gpu_label
from .lang import annotate_hint
from .testing import dist_perf_report


def get_optimal_config(op_name, *args, device_name=None, **kwargs):
    device_name = device_name if device_name else get_gpu_label()
    tuned = global_config_loader.get_tuned_cache(op_name, device_name)
    if tuned:
        configs = [t.get_optimal_config(*args, **kwargs) for t in tuned]
        return {k: v for d in configs if d for k, v in d.items()}
    return None


global_config_loader = ConfigLoader()


__all__ = [
    "kernelentry",
    "hcutune",
    "get_optimal_config",
    "annotate_hint",
    "dist_perf_report"
]
