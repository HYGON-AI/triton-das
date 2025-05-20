from .kernelentry import kernelentry
from .hcutuner import hcutune, ConfigLoader


def get_optimal_config(op_name, *args, **kwargs):
    tuned = global_config_loader.get_tuned_cache(op_name)
    if tuned:
        return tuned.get_optimal_config(*args, **kwargs)
    return None


global_config_loader = ConfigLoader()


__all__ = [
    "kernelentry",
    "hcutune",
    "get_optimal_config",
]
