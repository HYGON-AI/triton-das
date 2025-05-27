from .kernelentry import kernelentry
from .hcutuner import hcutune, ConfigLoader


def get_optimal_config(op_name, *args, **kwargs):
    tuned = global_config_loader.get_tuned_cache(op_name)
    if tuned:
        configs = [t.get_optimal_config(*args, **kwargs) for t in tuned]
        return {k: v for d in configs if d for k, v in d.items()}
    return None


global_config_loader = ConfigLoader()


__all__ = [
    "kernelentry",
    "hcutune",
    "get_optimal_config",
]
