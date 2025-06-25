import os
import uuid
from triton.testing import Mark, perf_report


class MPMark(Mark):
    def run(self, show_plots=False, print_data=False, save_path='', return_df=False, **kwargs):
        if save_path:
            run_id = str(uuid.uuid4()) # random run_id for multiprocessing
            save_path = os.path.join(save_path, run_id)
        return super().run(show_plots=show_plots, print_data=print_data, save_path=save_path,
                           return_df=return_df, **kwargs)


def split_list_balanced(lst, n):
    def division_point(i):
        return i * q + min(i, r)

    q, r = divmod(len(lst), n)
    return [lst[division_point(i): division_point(i+1)] for i in range(n)]


def split_x_vals(vals: list):
    """
    Benchmark cases are split evenly into subsets, each assigned a MPMark to run on a process.
    Environment:
    1. TRITON_HCUTUNE_WORLD_SIZE, the total number of processes
    2. TRITON_HCUTUNE_LOCAL_RANK, the current process id

    Each process is assigned the above environment variables by triton.tools.launch.py
    """
    rank = eval(os.getenv("TRITON_HCUTUNE_LOCAL_RANK", "0").strip())
    world_size = eval(os.getenv("TRITON_HCUTUNE_WORLD_SIZE", "1").strip())
    assert rank < world_size, f"TRITON_HCUTUNE_LOCAL_RANK should < TRITON_HCUTUNE_WORLD_SIZE: {rank} < {world_size}"
    _vals = split_list_balanced(vals, world_size)
    assert len(_vals) == world_size
    return _vals[rank]


def set_device():
    """
    TRITON_HCUTUNE_VISIBLE_DEVICES, benchmarking is distributed on the given devices
    """
    if "TRITON_HCUTUNE_VISIBLE_DEVICES" in os.environ:
        rank = eval(os.getenv("TRITON_HCUTUNE_LOCAL_RANK").strip())
        devices = os.getenv("TRITON_HCUTUNE_VISIBLE_DEVICES").split(",")
        os.environ["CUDA_VISIBLE_DEVICES"] = devices[rank % len(devices)]


def dist_perf_report(benchmarks):
    """
    Reimplement perf_report() to support data parallel benchmarking.

    :param benchmarks: Benchmarking configurations.
    :type benchmarks: List of :class:`Benchmark`
    """

    print(
        "[DEPRECATED] @triton.utils.dist_perf_report is deprecated, please use @triton.testing.perf_report instead."
    )
    return perf_report(benchmarks)

    # for o in benchmarks:
    #     o.x_vals = split_x_vals(o.x_vals)
    # set_device()
    # wrapper = lambda fn: MPMark(fn, benchmarks)
    # return wrapper
