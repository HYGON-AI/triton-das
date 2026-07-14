# HCU Scripts

This directory contains utility scripts customized for daily development on HCUs.

# Scripts Usage

If you want to use scripts locally, you should source the launch scripts firstly:

```bash
source launch.sh
```
1. If you want to run google test, use this command:
```bash
run_google_test
```
2. If you want to run lit, use this command:
```bash
run_lit 0
```
or
```bash
run_lit 1
```
3. If you want to run pytest, use this command:
```bash
run_pytest
```
4. If you want to build wheel, use this command:
```bash
source gen_triton_wheel.sh
```
you can specify the version and llvm-project build path, such as:
```bash
source gen_triton_wheel.sh --version 2.1.0 --llvm_build_dir path_to_llvm_build
```


# Performance report

Default runs a **nightly vs local** compare (Triton 3.6.x):

```bash
./scripts/hcu/run_perf_regression.sh
./scripts/hcu/run_perf_regression.sh -g 2
```

By default every run:
1. `git` sync + `python setup.py develop` for aiter (`third_party/aiter`)
2. Resolve + **re-download** the nightly Triton whl matching an ancestor of `HEAD` (version prefix `3.6`)
3. Bench nightly → reinstall local Triton → bench local → write `compare.md`

Skip the forced steps only when you pass paths:

```bash
./scripts/hcu/run_perf_regression.sh --aiter-dir /path/to/aiter
./scripts/hcu/run_perf_regression.sh \
  --aiter-dir /path/to/aiter \
  --nightly-whl /path/to/triton-....whl
```

`$AITER_ROOT` is treated like `--aiter-dir` (skip sync/install).

Other:

```bash
./scripts/hcu/run_perf_regression.sh --resolve-nightly-only
./scripts/hcu/run_perf_regression.sh --operators gemm_a16w16
```

Outputs go under `perf_reports/compare_<timestamp>/` by default.

Operators (aiter Triton kernels):

- `gemm_a16w16` — `aiter/ops/triton/gemm_a16w16.py` via `op_tests/op_benchmarks/triton/bench_gemm_a16w16.py`
- `flash_attention_forward` — `aiter/ops/triton/flash_attention_forward.py` via `bench_flash_attention_forward.py`
- `fused_moe` — `aiter/ops/triton/moe_op.py` via `bench_moe.py`

`Δms% > 0` in `compare.md` means local is slower than nightly.

If the default GPU (physical id 0) is hung, `perf_report.py` auto-probes devices and
sets `HIP_VISIBLE_DEVICES` to the first healthy GPU.

# Regression

If you want to update llvm package, please modify the llvm-hash.txt to a new git commit hash.