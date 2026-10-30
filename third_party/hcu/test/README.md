# HCU Test Suit

This directory contains utility scripts customized for daily development on HCU GPUs.

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

# Regression

If you want to update llvm package, please modify the llvm-hash.txt to a new git commit hash.

```bash
bash third_party/hcu/test/regression.sh --suite=scale -q
```

Compile checks stop at LLVM IR; assembly and execution checks require gfx946.

```bash
python -m pytest third_party/hcu/test/test_scaled_compile.py -k upstream_scaled_compile -q
```

## NMZ GEMM performance gates

The `all` suite includes eight FP16 GEMM gates for 72-CU gfx938 (BW1100).
They cover 4096/5120, row-row/row-col, and default/legacy MMAC layouts with
fixed schedules and runtime K. Other device profiles skip these references.
To run only the gates on an idle GPU:

```bash
HIP_VISIBLE_DEVICES=0 bash third_party/hcu/test/regression.sh --suite=gemm-perf -q
```

Each gate checks the full result against FP32, verifies that K-64 and K reuse
one binary, and measures the median of 11 HIP-graph samples of four launches.
The floor is 99% of the lowest median from three independent calibration runs
in opt:8576 with Triton `cdf6827` and LLVM `96fd797`. Calibration data and the
release comparison are in
[triton_gemm_asm](http://42.228.13.241:10068/xukang/triton_gemm_asm/-/tree/tune/opt8576-bmz-nmz-20261017/experiments/20261030-nmz-release-regression).
