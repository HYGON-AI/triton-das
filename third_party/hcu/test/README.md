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