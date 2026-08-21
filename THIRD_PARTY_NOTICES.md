# Third-Party Notices

This repository is a fork of [triton-lang/triton](https://github.com/triton-lang/triton)
at commit `85400f80bf859a34ad7a746ffda877faf80312ab` (`release/3.6.x`), licensed
under MIT. See [LICENSE](./LICENSE).

The entries below are **additional third-party sources** (not part of that
upstream baseline). Official Triton files that Hygon copied or modified in-tree
are not listed here.

Each entry records: project, repository, commit or version, copyright, license,
local path, and whether Hygon modified the files.

## ByteDance Triton-distributed / SIMT

- **Project:** Triton-distributed
- **Repository:** https://github.com/ByteDance-Seed/Triton-distributed
- **Commit/Version:** `ff4b0d59be46f03ccc01f249206c9f372fcd9d4d` (Triton-distributed `main`, 2025-09-24; Hygon integrated in commit `3d1db5bd755a14d44640bc44b51d5db94324b1fb`, 2025-10-15)
- **Copyright:** Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
- **License:** MIT
- **Modifications:** Yes (integrated into Triton tree with HCU lowering, path relocation, and tutorial/kernel adaptation; not a verbatim copy of the upstream tree)

Derived from:

- `include/TritonDistributed/Conversion/TritonDistributedToLLVM/Passes.h` → `include/triton/Conversion/TritonDistributedToLLVM/Passes.h`
- `include/TritonDistributed/Conversion/TritonDistributedToLLVM/TritonDistributedToLLVMPass.h` → `include/triton/Conversion/TritonDistributedToLLVM/TritonDistributedToLLVMPass.h`
- `include/TritonDistributed/Conversion/TritonDistributedToTritonGPU/Passes.h` → `include/triton/Conversion/TritonDistributedToTritonGPU/Passes.h`
- `include/TritonDistributed/Conversion/TritonDistributedToTritonGPU/TritonDistributedToTritonGPUPass.h` → `include/triton/Conversion/TritonDistributedToTritonGPU/TritonDistributedToTritonGPUPass.h`
- `include/TritonDistributed/Dialect/Distributed/IR/Dialect.h` → `include/triton/Dialect/Distributed/IR/Dialect.h`
- `include/TritonDistributed/Dialect/SIMT/IR/Dialect.h` → `include/triton/Dialect/SIMT/IR/Dialect.h`
- `lib/Conversion/TritonDistributedToLLVM/AMD/ConvertAMDDistributedToLLVM.cpp` → `lib/Conversion/TritonDistributedToLLVM/HCU/ConvertHCUDistributedToLLVM.cpp`
- `lib/Conversion/TritonDistributedToLLVM/AMD/DistributedOpToLLVM.cpp` → `lib/Conversion/TritonDistributedToLLVM/HCU/DistributedOpToLLVM.cpp`
- `lib/Conversion/TritonDistributedToLLVM/AMD/LibDeviceToLLVM.cpp` → `lib/Conversion/TritonDistributedToLLVM/HCU/LibDeviceToLLVM.cpp`
- `lib/Conversion/TritonDistributedToTritonGPU/TritonDistributedToTritonGPU.cpp` → `lib/Conversion/TritonDistributedToTritonGPU/TritonDistributedToTritonGPU.cpp`
- `lib/Dialect/Distributed/IR/Dialect.cpp` → `lib/Dialect/Distributed/IR/Dialect.cpp`
- `lib/Dialect/Distributed/IR/Ops.cpp` → `lib/Dialect/Distributed/IR/Ops.cpp`
- `lib/Dialect/SIMT/IR/Dialect.cpp` → `lib/Dialect/SIMT/IR/Dialect.cpp`
- `lib/Dialect/SIMT/IR/Ops.cpp` → `lib/Dialect/SIMT/IR/Ops.cpp`
- `python/src/ir.cc` → `python/src/dist/ir.cc`
- `python/src/passes.cc` → `python/src/dist/passes.cc`
- `python/src/triton_distributed.cc` → `python/src/dist/triton_distributed.cc`
- `third_party/amd/language/hip/comms.cpp` → `third_party/amd/language/hip/comms.cpp`
- `python/triton_dist/language/extra/cuda/libnvshmem_device.py` → `third_party/amd/language/hip/libnvshmem_device.py`
- `python/triton_dist/language/extra/hip/librocshmem_device.py` → `third_party/amd/language/hip/librocshmem_device.py`
- `python/triton_dist/language/extra/libshmem_device.py` → `python/triton/language/extra/libshmem_device.py`
- `python/triton_dist/kernels/amd/gemm_reduce_scatter.py` → `python/tutorials/dist/01-intra-node-gemm-rs-fused-sequential.py`, `python/tutorials/dist/02-intra-node-gemm-rs-producer-consumer.py`
- `python/triton_dist/kernels/amd/all_gather_gemm.py` → `python/tutorials/dist/03-intra-node-ag-gemm-producer-consumer.py`
- `python/triton_dist/kernels/nvidia/gemm_allreduce.py` → `python/tutorials/dist/04-intra-node-gemm-allreduce-fused-oneshot.py`, `python/tutorials/dist/05-intra-node-gemm-allreduce-persist-ring.py`, `python/tutorials/dist/06-intra-node-gemm-allreduce-unfused-producer-consumer-ringreduce.py`
- `python/triton_dist/utils.py` → `python/tutorials/dist/utils.py`
- `include/TritonDistributed/Conversion/TritonDistributedToLLVM/CMakeLists.txt` → `include/triton/Conversion/TritonDistributedToLLVM/CMakeLists.txt`
- `include/TritonDistributed/Conversion/TritonDistributedToTritonGPU/CMakeLists.txt` → `include/triton/Conversion/TritonDistributedToTritonGPU/CMakeLists.txt`
- `include/TritonDistributed/Dialect/Distributed/CMakeLists.txt` → `include/triton/Dialect/Distributed/CMakeLists.txt`
- `include/TritonDistributed/Dialect/Distributed/IR/CMakeLists.txt` → `include/triton/Dialect/Distributed/IR/CMakeLists.txt`
- `include/TritonDistributed/Dialect/SIMT/CMakeLists.txt` → `include/triton/Dialect/SIMT/CMakeLists.txt`
- `include/TritonDistributed/Dialect/SIMT/IR/CMakeLists.txt` → `include/triton/Dialect/SIMT/IR/CMakeLists.txt`
- `lib/Conversion/TritonDistributedToLLVM/CMakeLists.txt` → `lib/Conversion/TritonDistributedToLLVM/CMakeLists.txt`
- `lib/Conversion/TritonDistributedToTritonGPU/CMakeLists.txt` → `lib/Conversion/TritonDistributedToTritonGPU/CMakeLists.txt`
- `lib/Dialect/Distributed/CMakeLists.txt` → `lib/Dialect/Distributed/CMakeLists.txt`
- `lib/Dialect/Distributed/IR/CMakeLists.txt` → `lib/Dialect/Distributed/IR/CMakeLists.txt`
- `lib/Dialect/SIMT/CMakeLists.txt` → `lib/Dialect/SIMT/CMakeLists.txt`
- `lib/Dialect/SIMT/IR/CMakeLists.txt` → `lib/Dialect/SIMT/IR/CMakeLists.txt`

Local path:

- `include/triton/Conversion/TritonDistributedToLLVM/Passes.h`
- `include/triton/Conversion/TritonDistributedToLLVM/TritonDistributedToLLVMPass.h`
- `include/triton/Conversion/TritonDistributedToTritonGPU/Passes.h`
- `include/triton/Conversion/TritonDistributedToTritonGPU/TritonDistributedToTritonGPUPass.h`
- `include/triton/Dialect/Distributed/IR/Dialect.h`
- `include/triton/Dialect/SIMT/IR/Dialect.h`
- `lib/Conversion/TritonDistributedToLLVM/HCU/ConvertHCUDistributedToLLVM.cpp`
- `lib/Conversion/TritonDistributedToLLVM/HCU/DistributedOpToLLVM.cpp`
- `lib/Conversion/TritonDistributedToLLVM/HCU/LibDeviceToLLVM.cpp`
- `lib/Conversion/TritonDistributedToTritonGPU/TritonDistributedToTritonGPU.cpp`
- `lib/Dialect/Distributed/IR/Dialect.cpp`
- `lib/Dialect/Distributed/IR/Ops.cpp`
- `lib/Dialect/SIMT/IR/Dialect.cpp`
- `lib/Dialect/SIMT/IR/Ops.cpp`
- `python/src/dist/ir.cc`
- `python/src/dist/passes.cc`
- `python/src/dist/triton_distributed.cc`
- `third_party/amd/language/hip/comms.cpp`
- `third_party/amd/language/hip/libnvshmem_device.py`
- `third_party/amd/language/hip/librocshmem_device.py`
- `python/triton/language/extra/libshmem_device.py`
- `python/tutorials/dist/01-intra-node-gemm-rs-fused-sequential.py`
- `python/tutorials/dist/02-intra-node-gemm-rs-producer-consumer.py`
- `python/tutorials/dist/03-intra-node-ag-gemm-producer-consumer.py`
- `python/tutorials/dist/04-intra-node-gemm-allreduce-fused-oneshot.py`
- `python/tutorials/dist/05-intra-node-gemm-allreduce-persist-ring.py`
- `python/tutorials/dist/06-intra-node-gemm-allreduce-unfused-producer-consumer-ringreduce.py`
- `python/tutorials/dist/utils.py`
- `include/triton/Conversion/TritonDistributedToLLVM/CMakeLists.txt`
- `include/triton/Conversion/TritonDistributedToTritonGPU/CMakeLists.txt`
- `include/triton/Dialect/Distributed/CMakeLists.txt`
- `include/triton/Dialect/Distributed/IR/CMakeLists.txt`
- `include/triton/Dialect/SIMT/CMakeLists.txt`
- `include/triton/Dialect/SIMT/IR/CMakeLists.txt`
- `lib/Conversion/TritonDistributedToLLVM/CMakeLists.txt`
- `lib/Conversion/TritonDistributedToTritonGPU/CMakeLists.txt`
- `lib/Dialect/Distributed/CMakeLists.txt`
- `lib/Dialect/Distributed/IR/CMakeLists.txt`
- `lib/Dialect/SIMT/CMakeLists.txt`
- `lib/Dialect/SIMT/IR/CMakeLists.txt`

## ROCm aiter (AMD Gluon Paged Attention)

- **Project:** ROCm aiter
- **Repository:** https://github.com/ROCm/aiter
- **Commit/Version:** `f061fba914d3646d25ce7ade41054f1e34f39a71` (aiter `main`, 2026-05-19; Hygon copied into Triton regression tests in commit `40135b34d8`, 2026-05-20)
- **Copyright:** Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved. (in-file headers; aiter repository LICENSE is also MIT / AMD)
- **License:** MIT
- **Modifications:** Yes (HCU regression: vendored as standalone Triton tests, `aiter` imports replaced with local shims)

Derived from:

- `aiter/ops/triton/gluon/pa_decode_gluon.py` → includes `paged_attention_decode_v2_gluon_large_block_dot_kernel`
- `aiter/ops/triton/gluon/pa_mqa_logits.py` → includes `_gluon_deepgemm_fp8_paged_mqa_logits`

Local path:

- `third_party/hcu/test/gluon/gluon_kernel_pa_decode.py`
- `third_party/hcu/test/gluon/gluon_kernel_pa_mqa_logits.py`

## vLLM

- **Project:** vLLM
- **Repository:** https://github.com/vllm-project/vllm
- **Commit/Version:** `cb080f32e38e87beda897d0602bf6a0d0c79d00f` (short prefix `cb080f32` recorded in `awq_perf.py`; snapshot of `main` when the AWQ kernel was copied)
- **Copyright:** Copyright contributors to the vLLM project
- **License:** Apache-2.0
- **Modifications:** Yes (HCU perf)

Local path:

- `third_party/hcu/perf/awq/awq_triton.py`
- `third_party/hcu/perf/awq/awq_perf.py`

## SGLang

- **Project:** SGLang
- **Repository:** https://github.com/sgl-project/sglang
- **Commit/Version:** `1a6e97577acb017fa9c25daf8a533969e941aa09` (PR `#3730`, *Feature DeepSeek V3/R1 INT8 Quantization (block-wise)*)
- **Copyright:** Copyright 2023-2024 SGLang Team
- **License:** Apache-2.0
- **Modifications:** Yes (HCU int8 GEMM perf)

Local path:

- `third_party/hcu/perf/gemm/int8_utils.py`
- `third_party/hcu/perf/gemm/test_block_int8.py`

## Hugging Face Transformers

- **Project:** Transformers
- **Repository:** https://github.com/huggingface/transformers
- **Commit/Version:** tag `v4.50.3`; local comment cites `src/transformers/activations.py` line 126
- **Copyright:** Copyright 2018- The Hugging Face team. All rights reserved.
- **License:** Apache-2.0
- **Modifications:** Yes (golden GELU/GEGLU helper used in HCU perf)

Local path:

- `third_party/hcu/perf/geglu.py` (function `geglu_golden_forward` only)
