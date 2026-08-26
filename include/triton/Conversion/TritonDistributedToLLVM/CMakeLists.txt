# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls --name TritonDistributedToLLVM)
add_public_tablegen_target(TritonDistributedToLLVMConversionPassIncGen)
