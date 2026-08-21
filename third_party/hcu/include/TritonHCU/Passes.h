// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef TRITON_HCU_PASSES_H
#define TRITON_HCU_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {

#define GEN_PASS_DECL
#include "TritonHCU/Passes.h.inc"

} // namespace mlir

namespace mlir::triton {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonHCUToLLVMPass(StringRef targetArch, bool ftz);

std::unique_ptr<OperationPass<ModuleOp>>
createHCUConvertWarpSpecializeToLLVM(StringRef targetArch, int waspNumLoadWarps, 
    int waspNumMmaWarps, bool wdraEnabled, int wdraNumLoadRegs,
    int wdraNumMmaRegsMain, int wdraNumMmaRegsTail);

} // namespace mlir::triton

#endif
