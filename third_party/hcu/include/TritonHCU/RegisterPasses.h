// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

// HCU pass registration — included by RegisterTritonDialects.h
// Uses GEN_PASS_REGISTRATION with a distinct name to avoid conflict
// with AMD's registerTritonAMDGPUPasses (intentional, ours uses TritonHCU name)().

#include "TritonHCU/Passes.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_REGISTRATION
#include "TritonHCU/Passes.h.inc"

} // namespace mlir
