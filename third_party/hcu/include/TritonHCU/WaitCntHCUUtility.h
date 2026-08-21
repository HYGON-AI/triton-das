// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef TRITON_THIRD_PARTY_HCU_INCLUDE_TRITONHCU_WAITCNTHCUUTILITY_H_
#define TRITON_THIRD_PARTY_HCU_INCLUDE_TRITONHCU_WAITCNTHCUUTILITY_H_

#include "mlir/IR/BuiltinOps.h"

namespace mlir::triton::HCU {

// HCU Note: Starting with ZD and newer chips, s_barrier no longer implies
// s_waitcnt vmcnt(0). For patterns that require this ordering guarantee,
// we detect them explicitly and insert s_waitcnt vmcnt(0) in IR.
void addWaitCntHCU(ModuleOp mod);

} // namespace mlir::triton::HCU

#endif // TRITON_THIRD_PARTY_HCU_INCLUDE_TRITONHCU_WAITCNTHCUUTILITY_H_
