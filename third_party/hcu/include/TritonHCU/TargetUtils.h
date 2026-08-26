#ifndef TRITON_HCU_TARGETUTILS_H
#define TRITON_HCU_TARGETUTILS_H

#include "llvm/ADT/StringRef.h"
#include "TritonAMDGPUToLLVM/TargetUtils.h"

namespace mlir::triton::HCU {

using AMD::HCUISAFeature;
using AMD::HCUISAFeatureMask;
using AMD::deduceHCUISAFeature;
using AMD::supportsHCUISAFeature;

} // namespace mlir::triton::HCU
#endif
