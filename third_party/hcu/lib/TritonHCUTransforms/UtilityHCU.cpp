#include "TritonHCU/Utility.h"
#include "TritonAMDGPUTransforms/MfmaGroup.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"

#include <deque>

namespace mlir::triton::HCU {

namespace ttg = mlir::triton::gpu;

ttg::MmacLayout getMmacLayoutDefault(llvm::StringRef archGen) {
  static llvm::DenseMap<llvm::StringRef, ttg::MmacLayout> defaultLayoutMap = {
      {"gfx926", ttg::MmacLayout::LEGACY},
      {"gfx928", ttg::MmacLayout::LEGACY},
      {"gfx92a", ttg::MmacLayout::LEGACY},
      {"gfx936", ttg::MmacLayout::LEGACY},
      {"gfx938", ttg::MmacLayout::INTERLEAVE},
      {"gfx946", ttg::MmacLayout::INTERLEAVE},
      {"gfx948", ttg::MmacLayout::INTERLEAVE},
  };
  auto it = defaultLayoutMap.find(archGen);
  return it != defaultLayoutMap.end() ? it->second : ttg::MmacLayout::MFMA;
}

// AMD MFMA to HCU-equivalent mapping (gfx938+):
// | AMD MFMA isTransposed=false | MmacLayout::INTERLEAVE_TRANSPOSE |
// | AMD MFMA isTransposed=true  | MmacLayout::INTERLEAVE           |
// HCU encoding keeps isTransposed=false; transpose intent is materialized in
// mmacLayout only.
ttg::MmacLayout getMfmaMappedMmacLayout(bool logicalIsTransposed,
                             bool supportMmacLayout,
                             const MfmaIntrinsic &intrinsic) {
  if (!supportMmacLayout)
    return ttg::MmacLayout::LEGACY;
  if (!intrinsic.supportsMmacLit)
    return intrinsic.supportsMmacLts ? ttg::MmacLayout::TRANSPOSE : ttg::MmacLayout::LEGACY;
  return logicalIsTransposed || !intrinsic.supportsMmacLts
             ? ttg::MmacLayout::INTERLEAVE
             : ttg::MmacLayout::INTERLEAVE_TRANSPOSE;
}

LogicalResult validateMmacLayout(Operation *op, ttg::MmacLayout layout,
                                 const MfmaIntrinsic &intrinsic) {
  bool needsLit = layout == ttg::MmacLayout::INTERLEAVE ||
                  layout == ttg::MmacLayout::INTERLEAVE_TRANSPOSE;
  bool needsLts = layout == ttg::MmacLayout::TRANSPOSE ||
                  layout == ttg::MmacLayout::INTERLEAVE_TRANSPOSE;
  if ((needsLit && !intrinsic.supportsMmacLit) ||
      (needsLts && !intrinsic.supportsMmacLts))
    return op->emitError() << "MMAC layout " << ttg::stringifyMmacLayout(layout)
                           << " is not supported by " << intrinsic.name;
  return success();
}

FailureOr<std::pair<triton::DotOpInterface, unsigned>>
getDotOpIdxFromMatrixLoad(triton::MatrixLoadOp matrixOp) {
  SetVector<Operation *> slices;
  mlir::getForwardSlice(matrixOp.getResult(), &slices);

  std::deque<Operation *> worklist(slices.begin(), slices.end());
  while (!worklist.empty()) {
    Operation *op = worklist.front();
    worklist.pop_front();

    if (auto dotOp = dyn_cast<triton::DotOpInterface>(op)) {
      for (unsigned idx = 0; idx < dotOp->getNumOperands(); ++idx) {
        auto defineOp = dotOp->getOperand(idx).getDefiningOp();
        if (defineOp == matrixOp || (defineOp && slices.contains(defineOp))) {
          return success(std::make_pair(dotOp, idx));
        }
      }
    } else if (auto scfYieldOp = dyn_cast<scf::YieldOp>(op)) {
      for (auto &&yieldOperand : llvm::enumerate(scfYieldOp.getOperands())) {
        unsigned argIdx = yieldOperand.index();
        Value argVal = yieldOperand.value();

        Operation *defOp = argVal.getDefiningOp();
        if (defOp == matrixOp || (defOp && slices.contains(defOp))) {
          auto scfRes = scfYieldOp->getParentOp()->getResult(argIdx);
          SetVector<Operation *> outSlice;
          mlir::getForwardSlice(scfRes, &outSlice);
          for (Operation *outOp : outSlice)
            if (slices.insert(outOp))
              worklist.push_back(outOp);
        }
      }
    }
  }

  return failure();
}
} // namespace mlir::triton::HCU
