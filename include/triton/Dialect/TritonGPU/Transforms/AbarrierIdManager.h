#ifndef TRITON_TRITONGPU_TRANSFORMS_ABARRIER_ID_MANAGER_H_
#define TRITON_TRITONGPU_TRANSFORMS_ABARRIER_ID_MANAGER_H_

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace triton {

// Get available abarrier IDs from 0 to 15, excluding those already used by
// ROCDL::HCUAbarrierInit operations in the module. Returns a vector of available
// barrier IDs of the requested size, or reports an error if not enough are
// available. This function uses a global singleton manager to track barrier ID
// allocation.
llvm::SmallVector<int> getAvailableAbarrierIds(scf::ForOp forOp,
                                                 int numBarriers);

// Release barrier IDs back to the global manager. This should be called when
// barriers are no longer needed.
void releaseAbarrierIds(llvm::ArrayRef<int> ids);

// Register a barrier ID as used in the global manager. This should be called
// when a barrier is created outside of the normal allocation flow.
void registerAbarrierId(int id);

// Reset the global abarrier ID manager. This is typically used to clear
// allocator state between separate transformations/passes.
void resetAbarrierIds();

// Create an allocation and init the abarriers.
void createAbarrier(scf::ForOp forOp, int barId, int arriveCount);

} // namespace triton
} // namespace mlir

#endif // TRITON_TRITONGPU_TRANSFORMS_ABARRIER_ID_MANAGER_H_
