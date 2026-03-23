#include "triton/Dialect/TritonGPU/Transforms/AbarrierIdManager.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/ErrorHandling.h"
#include <mutex>

using namespace mlir;

namespace {
class AbarrierIdManager {
private:
  static constexpr int MAX_BARRIER_IDS = 16;
  llvm::DenseSet<int> usedIds;
  mutable std::mutex mutex;

  AbarrierIdManager() = default;
  ~AbarrierIdManager() = default;
  AbarrierIdManager(const AbarrierIdManager &) = delete;
  AbarrierIdManager &operator=(const AbarrierIdManager &) = delete;

public:
  static AbarrierIdManager &getInstance() {
    static AbarrierIdManager instance;
    return instance;
  }

  llvm::SmallVector<int> allocateIds(int numBarriers,
                                     Operation *errorOp = nullptr) {
    std::lock_guard<std::mutex> lock(mutex);
    llvm::SmallVector<int> allocatedIds;

    for (int i = 0; i < MAX_BARRIER_IDS && allocatedIds.size() < numBarriers;
         ++i) {
      if (usedIds.find(i) == usedIds.end()) {
        allocatedIds.push_back(i);
        usedIds.insert(i);
      }
    }

    if (allocatedIds.size() < numBarriers) {
      if (errorOp) {
        errorOp->emitError()
            << "Not enough available abarrier IDs. Requested " << numBarriers
            << ", but only " << allocatedIds.size() << " available (total "
            << MAX_BARRIER_IDS << ", " << usedIds.size() << " already used)";
      }
      llvm::report_fatal_error("Not enough available abarrier IDs");
    }

    return allocatedIds;
  }

  void releaseIds(llvm::ArrayRef<int> ids) {
    std::lock_guard<std::mutex> lock(mutex);
    for (int id : ids) {
      usedIds.erase(id);
    }
  }

  void registerUsedId(int id) {
    std::lock_guard<std::mutex> lock(mutex);
    usedIds.insert(id);
  }

  void reset() {
    std::lock_guard<std::mutex> lock(mutex);
    usedIds.clear();
  }

  int getUsedCount() const {
    std::lock_guard<std::mutex> lock(mutex);
    return usedIds.size();
  }

  int getAvailableCount() const {
    std::lock_guard<std::mutex> lock(mutex);
    return MAX_BARRIER_IDS - usedIds.size();
  }
};
} // namespace

llvm::SmallVector<int>
mlir::triton::getAvailableAbarrierIds(scf::ForOp forOp, int numBarriers) {
  auto &manager = AbarrierIdManager::getInstance();
  return manager.allocateIds(numBarriers, forOp);
}

void mlir::triton::releaseAbarrierIds(llvm::ArrayRef<int> ids) {
  auto &manager = AbarrierIdManager::getInstance();
  manager.releaseIds(ids);
}

void mlir::triton::registerAbarrierId(int id) {
  auto &manager = AbarrierIdManager::getInstance();
  manager.registerUsedId(id);
}

void mlir::triton::resetAbarrierIds() {
  auto &manager = AbarrierIdManager::getInstance();
  manager.reset();
}

void mlir::triton::createAbarrier(scf::ForOp forOp, int barId, int arriveCount) {
  auto &manager = AbarrierIdManager::getInstance();
  manager.registerUsedId(barId);

  ImplicitLocOpBuilder rewriter(forOp.getLoc(), forOp);

  rewriter.create<ROCDL::HCUAbarrierInitOp>(forOp.getLoc(), barId, arriveCount);

  rewriter.setInsertionPointAfter(forOp);
  rewriter.create<ROCDL::HCUAbarrierInvOp>(barId);
}
