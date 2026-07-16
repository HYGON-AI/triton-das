#ifndef TRITONHCU_WDRASPLITPLAN_H
#define TRITONHCU_WDRASPLITPLAN_H

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Operation.h"

#include <cstdint>
#include <optional>

namespace mlir::triton {
class DotOpInterface;

namespace HCU {

// The loop-level policy is attached before the first HCU accelerate-matmul
// pass. A plan is dot-specific because one specialize loop may contain more
// than one dot.
inline constexpr StringLiteral kWdraSplitPolicyAttrName =
    "hcu.wdra_split_policy";
inline constexpr StringLiteral kWdraSplitPlanAttrName = "hcu.wdra_split";

struct WdraSplitPlan {
  // C dimension to partition: 0=M, 1=N.
  unsigned resultDim;
  // Number of MMA consumer groups.
  unsigned factor;
  // tt.dot operand that is sliced: 0=A for M, 1=B for N.
  unsigned partitionedOperand;
};

// Computes the deterministic, shape-only WDRA choice shared by early
// selection and DataPartition. DataPartition still validates the full
// def-use closure before applying it.
std::optional<WdraSplitPlan>
inferWdraSplitPlan(DotOpInterface dot, unsigned factor);

void setWdraSplitPolicy(Operation *loop, unsigned factor);
bool hasWdraSplitPolicy(Operation *op);

void setWdraSplitPlan(Operation *dot, WdraSplitPlan plan);
std::optional<WdraSplitPlan> getWdraSplitPlan(Operation *dot);

// Returns the logical shape a consumer group will process. Only the
// partitioned non-K output dimension is divided; K and the shared operand
// remain full-size.
SmallVector<int64_t>
getWdraEffectiveResultShape(DotOpInterface dot,
                            const WdraSplitPlan &plan);

} // namespace HCU
} // namespace mlir::triton

#endif // TRITONHCU_WDRASPLITPLAN_H
