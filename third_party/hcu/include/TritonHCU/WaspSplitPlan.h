#ifndef TRITONHCU_WASPSPLITPLAN_H
#define TRITONHCU_WASPSPLITPLAN_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>

namespace mlir::triton {
class DotOpInterface;

namespace HCU {

// The loop-level policy is attached before the first HCU accelerate-matmul
// pass. A plan is dot-specific because one specialize loop may contain more
// than one dot.
inline constexpr StringLiteral kWaspSplitPolicyAttrName =
    "hcu.wasp_split_policy";
inline constexpr StringLiteral kWaspSplitPlanAttrName = "hcu.wasp_split";
// Module attr: requested Load-first partition waves; length also fixes roles.
inline constexpr StringLiteral kWaspPartitionWarpsAttrName = "hcu.wasp_partition_warps";
// Module attrs: instruction-scheduling / empty-abarrier placement knobs.
// Set from HIPOptions in the HCU backend before WASP / LLVM conversion.
// Sched-barrier attrs are derived from enable_v_mmac_cluster != 0.
inline constexpr StringLiteral kSchedBarrierBeforeMmacAttrName =
    "hcu.sched_barrier_before_mmac";
inline constexpr StringLiteral kEmptyArriveAfterMmacAttrName =
    "hcu.empty_arrive_after_mmac";
inline constexpr StringLiteral kSchedBarrierBetweenABLoadsAttrName =
    "hcu.sched_barrier_between_a_b_loads";
// Unit attr on `rocdl.s.barrier` inserted by tritonhcu-consumer-pingpong.
// ConvertWarpSpecializeToLLVM maps these to one shared ebarrier so the two
// MMA consumers actually meet (per-partition barriers would not).
inline constexpr StringLiteral kConsumerPingpongBarrierAttrName =
    "hcu.consumer_pingpong";

struct WaspSplitPlan {
  // C dimension to partition: 0=M, 1=N.
  unsigned resultDim;
  // Number of MMA consumer groups.
  unsigned factor;
  // tt.dot operand that is sliced: 0=A for M, 1=B for N.
  unsigned partitionedOperand;
};

// Load-first topology for 1P1C, 1P2C and 2P2C. The tuple is the only
// stored state: its length determines roles and its elements determine waves.
enum class WaspTopoKind : int32_t {
  OnePOneC = 0, // [Load, MMA], no DataPartition split
  OnePTwoC = 1, // [Load, MMA_main, MMA_tail]
  TwoPTwoC = 2, // [Load_main, Load_tail, MMA_main, MMA_tail]
};

class WaspTopology {
public:
  // Accept 2/3/4 partitions, each with 1/2/4/8 waves, total <= 16.
  // Partitions of the same role have equal sizes; Consumer >= Producer.
  // WDRA's additional four-wave restriction belongs to the backend.
  static std::optional<WaspTopology> get(DenseI32ArrayAttr partitionWarps);

  WaspTopoKind getKind() const {
    return static_cast<WaspTopoKind>(numPartitions() - 2);
  }
  unsigned numPartitions() const { return partitionWarps.size(); }
  unsigned numLoadGroups() const { return numPartitions() == 4 ? 2 : 1; }
  unsigned numConsumerGroups() const {
    return numPartitions() - numLoadGroups();
  }

  ArrayRef<int32_t> getPartitionWarps() const {
    return partitionWarps.asArrayRef();
  }
  unsigned getPartitionWarps(unsigned idx) const {
    return getPartitionWarps()[idx];
  }
  unsigned getTotalLoadWarps() const {
    return numLoadGroups() * getPartitionWarps(0);
  }
  unsigned getTotalConsumerWarps() const {
    return numConsumerGroups() * getPartitionWarps(numLoadGroups());
  }

  bool hasSplitConsumers() const { return numConsumerGroups() > 1; }
  // TwoPTwoC slices producer buffers; OnePTwoC keeps a full-tile load.
  bool producerSliced() const { return getKind() == WaspTopoKind::TwoPTwoC; }
  // Independent empty/ready barrier sets per pair.
  unsigned numBarrierPairs() const { return numLoadGroups(); }

  // Indices are Load-first (actual regions obey this after partition reorder).
  bool isLoadPartition(unsigned idx) const { return idx < numLoadGroups(); }
  bool isMmaPartition(unsigned idx) const { return !isLoadPartition(idx); }
  bool isMmaMainPartition(unsigned idx) const { return idx == numLoadGroups(); }
  bool isMmaTailPartition(unsigned idx) const {
    return hasSplitConsumers() && idx == numLoadGroups() + 1;
  }

private:
  explicit WaspTopology(DenseI32ArrayAttr partitionWarps)
      : partitionWarps(partitionWarps) {}
  DenseI32ArrayAttr partitionWarps;
};

// Before OptimizePartitionWarps, read the requested tuple from the Module.
// Afterwards, use WaspTopology::get(wsOp.getPartitionNumWarpsAttr()) instead.
// Missing/invalid tuples return nullopt; callers allowing standalone operation
// must distinguish an absent attribute from a present but invalid one.
std::optional<WaspTopology> getWaspTopologyFromModule(Operation *op);

void setSchedBarrierBeforeMmacAttr(ModuleOp mod, bool enabled);
void setEmptyArriveAfterMmacAttr(ModuleOp mod, bool enabled);
void setSchedBarrierBetweenABLoadsAttr(ModuleOp mod, bool enabled);
bool getSchedBarrierBeforeMmacAttr(Operation *op, bool defaultVal = false);
bool getEmptyArriveAfterMmacAttr(Operation *op, bool defaultVal = false);
bool getSchedBarrierBetweenABLoadsAttr(Operation *op, bool defaultVal = false);

// Computes the deterministic, shape-only WASP split shared by early
// selection and DataPartition. DataPartition still validates the full
// def-use closure before applying it.
std::optional<WaspSplitPlan> inferWaspSplitPlan(DotOpInterface dot,
                                                unsigned factor);

void setWaspSplitPolicy(Operation *loop, unsigned factor);
bool hasWaspSplitPolicy(Operation *op);

void setWaspSplitPlan(Operation *dot, WaspSplitPlan plan);
std::optional<WaspSplitPlan> getWaspSplitPlan(Operation *dot);

// Returns the logical shape a consumer group will process. Only the
// partitioned non-K output dimension is divided; K and the shared operand
// remain full-size.
SmallVector<int64_t> getWaspEffectiveResultShape(DotOpInterface dot,
                                                 const WaspSplitPlan &plan);

} // namespace HCU
} // namespace mlir::triton

#endif // TRITONHCU_WASPSPLITPLAN_H
