// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef TRITONHCU_WDRASPLITPLAN_H
#define TRITONHCU_WDRASPLITPLAN_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"
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
// Module attr: integer encoding of WdraTopoKind (see deriveWdraTopology).
inline constexpr StringLiteral kWdraTopoAttrName = "hcu.wdra_topo";
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

struct WdraSplitPlan {
  // C dimension to partition: 0=M, 1=N.
  unsigned resultDim;
  // Number of MMA consumer groups.
  unsigned factor;
  // tt.dot operand that is sliced: 0=A for M, 1=B for N.
  unsigned partitionedOperand;
};

// Warp-group topology for WDRA. Partition roles after OptimizePartitionWarps
// Load-first reorder (ascending wave id):
//   WASP / unsplit: [Load, MMA]
//   OnePTwoC (12-wave): [Load, MMA_main, MMA_tail]
//   TwoPTwoC (16-wave): [Load_main, Load_tail, MMA_main, MMA_tail]
enum class WdraTopoKind : int32_t {
  None = 0,     // WDRA off or single-consumer (no DataPartition split)
  OnePTwoC = 1, // load=4, mma=8
  TwoPTwoC = 2, // load=8, mma=8
};

struct WdraTopology {
  WdraTopoKind kind = WdraTopoKind::None;
  unsigned numConsumerGroups = 1; // MMA halves (factor)
  unsigned numLoadGroups = 1;     // Load halves / pairs
  unsigned warpsPerPartition = 4;

  bool enabled() const { return kind != WdraTopoKind::None; }
  // TwoPTwoC slices producer buffers; OnePTwoC keeps a full-tile load.
  bool producerSliced() const { return kind == WdraTopoKind::TwoPTwoC; }
  // Independent empty/ready barrier sets per pair.
  unsigned numBarrierPairs() const {
    return kind == WdraTopoKind::TwoPTwoC ? numLoadGroups : 1;
  }
  // Pre-arrive on empty bars so buffers start empty.
  unsigned emptyPreArriveCount() const {
    return kind == WdraTopoKind::OnePTwoC ? numConsumerGroups : 1;
  }
  unsigned numPartitions() const {
    switch (kind) {
    case WdraTopoKind::OnePTwoC:
      return 3; // Load, MMA_main, MMA_tail
    case WdraTopoKind::TwoPTwoC:
      return 4; // Load_main, Load_tail, MMA_main, MMA_tail
    default:
      return 2; // Load, MMA
    }
  }
  // Load-first helpers (valid after OptimizePartitionWarps reorder).
  bool isLoadPartition(unsigned idx) const {
    switch (kind) {
    case WdraTopoKind::OnePTwoC:
      return idx == 0;
    case WdraTopoKind::TwoPTwoC:
      return idx < numLoadGroups;
    default:
      return idx == 0;
    }
  }
  bool isMmaPartition(unsigned idx) const { return !isLoadPartition(idx); }
  bool isMmaMainPartition(unsigned idx) const {
    switch (kind) {
    case WdraTopoKind::OnePTwoC:
      return idx == 1;
    case WdraTopoKind::TwoPTwoC:
      return idx == numLoadGroups; // 2
    default:
      return idx == 1;
    }
  }
  bool isMmaTailPartition(unsigned idx) const {
    switch (kind) {
    case WdraTopoKind::OnePTwoC:
      return idx == 2;
    case WdraTopoKind::TwoPTwoC:
      return idx == numLoadGroups + 1; // 3
    default:
      return false;
    }
  }
};

// Derive topology from frontend wasp/wdra warp counts.
//   (load=4, mma=8) → OnePTwoC (existing)
//   (load=8, mma=8) → TwoPTwoC (paired producers/consumers)
WdraTopology deriveWdraTopology(bool wdraEnabled, int waspNumLoadWarps,
                                int waspNumMmaWarps);

void setWdraTopologyAttr(ModuleOp mod, WdraTopology topo);
WdraTopology getWdraTopologyAttr(Operation *op);

void setSchedBarrierBeforeMmacAttr(ModuleOp mod, bool enabled);
void setEmptyArriveAfterMmacAttr(ModuleOp mod, bool enabled);
void setSchedBarrierBetweenABLoadsAttr(ModuleOp mod, bool enabled);
bool getSchedBarrierBeforeMmacAttr(Operation *op, bool defaultVal = false);
bool getEmptyArriveAfterMmacAttr(Operation *op, bool defaultVal = false);
bool getSchedBarrierBetweenABLoadsAttr(Operation *op, bool defaultVal = false);

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
