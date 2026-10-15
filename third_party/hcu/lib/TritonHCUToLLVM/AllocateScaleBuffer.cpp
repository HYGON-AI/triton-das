#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TritonHCU/ScaleMemoryPlan.h"
#include "TritonHCUToLLVM/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;
namespace {

constexpr int64_t kNumRows = 512;
constexpr int64_t kRowsPerSlot = 16; // 16 rows * 16 B/row = 256 B.
constexpr int64_t kNumSlots = kNumRows / kRowsPerSlot;
constexpr int64_t kRowAlignment = 8; // Keeps m0 aligned and bankXor unchanged.
constexpr int64_t kScaleGroupSize = 32; // One E8M0 scale per 32 input elements.

// Allocate the HCU Scale Buffer after scale transfer and MMAC layouts are
// fixed.
//
// Hardware storage and addressing
// -------------------------------
// Scale Buffer is dedicated on-chip scale storage, separate from Shared/LDS.
// Spec sections 2.5.9, 3.14.4 and 3.16.2 define its capacity, MMAC addressing
// and DS transfer rules. All addresses below are LOGICAL row/col coordinates;
// the drawing does not imply a row-major physical SRAM byte layout.
//
//                    col 0                 col 15
//                  +-----------------------------+
//   row   0        |       16 x 1-byte scales     |  slot 0
//       ...        |                             |  16 rows = 256 B
//   row  15        |                             |
//                  +-----------------------------+
//   row  16        |                             |  slot 1
//       ...        |                             |
//   row  31        |                             |
//                  +-----------------------------+
//       ...        |                             |
//                  +-----------------------------+
//   row 496        |                             |  slot 31
//       ...        |                             |
//   row 511        |                             |
//                  +-----------------------------+
//       512 rows x 16 cols x 1 B = 8 KiB = 32 slots
//
// Within one M16/N16 MMAC group, col selects M/N and row selects the K32
// scale group. A K64 MMAC consumes two adjacent scale rows per operand.
// More M16/N16 groups therefore need additional row ranges, not more columns.
// A/B occupy the same Scale Buffer address space and must not overlap while
// both are live. Kernel metadata reserves whole slots, not individual rows.
//
// ds2buf writes scales; MMAC reads them. These use different alignment rules:
//
//   LDS -- ds2buf(mode 0/1/2) --> Scale Buffer -- scaled MMAC --> registers
//          b = 1/2/4 B/lane                      A/B row addresses
//          Wave64: 64*b B
//          destination: 4*b rows x 16 cols
//
// ds2buf mode requires its destination base to meet its 1/2/4-row alignment.
// MMAC encodes two row bases in m0; each MUST be 8-row aligned (128 B of
// logical capacity). Each instruction adds an independent 5-bit A/B offset:
//
//   A row = m0.A_base + A_offset      B row = m0.B_base + B_offset
//                       0..31                            0..31
//
//   aligned base       +-------- 32-row addressing window --------+
//                      | ... | offset | offset+1 | ...           |
//                      +-----------------------------------------+
//                             K64 reads two adjacent rows
//
// The 8-row rule applies to the base, not every final row. A 32-row window
// limits one instruction's offsets; it does not limit the full scale tensor.
// MMAC lowering selects aligned windows as it traverses the tensor. The copy
// plan must preserve the required K32 adjacency and static offsets across
// the consuming waves. All actual accesses must remain inside allocated rows.
//
// Scale Buffer SRAM bank selection
// --------------------------------
// bank(row)=(row/4)%2: rows 0..3 and 8..11 use bank 0; rows 4..7 and
// 12..15 use bank 1. Every scaled MMAC must read A/B from opposite banks
// for full throughput. Moving an 8-row-aligned allocation base cannot change
// this bank bit. ScaleMemoryPlan therefore maps A/B with opposite bankXor
// values; both the copy and MMAC use that mapping. Allocation preserves it
// by assigning 8-row-aligned region bases. This is separate from LDS banks.
//
// Allocation geometry shared with lowering
// ---------------------------------------
// ScaleMemoryPlan defines b (bytes/lane), g (M16/N16 groups per ds2buf
// instTile), and p=4/g (phases/group). One instruction executed by one Wave64
// copies source instTile [16*g, b*p] into 4*b Scale Buffer rows. The complete
// CTA scale tensor [R,S], R=M or N and S=BK/32, may need many such instTiles:
//
//   instCols = b*p
//   numInsts = ceil(R/(16*g)) * ceil(S/instCols)
//   rowStride = alignTo(4*b, 8)
//   rows = numInsts * rowStride
//
//   region base   instTile 0       instTile 1                  region end
//        |        [4*b rows] gap   [4*b rows] gap ...                |
//        +--------+---------------+--------------------------------+
//                 <--- rowStride --->
//
// Region bases and reserved instTile starts use 8-row alignment so both
// ds2buf and MMAC can address them. A mode-0 copy may start four rows into
// its reservation to select the other SRAM bank. Gaps are allocation overhead,
// not LDS padding. ScaleMemoryPlan::getRow() maps each logical scale to a row
// relative to the region base; copy and MMAC lowering use the SAME mapping.
// This pass revalidates the earlier b/g choice against the final Shared and
// MFMA layouts, then assigns physical row bases. It does not change the Global
// load path, LDS layout, software-pipeline schedule or transfer instructions.
//
// Lifetime and reuse
// ------------------
// Each scale_write defines one storage descriptor with one direct dot consumer.
// In straight-line code its SSA live interval ends after that dot, independent
// of how long the numeric dot result remains live in registers.
//
// Allocation and synchronization are separate, as for Shared memory: reuse rows
// for disjoint SSA intervals; ScaleMembarAnalysis checks physical RAW/WAR/WAW
// hazards and inserts local barriers where needed. The allocator does not find
// or insert barriers. CFG/loops, predicates and untracked observers remain
// conservative in this allocator.
//
//   SSA:       write A/B ---- dot | write C/D ---- dot
//   rows X:    [===============) [=================)
//   membar:                 dot -- barrier -- write C/D
//
// Aligned first-fit permits spatial overlap only for disjoint live intervals.
// The kernel reserves ceil(peakRows/16) slots in hcu.scale_buffer. More than
// 32 slots is a compile-time error. Only the copy geometry and physical bases
// are attached to scale writes and dots for lowering; candidate reports and
// liveness intervals remain allocator-local.

#define GEN_PASS_DEF_ALLOCATESCALEBUFFER
#include "TritonHCUToLLVM/Passes.h.inc"

// Unmodelled observers could read a physical region without using its logical
// SSA value. Do not reuse in their presence. Standard async group operations
// only synchronize existing Global/Shared transfers; they do not read scales.
bool hasOnlyTrackedScaleAccesses(Block &block) {
  for (Operation &op : block) {
    if (op.getNumRegions())
      return false;
    if (isa<amdgpu::DotScaledOp, amdgpu::ScaleWriteOp,
            triton::gpu::LocalBarrierOp, mlir::gpu::BarrierOp,
            triton::gpu::AsyncCommitGroupOp, triton::gpu::AsyncWaitOp>(&op))
      continue;
    auto iface = dyn_cast<MemoryEffectOpInterface>(&op);
    if (!iface) {
      if (!isMemoryEffectFree(&op))
        return false;
      continue;
    }
    SmallVector<MemoryEffects::EffectInstance> effects;
    iface.getEffects(effects);
    for (const auto &effect : effects)
      if (effect.getResource() == SideEffects::DefaultResource::get() ||
          effect.getResource() == amdgpu::ScaledMemory::get())
        return false;
  }
  return true;
}

// Validate the fixed producer/Shared choice against the final MMAC layout.
FailureOr<HCU::ScaleMemoryPlan> getScalePlan(amdgpu::ScaleWriteOp write) {
  auto descriptor = write.getResult().getType();
  auto type =
      RankedTensorType::get(descriptor.getShape(), descriptor.getElementType());
  if (!HCU::ScaleMemoryPlan::isSupportedType(type) ||
      write.getSrc().getType().getShape() != type.getShape() ||
      write.getSrc().getType().getElementType() != type.getElementType()) {
    write.emitError("scale requires matching positive "
                    "static rank-two i8 LDS and result shapes");
    return failure();
  }
  auto shared = dyn_cast<triton::gpu::SwizzledSharedEncodingAttr>(
      write.getSrc().getType().getEncoding());
  if (!shared || shared.getMaxPhase() != 1 ||
      (shared.getOrder() != ArrayRef<unsigned>({1, 0}) &&
       shared.getOrder() != ArrayRef<unsigned>({0, 1}))) {
    write.emitError(
        "copy candidate family requires compact unswizzled rank-two LDS");
    return failure();
  }
  auto sourceType = write.getSrc().getType();
  unsigned fast = shared.getOrder()[0];
  if (sourceType.getAllocShape().take_back(2)[fast] !=
      sourceType.getShape()[fast]) {
    write.emitError("scale copy requires a compact LDS view; "
                    "strided subslices are unsupported");
    return failure();
  }
  if (!write.getResult().hasOneUse()) {
    write.emitError("logical scale requires one direct HCU scaled "
                    "dot consumer");
    return failure();
  }
  auto dot = cast<amdgpu::DotScaledOp>(*write.getResult().getUsers().begin());
  auto mfma = cast<triton::gpu::AMDMfmaEncodingAttr>(
      cast<RankedTensorType>(dot.getType()).getEncoding());
  unsigned side = dot.getAScale() == write.getResult() ? 0 : 1;
  // Validate only the descriptor's fixed plan; allocation does not select
  // another geometry or recompute candidate read-service costs.
  auto plan = HCU::ScaleMemoryPlan::fromMemDesc(descriptor);
  if (const char *reason = plan.getRejectionReason(
          type, mfma.getInstrShape()[2] / kScaleGroupSize,
          mfma.getTilesPerWarp()[side], mfma.getWarpsPerCTA()[side],
          shared.getOrder()[0])) {
    write.emitError() << "invalid scale-copy/m0-window plan: " << reason;
    return failure();
  }
  return plan;
}

// All allocation state is local to this module; no analysis reports enter IR.
class ScaleBufferAllocator {
public:
  explicit ScaleBufferAllocator(ModuleOp module) {
    module->walk<WalkOrder::PostOrder>(
        [&](Operation *op) { operationIds[op] = operationIds.size(); });

    auto functions = module.getOps<triton::FuncOp>();
    if (!llvm::hasSingleElement(functions) ||
        !(*functions.begin()).getBody().hasOneBlock())
      return;
    Block &block = (*functions.begin()).getBody().front();
    if (hasOnlyTrackedScaleAccesses(block))
      reuseBlock = &block;
  }

  LogicalResult allocate(amdgpu::ScaleWriteOp write) {
    auto plan = getScalePlan(write);
    if (failed(plan))
      return failure();
    auto dot = cast<amdgpu::DotScaledOp>(*write.getResult().getUsers().begin());
    int64_t base = allocateRows(getLiveInterval(write, dot), plan->rows);
    write.setRowBaseAttr(Builder(write.getContext()).getI32IntegerAttr(base));
    return success();
  }

  int64_t getNumSlots() const {
    return llvm::divideCeil(peakRows, kRowsPerSlot);
  }

private:
  Interval<size_t> getLiveInterval(amdgpu::ScaleWriteOp write,
                                   amdgpu::DotScaledOp dot) {
    size_t scopeEnd = operationIds.size() + 1;
    if (write->getBlock() != reuseBlock || dot->getBlock() != reuseBlock ||
        write.getPred())
      return {0, scopeEnd};

    // The descriptor has one direct use. Physical reuse hazards are handled
    // by ScaleMembarAnalysis after allocation, just as for Shared memory.
    return {operationIds.lookup(write), operationIds.lookup(dot) + 1};
  }

  int64_t allocateRows(Interval<size_t> live, int64_t count) {
    int64_t base = 0;
    while (true) {
      auto conflict = llvm::find_if(regions, [&](const Region &region) {
        return live.intersects(region.live) &&
               Interval<int64_t>(base, base + count).intersects(region.rows);
      });
      if (conflict == regions.end())
        break;
      base = llvm::alignTo(conflict->rows.end(), kRowAlignment);
    }
    regions.push_back({live, {base, base + count}});
    peakRows = std::max(peakRows, base + count);
    return base;
  }

  struct Region {
    Interval<size_t> live;
    Interval<int64_t> rows;
  };

  DenseMap<Operation *, size_t> operationIds;
  Block *reuseBlock = nullptr;
  SmallVector<Region> regions;
  int64_t peakRows = 0;
};

struct AllocateScaleBufferPass
    : impl::AllocateScaleBufferBase<AllocateScaleBufferPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    ScaleBufferAllocator allocator(module);
    auto result = module.walk([&](amdgpu::ScaleWriteOp write) {
      return WalkResult(allocator.allocate(write));
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    int64_t slots = allocator.getNumSlots();
    if (slots > kNumSlots) {
      module.emitError() << "scale memory exceeds " << kNumSlots << " slots";
      signalPassFailure();
      return;
    }
    // Required by the LLVM backend/driver resource-allocation contract.
    // compiler.py forwards this slot count (16 rows / 256 B per slot) as
    // the LLVM kernel attribute "hcu-scale-buffer-size".
    if (slots)
      module->setAttr("hcu.scale_buffer",
                      Builder(&getContext()).getI32IntegerAttr(slots));
  }
};
} // namespace

namespace mlir::triton::HCU {
std::unique_ptr<OperationPass<ModuleOp>> createAllocateScaleBuffer() {
  return std::make_unique<AllocateScaleBufferPass>();
}
} // namespace mlir::triton::HCU
