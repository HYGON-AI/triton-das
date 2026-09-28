#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TritonHCU/WaspSplitPlan.h"
#include "TritonHCU/TargetUtils.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/ScopeExit.h"
#include <climits>
#include <memory>

using namespace mlir;
using namespace triton;
using namespace triton::gpu;
namespace ttng = triton::nvidia_gpu;
namespace hcu = triton::HCU;
namespace tta = mlir::triton::amdgpu;

//===----------------------------------------------------------------------===//
// relayoutWarps
//===----------------------------------------------------------------------===//

using RunPipelineFn = function_ref<LogicalResult(OpPassManager &, ModuleOp)>;

// Take the body of a partition into a new `tt.func`. We can use this to run a
// full compiler pipeline on the partition.
static OwningOpRef<ModuleOp> takeIntoFunction(ModuleAxisInfoAnalysis &axisInfo,
                                              Region *partition, int numWarps) {
  // Forward the module attributes (target, number of threads per warp, etc.)
  // onto the container module.
  ModuleOp mod = axisInfo.getModuleOp();
  OwningOpRef<ModuleOp> container = ModuleOp::create(mod.getLoc());
  Block *containerBlock = container->getBody();

  auto b = OpBuilder::atBlockBegin(containerBlock);
  FunctionType funcType = b.getFunctionType(partition->getArgumentTypes(), {});
  auto containerFunc = b.create<FuncOp>(mod.getLoc(), "container", funcType);
  containerFunc.getBody().takeBody(*partition);
  container.get()->setAttrs(mod->getAttrs());
  container.get()->setAttr(AttrNumWarpsName, b.getI32IntegerAttr(numWarps));

  // Replace `ttg.warp_return` with `tt.return` to make the IR valid.
  containerFunc.walk([&](WarpReturnOp op) {
    b.setInsertionPoint(op);
    b.create<ReturnOp>(op.getLoc());
    op.erase();
  });

  // This should make valid IR.
  if (failed(mlir::verify(*container)))
    llvm::report_fatal_error("expected partition region to make valid IR");

  // Attach axis info properties.
  auto wsOp = partition->getParentOfType<WarpSpecializeOp>();
  auto *funcInfo =
      axisInfo.getFuncData(wsOp->getParentOfType<FunctionOpInterface>());
  assert(funcInfo && "expected to find function axis info");
  for (auto [i, capture] : llvm::enumerate(wsOp.getExplicitCaptures())) {
    AxisInfo info = funcInfo->lookup(capture);
    containerFunc.setArgAttr(i, "tt.contiguity",
                             b.getI64IntegerAttr(info.getContiguity(0)));
    containerFunc.setArgAttr(i, "tt.divisibility",
                             b.getI64IntegerAttr(info.getDivisibility(0)));
    containerFunc.setArgAttr(i, "tt.constancy",
                             b.getI64IntegerAttr(info.getConstancy(0)));
  }

  return container;
}

// Take the partition body out of the container module and function.
static void extractPartitionBody(OwningOpRef<ModuleOp> container,
                                 Region *partition) {
  auto containerFunc = cast<FuncOp>(container->lookupSymbol("container"));

  // Rewrite the returns.
  containerFunc.walk([](ReturnOp op) {
    OpBuilder b(op);
    b.create<WarpReturnOp>(op.getLoc());
    op.erase();
  });

  partition->takeBody(containerFunc.getBody());
}

// Preserve the existing distribution policy: shrink the largest dimension,
// and grow the dimension selected by the caller.
static SmallVector<unsigned> resizeWarpsPerCTA(ArrayRef<unsigned> oldWarps,
                                               unsigned newNumWarps,
                                               unsigned growAxis) {
  SmallVector<unsigned> waves(oldWarps);
  unsigned count = product(waves);
  while (count > newNumWarps) {
    *llvm::max_element(waves) /= 2;
    count /= 2;
  }
  while (count < newNumWarps) {
    waves[growAxis] *= 2;
    count *= 2;
  }
  return waves;
}

// Reset the layouts of operations in a region and re-run layout assignment.
static LogicalResult relayoutWarps(ModuleAxisInfoAnalysis &axisInfo,
                                   Region *partition, int prevNumWarps,
                                   int newNumWarps, RunPipelineFn runPipeline) {
  bool hasMlsOperand = false;
  partition->walk([&](LocalLoadOp load) {
    hasMlsOperand |= isa<HCUMlsSharedEncodingAttr>(load.getSrc().getType().getEncoding());
  });
  if (hasMlsOperand) {
    // MLS writes already fixed the LDS / ds_read_matrix contract. Clearing
    // and rebuilding tensor encodings can lose the matching MMA layout.
    // Resize only the wave distribution, preserving instruction shape,
    // tilesPerWarp and K packing; leave shared encodings unchanged.
    AttrTypeReplacer replacer;
    replacer.addReplacement([&](AMDMfmaEncodingAttr enc) -> Attribute {
      return AMDMfmaEncodingAttr::get(
          enc.getContext(), enc.getVersion(),
          resizeWarpsPerCTA(enc.getWarpsPerCTA(), newNumWarps, /*growAxis=*/0),
          enc.getInstrShape(), enc.getIsTransposed(), enc.getCTALayout(),
          enc.getTilesPerWarp(), enc.getElementBitWidth(), enc.getMmacLayout());
    });
    replacer.addReplacement([&](BlockedEncodingAttr enc) -> Attribute {
      return BlockedEncodingAttr::get(
          enc.getContext(), enc.getSizePerThread(), enc.getThreadsPerWarp(),
          resizeWarpsPerCTA(enc.getWarpsPerCTA(), newNumWarps, /*growAxis=*/0),
          enc.getOrder(), enc.getCTALayout());
    });
    // Attribute replacement also visits DotOperand/Slice parents and all loop
    // argument types. Memdesc shared encodings are deliberately unchanged.
    for (Operation &op : partition->front())
      replacer.recursivelyReplaceElementsIn(&op, /*replaceAttrs=*/true,
                                            /*replaceLocs=*/false,
                                            /*replaceTypes=*/true);
    // DenseElementsAttr does not recursively expose its encoded tensor type.
    partition->walk([](arith::ConstantOp op) {
      if (auto value = dyn_cast<DenseElementsAttr>(op.getValue()))
        op.setValueAttr(value.reshape(cast<ShapedType>(op.getType())));
    });
    return success();
  }
  OwningOpRef<ModuleOp> container =
      takeIntoFunction(axisInfo, partition, prevNumWarps);

  // Start by removing all tensor encodings.
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement(
      [](RankedTensorType ty) { return ty.cloneWithEncoding({}); });
  // But don't remove them from the tensors inside descriptors.
  replacer.addReplacement([](TensorDescType ty) -> std::pair<Type, WalkResult> {
    return {ty, WalkResult::skip()};
  });
  replacer.recursivelyReplaceElementsIn(*container, /*replaceAttrs=*/false,
                                        /*replaceLocs=*/false,
                                        /*replaceTypes=*/true);

  ModuleOp mod = axisInfo.getModuleOp();
  auto target = mod->getAttrOfType<StringAttr>(AttrTargetName);
  if (!target)
    return mlir::emitError(mod.getLoc(), "module missing target specification");
  int threadsPerWarp = TritonGPUDialect::getThreadsPerWarp(mod);
  int numCTAs = TritonGPUDialect::getNumCTAs(mod);

  // Enable `convert-triton-to-tritongpu` to rematerialize source layouts for
  // TTG dialect operations. They will get cleared later.
  OpPassManager pm;
  pm.addPass(
      createConvertTritonToTritonGPU({target.str(), newNumWarps, threadsPerWarp,
                                      numCTAs, /*enableSourceRemat=*/true}));
  pm.addPass(createRelayoutTritonGPU());
  if (failed(runPipeline(pm, *container)))
    return failure();
  // Clear source rematerializations by propagating the source layout.
  container->walk([](UnrealizedConversionCastOp op) {
    op.getResult(0).replaceAllUsesWith(op.getOperand(0));
    op.erase();
  });

  pm.clear();
  pm.addPass(createTritonGPUCoalesce());
  pm.addPass(createTritonGPURemoveLayoutConversions());
  pm.addPass(createTritonGPUOptimizeThreadLocality());
  //pm.addPass(createTritonGPUAccelerateMatmul());
  pm.addPass(createTritonGPURemoveLayoutConversions());
  if (failed(runPipeline(pm, *container)))
    return failure();

  extractPartitionBody(std::move(container), partition);
  return success();
}

//===----------------------------------------------------------------------===//
// reorderPartitionsLoadFirst
//===----------------------------------------------------------------------===//

// True if this partition is a Load/producer group (vs MMA/consumer).
static bool regionIsLoadPartition(Region *region) {
  bool hasProducer = false;
  bool hasDot = false;
  region->walk([&](Operation *op) {
    if (isa<LoadOp, LocalStoreOp, tta::MatrixLoadToLocalOp>(op))
      hasProducer = true;
    if (isa<triton::DotOpInterface>(op))
      hasDot = true;
  });
  if (hasProducer && !hasDot)
    return true;
  if (hasDot)
    return false;
  return hasProducer;
}

static int regionMinTaskId(Region *region) {
  int minId = INT_MAX;
  region->walk([&](Operation *op) {
    if (auto attr = op->getAttrOfType<DenseI32ArrayAttr>("async_task_id")) {
      for (int32_t id : attr.asArrayRef())
        minId = std::min(minId, static_cast<int>(id));
    }
  });
  return minId == INT_MAX ? 0 : minId;
}

// WASP + WDRA: put all Load partitions before MMA; within each role, main
// (task 0) before tail (task 1). Swaps region bodies in place and permutes
// partitionNumWarps the same way.
static void reorderPartitionsLoadFirst(WarpSpecializeOp wsOp) {
  auto regions = llvm::to_vector(wsOp.getPartitionRegions());
  unsigned n = regions.size();
  if (n < 2)
    return;

  SmallVector<unsigned> order;
  for (unsigned i = 0; i < n; ++i)
    order.push_back(i);
  llvm::stable_sort(order, [&](unsigned a, unsigned b) {
    bool aLoad = regionIsLoadPartition(regions[a]);
    bool bLoad = regionIsLoadPartition(regions[b]);
    if (aLoad != bLoad)
      return aLoad && !bLoad;
    return regionMinTaskId(regions[a]) < regionMinTaskId(regions[b]);
  });

  bool identity = true;
  for (unsigned i = 0; i < n; ++i) {
    if (order[i] != i) {
      identity = false;
      break;
    }
  }
  if (identity)
    return;

  SmallVector<std::unique_ptr<Region>> temps;
  temps.reserve(n);
  for (unsigned i = 0; i < n; ++i) {
    temps.push_back(std::make_unique<Region>());
    temps.back()->takeBody(*regions[i]);
  }
  for (unsigned i = 0; i < n; ++i)
    regions[i]->takeBody(*temps[order[i]]);

  SmallVector<int32_t> oldWarps = llvm::to_vector(wsOp.getPartitionNumWarps());
  SmallVector<int32_t> newWarps(n);
  for (unsigned i = 0; i < n; ++i)
    newWarps[i] = oldWarps[order[i]];
  wsOp.setPartitionNumWarps(newWarps);
}

//===----------------------------------------------------------------------===//
// optimizePartitionWarps
//===----------------------------------------------------------------------===//

// Get the number of i32 registers required to store a tensor.
static unsigned getTensorNumI32Regs(RankedTensorType ty) {
  unsigned numElems = getTotalElemsPerThread(ty) *
                      product(getThreadsPerWarp(ty)) *
                      product(getWarpsPerCTA(ty));
  unsigned elSize =
      isa<PointerType>(ty.getElementType()) ? 64 : ty.getElementTypeBitWidth();
  return numElems * elSize / 32;
}

// GEMM Producer heuristic, in per-lane 32-bit VGPRs, after wave relayout:
//   MLS: tiles go directly to LDS; no payload VGPRs.
//   buffer load to LDS: no payload VGPRs; one offset per 32-bit transfer.
//   buffer load: sum A/B payload words plus one i32 offset per element.
// At this point buffer loads are usually still tt.load; buffer-op conversion
// runs later. This models the gfx946 buffer path, not full 64-bit pointer
// tensors. Reserve 24 for control, masks, LDS addresses and temporaries.
// Keep twice the largest non-pointer tensor as a floor for other arithmetic.
// For direct-to-LDS Producers without register-returning loads, use one copy:
// their offset/mask tensors do not imply a second live tile payload, and the
// fixed reserve covers address temporaries. This is a regular GEMM heuristic.
// Retain the old estimate for partitions without a recognized global load.
// FP16 64x64x64, four waves: A/B each have 16 elements/lane, so
// MLS P=24; buffer-to-LDS P=24+2*(16/2)=40; buffer P=24+2*(16/2+16)=72.
// FP16 direct-to-LDS uses one starting offset per pair, not per element.
// For 1P2C, C=(504-P)/2; this is not a fixed surcharge over MLS.
// This is not liveness analysis: spill checks still decide whether it fits.
static unsigned estimateProducerRegs(Region *partition, unsigned fallback) {
  bool hasGlobalLoad = false;
  bool hasBufferLoadLds = false, hasRegisterLoad = false;
  unsigned loadRegs = 0, maxTensorRegs = 0;
  partition->walk([&](Operation *op) {
    if (isa<tta::MatrixLoadToLocalOp>(op))
      hasGlobalLoad = true;
    if (auto load = dyn_cast<tta::BufferLoadToLocalOp>(op)) {
      hasGlobalLoad = true;
      hasBufferLoadLds = true;
      unsigned elems = getTotalElemsPerThread(load.getOffsets().getType());
      unsigned bits = load.getDest().getType().getElementTypeBitWidth();
      // gfx946 direct-to-LDS loads transfer 32 bits per lane. Lowering uses
      // only the starting offset of each vector; no loaded data lives in VGPRs.
      loadRegs += llvm::divideCeil(elems * bits, 32u);
    }
    for (Type type :
         llvm::concat<Type>(op->getOperandTypes(), op->getResultTypes())) {
      auto tensor = dyn_cast<RankedTensorType>(type);
      if (!tensor || isa<PointerType>(tensor.getElementType()))
        continue;
      unsigned words = llvm::divideCeil(
          getTotalElemsPerThread(tensor) * tensor.getElementTypeBitWidth(), 32u);
      maxTensorRegs = std::max(maxTensorRegs, words);
    }
    if (isa<LoadOp, tta::BufferLoadOp>(op)) {
      if (auto tensor = dyn_cast<RankedTensorType>(op->getResult(0).getType())) {
        hasGlobalLoad = true;
        hasRegisterLoad = true;
        unsigned elems = getTotalElemsPerThread(tensor);
        unsigned bits = isa<PointerType>(tensor.getElementType())
                            ? 64 : tensor.getElementTypeBitWidth();
        loadRegs += llvm::divideCeil(elems * bits, 32u) + elems;
      }
    }
  });
  unsigned tensorCopies = hasBufferLoadLds && !hasRegisterLoad ? 1u : 2u;
  unsigned tensorRegs = tensorCopies * maxTensorRegs;
  return hasGlobalLoad
             ? llvm::alignTo(24u + std::max(loadRegs, tensorRegs), 4u)
             : fallback;
}

static LogicalResult optimizePartitionNumWarps(ModuleAxisInfoAnalysis &axisInfo,
                                               WarpSpecializeOp wsOp,
                                               RunPipelineFn runPipeline,
                                               bool wdraEnabled) {
  ModuleOp mod = axisInfo.getModuleOp();
  bool explicitPartitions = mod->hasAttr("hcu.wasp_explicit_partitions");
  auto topology = hcu::getWaspTopologyFromModule(mod);
  if (!topology)
    return wsOp.emitError("missing or invalid hcu.wasp_partition_warps attribute");
  const hcu::WaspTopology &topo = *topology;
  if (wsOp.getPartitionNumWarps().size() != topo.numPartitions())
    return wsOp.emitError("HCU WASP partition count does not match topology; "
                          "the default region is not a separate wave group");

  // The tuple is authoritative. Do not reconstruct it from role totals.
  // Gluon has already assigned layouts and must match it without resizing.
  SmallVector<int32_t> partitionNumWarps = llvm::to_vector(topo.getPartitionWarps());
  if (explicitPartitions &&
      topo.getPartitionWarps() != wsOp.getPartitionNumWarps())
    return wsOp.emitError(
        "WASP partition waves do not match wasp_partition_warps");
  if (wdraEnabled && llvm::any_of(topo.getPartitionWarps(),
                                 [](int n) { return n != 4; }))
    return wsOp.emitError("HCU WDRA requires 4 waves per partition");

  // Unify WASP/WDRA wave layout before estimating regs / assigning warps:
  // Load* then MMA*, main (task 0) before tail (task 1).
  if (!explicitPartitions)
    reorderPartitionsLoadFirst(wsOp);

  // Extremely rough estimate of the number of registers needed per partition.
  // For each partition, get the number of i32 registers used by the largest
  // tensor value.
  //
  // Because the partition region is isolated from above, we could in theory
  // compile it to PTX and read the number of registers that got allocated.
  SmallVector<unsigned> maxTensorRegs;
  for (Region *partition : wsOp.getPartitionRegions()) {
    unsigned &tensorRegs = maxTensorRegs.emplace_back(0);
    partition->walk([&](Operation *op) {
      for (Type type :
           llvm::concat<Type>(op->getOperandTypes(), op->getResultTypes())) {
        if (auto tensor = dyn_cast<RankedTensorType>(type))
          tensorRegs = std::max(tensorRegs, getTensorNumI32Regs(tensor));
      }
    });
    // Assume that the largest tensor accounts for half of the registers used
    // by a warpgroup.
    tensorRegs *= 2;
  }

  // Reduce the number of warps used by partitions. For partitions with no
  // tensor computations, always reduce them to 1 warp.
  //
  // We can't use `nvvm.setmaxnreg` because this requires a known value for
  // `maxnreg` on the kernel, which is currently controlled by the frontend.
  // Thus, assume PTXAS will evenly distribute the total pool of registers
  // across all warps.
  //
  // If the compiler could control that, then we could allow non-uniform
  // register distributions, mostly beneficial for single-warp warpgroups that
  // just do some artihmetic.
  constexpr unsigned nTotalRegs = 1 << 17; // for ShB
  const unsigned threadsPerWarp =
      TritonGPUDialect::getThreadsPerWarp(axisInfo.getModuleOp());
  const unsigned defaultNumWarps = lookupNumWarps(wsOp);

  /*
  // Determine if a partition has a lower limit on the number of warps.
  SmallVector<int32_t> minWarpsForPartition(partitionNumWarps.size(), 1);
  for (auto [minWarps, region] :
       llvm::zip(minWarpsForPartition, wsOp.getPartitionRegions())) {
    region->walk([minWarps = &minWarps](Operation *op) {
      // Some instructions have critical throughput if have low register usage.
      // Make sure there are enough warps for these ops to execute quickly.
      if (isa<ttng::AsyncTMAGatherOp, ttng::AsyncTMAScatterOp,
              ttng::AsyncTMACopyGlobalToLocalOp>(op))
        *minWarps = 2;
      // TMEM ops require at least 4 warps to be able to read all lanes.
      else if (isa<ttng::TMEMLoadOp, ttng::TMEMStoreOp, ttng::TMEMAllocOp>(op))
        *minWarps = 4;
    });
  }

  bool changed;
  do {
    changed = false;

    // Assuming even distribution of registers, given the total number of warps
    // currently allocated, we can guess the number of registers PTXAS will
    // distribute to each warp.
    //
    // For example, given 18 warps and a tensor<128x256xf32> contained in an
    // 8-warp partition, we have (nTotalRegs/32/18) = ~113 regs per thread, and
    // the tensor requires 128 regs per thread in its partition. In this case,
    // nothing can be done.
    //
    // However, given a tensor<128x128xf32>, this requires only 64 regs per
    // thread in 8 warps. If we reduce the size of the warp to 4, the overall
    // regs per thread increases to (nTotalRegs/32/14) = ~146 regs per thread,
    // while the tensor now requires 128 regs per thread. This works.
    //
    // The next iteration sees ~170 regs per thread, but the tensor will require
    // 256, which is too many. So the algorithm stops at 4 warps. Evidently, if
    // there are other partitions that can be reduced, we have to iterate this
    // algorithm.
    int32_t curTotalNumWarps = std::accumulate(
        partitionNumWarps.begin(), partitionNumWarps.end(), defaultNumWarps);

    for (auto [minWarps, numWarps, tensorRegs] :
         llvm::zip(minWarpsForPartition, partitionNumWarps, maxTensorRegs)) {
      if (numWarps <= minWarps)
        continue;
      // Check if reducing the number of warps will still fit the tensor. If it
      // didn't fit to begin with, it won't fit after shrinking.
      unsigned reqRegsPerThread = tensorRegs / threadsPerWarp / (numWarps / 2);
      unsigned nextTotalNumWarps = curTotalNumWarps - (numWarps / 2);
      unsigned nextRegsPerThread =
          nTotalRegs / threadsPerWarp / nextTotalNumWarps;
      if (reqRegsPerThread <= nextRegsPerThread) {
        numWarps /= 2;
        changed = true;
        break;
      }
    }
  } while (changed);
  */

  SmallVector<int32_t> estRegUsage(partitionNumWarps.size());
  for (auto [partition, newNumWarps, prevNumWarps, tensorRegs, estRegs] :
       llvm::zip(wsOp.getPartitionRegions(), partitionNumWarps,
                 wsOp.getPartitionNumWarps(), maxTensorRegs, estRegUsage)) {
    // tensorRegs is the whole partition's i32 register count (already
    // doubled above). Convert to per-lane VGPRs and reserve 24 for scalars,
    // pointers and temporaries. Only Producer estimates drive allocation.
    estRegs = llvm::alignTo(
        llvm::divideCeil(tensorRegs, threadsPerWarp * newNumWarps) + 24u, 4u);

    // num_warps is just the initial layout size. MLS Producers often have
    // scalar operands only, so their wave distribution must be resized even
    // when tensorRegs=0 and the tensor relayout below is skipped. The shared
    // tile/instruction remain unchanged; each wave issues more/fewer copies.
    if (newNumWarps != prevNumWarps) {
      partition->walk([&](tta::MatrixLoadToLocalOp load) {
        auto enc = load->getAttrOfType<tta::MlsEncodingAttr>(
            tta::MlsEncodingAttr::getMnemonic());
        auto waves = resizeWarpsPerCTA(enc.getWarpsPerCTA(), newNumWarps,
                                      enc.getOrder()[0]);
        load->setAttr(tta::MlsEncodingAttr::getMnemonic(),
                      tta::MlsEncodingAttr::get(
                          enc.getContext(), enc.getOpIdx(), enc.getMlsTile(),
                          enc.getElemBitWidth(), enc.getElemBitTyKind(),
                          enc.getAlt2Kind(), enc.getVersion(), enc.getOrder(),
                          waves));
      });
    }
    // Layouts need to be reassigned if the number of warps changed and there
    // are tensor computations.
    if (newNumWarps == prevNumWarps || !tensorRegs)
      continue;
    // We need to reassign layouts.
    if (failed(relayoutWarps(axisInfo, partition, prevNumWarps, newNumWarps,
                             runPipeline)))
      return failure();
  }
  if (wdraEnabled) {
    auto arch = getAMDArch(axisInfo.getModuleOp());
    if (!arch)
      return wsOp.emitError("WDRA requires a hip target in ttg.target");
    // WDRA hardware constraints for four-wave partitions: each SIMD has one
    // wave from each partition, so B partitions means B waves per SIMD.
    // Supported topologies have B = 2/3/4 (8/12/16 waves per TG).
    // For per-lane VGPR quotas N[i] and S = sum(N[i]):
    //   4 <= N[i] <= 256, N[i] % 4 == 0;
    //   S <= getVgprSize(arch) (512 Wave64 VGPR units per SIMD on gfx946);
    //   S % (4 * B) == 0, so S / B is integral and also 4-aligned.
    // Compare quotas directly with the per-SIMD capacity, without multiplying
    // by 64 lanes or 4 SIMDs. The capacity need not be fully consumed.
    // Reserve each Producer independently, then share the remaining budget
    // between Consumers. Producer estimates may exceed 256 or leave too
    // little room for the minimum Consumer quotas; this heuristic does not
    // guarantee feasibility. LLVM lowering rejects invalid selected quotas.
    const unsigned producers = topo.numLoadGroups();
    auto partitions = wsOp.getPartitionRegions();
    for (unsigned i = 0; i < producers; ++i)
      estRegUsage[i] = estimateProducerRegs(partitions[i], estRegUsage[i]);
    const unsigned totalAlignment = 4 * estRegUsage.size();
    int remaining = llvm::alignDown(hcu::getVgprSize(*arch), totalAlignment);
    for (unsigned i = 0; i < producers; ++i)
      remaining -= estRegUsage[i];
    for (unsigned i = producers; i < estRegUsage.size(); ++i) {
      unsigned left = estRegUsage.size() - i;
      estRegUsage[i] = std::clamp(remaining / int(left) / 4 * 4, 4, 256);
      remaining -= estRegUsage[i];
    }
    // A capped Consumer can leave budget unused (notably 1P1C). Round the
    // total up minimally, preferring Consumers. LLVM lowering validates bounds
    // and capacity for the selected quotas; an explicit
    // user configuration must not fail because of a rough estimate.
    int sum = std::accumulate(estRegUsage.begin(), estRegUsage.end(), 0);
    int padding = (totalAlignment - sum % totalAlignment) % totalAlignment;
    for (int i = estRegUsage.size() - 1; i >= 0 && padding; --i) {
      int extra = std::min(padding, std::max(0, 256 - estRegUsage[i]));
      estRegUsage[i] += extra;
      padding -= extra;
    }
    if (!axisInfo.getModuleOp()->hasAttr("hcu.wasp_partition_regs"))
      axisInfo.getModuleOp()->setAttr(
          "hcu.wasp_partition_regs",
          DenseI32ArrayAttr::get(wsOp.getContext(), estRegUsage));
  }
  wsOp.setPartitionNumWarps(partitionNumWarps);
  return success();
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace mlir::triton::gpu {
#define GEN_PASS_DEF_TRITONGPUOPTIMIZEPARTITIONWARPSHCU
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu

namespace {
struct OptimizePartitionWarps
    : triton::gpu::impl::TritonGPUOptimizePartitionWarpsHCUBase<
          OptimizePartitionWarps> {
  using TritonGPUOptimizePartitionWarpsHCUBase::
      TritonGPUOptimizePartitionWarpsHCUBase;

  void runOnOperation() override;
};
} // namespace

void OptimizePartitionWarps::runOnOperation() {
  ModuleAxisInfoAnalysis axisInfo(getOperation());
  auto runPipelineFn = [&](OpPassManager &pm, ModuleOp container) {
    // The module must be directly nested under the current op for `runPipeline`
    // to work.
    getOperation().push_back(container);
    auto remove = llvm::make_scope_exit([&] { container->remove(); });
    return runPipeline(pm, container);
  };
  WalkResult result = getOperation().walk([&](WarpSpecializeOp wsOp) {
    if (failed(optimizePartitionNumWarps(axisInfo, wsOp, runPipelineFn,
                                         wdraEnabled)))
      return WalkResult::interrupt();
    return WalkResult::skip();
  });
  if (result.wasInterrupted())
    return signalPassFailure();

  ModuleOp mod = getOperation();
  int maxNumWarps = 0;
  mod.walk([&](WarpSpecializeOp op) {
    maxNumWarps = std::max<int>(maxNumWarps, op.getTotalPartitionWarps());
  });

  OpBuilder b(mod);
  auto totalNumWarps = maxNumWarps;
  mod->setAttr("ttg.total-num-warps", b.getI32IntegerAttr(totalNumWarps));

  mod.walk([&](WarpSpecializeOp op) {
    ArrayRef<int32_t> arr = op.getPartitionNumWarps();

    // Pack waves in Load-first order, including mixed partition sizes.
    // LLVM lowering subtracts each start ID before computing local thread IDs.
    SmallVector<int32_t> startIds(arr.size());
    int next = 0;
    for (unsigned i = 0; i < arr.size(); ++i) {
      startIds[i] = next;
      next += arr[i];
    }
    op.setWarpGroupStartIds(startIds);
  });
}
