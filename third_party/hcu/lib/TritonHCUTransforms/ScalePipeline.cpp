#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TritonAMDGPUToLLVM/TargetUtils.h"
#include "TritonHCU/Passes.h"
#include "TritonHCU/ScaleMemoryPlan.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "third_party/amd/lib/TritonAMDGPUToLLVM/Utility.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <array>
#include <functional>

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace tta = mlir::triton::amdgpu;

namespace mlir {
#define GEN_PASS_DEF_TRITONHCUSCALEPIPELINE
#include "TritonHCU/Passes.h.inc"

namespace {
namespace HCU = mlir::triton::HCU;

// Recognize a scalar base plus tensor offsets before pointer canonicalization.
// Scale tile offsets are assumed to fit buffer addressing; only the common
// base structure is checked here. gfx946 can also use native Global-to-LDS
// copies, which do not require a common buffer base.
static bool hasUniformScaleBase(Value ptr) {
  while (isa<RankedTensorType>(ptr.getType())) {
    Operation *op = ptr.getDefiningOp();
    if (!op)
      return false;
    if (auto add = dyn_cast<tt::AddPtrOp>(op))
      ptr = add.getPtr();
    else if (isa<ttg::ConvertLayoutOp, tt::BroadcastOp, tt::ExpandDimsOp,
                 tt::SplatOp>(op))
      ptr = op->getOperand(0);
    else
      return false;
  }
  return isa<tt::PointerType>(ptr.getType());
}

struct ScaleProducer {
  bool direct;
  unsigned vector;
  RankedTensorType regType;
  ttg::MemDescType memType;
};

static std::optional<ScaleProducer>
selectScaleProducer(IRRewriter &rewriter, tt::DotScaledOp dotOp,
                    TypedValue<RankedTensorType> scale, unsigned side,
                    bool preferDirect, ttg::MemDescType existingMemory = {}) {
  auto *ctx = dotOp.getContext();
  auto mma = cast<ttg::AMDMfmaEncodingAttr>(dotOp.getType().getEncoding());
  auto ctaLayout = ttg::getCTALayout(mma);
  auto shape = scale.getType().getShape();
  Value source = scale;
  bool exclusive = source.hasOneUse();
  while (auto convert = source.getDefiningOp<ttg::ConvertLayoutOp>()) {
    source = convert.getSrc();
    exclusive &= source.hasOneUse();
  }
  auto sourceType = cast<RankedTensorType>(source.getType());
  auto load = source.getDefiningOp<triton::LoadOp>();
  unsigned numWarps =
      ttg::lookupNumWarps(load ? load.getOperation() : dotOp.getOperation());
  auto module = dotOp->getParentOfType<ModuleOp>();
  auto arch = getAMDArch(module);
  bool nativeGlobalEnabled = arch && (*arch == "gfx946" || *arch == "gfx948");
  bool nativeDirectEnabled =
      load && (nativeGlobalEnabled || hasUniformScaleBase(load.getPtr()));
  triton::ModuleAxisInfoAnalysis axis(module);
  auto *info = load ? axis.getAxisInfo(load.getPtr()) : nullptr;
  unsigned globalOrder = ttg::getOrder(sourceType)[0];
  if (info && info->getContiguity(0) != info->getContiguity(1))
    globalOrder = info->getContiguity(0) > info->getContiguity(1) ? 0 : 1;
  // Standard pipelining guards a prologue load with a CTA-uniform splat.
  // Preserve that generated predicate without rejecting an otherwise legal
  // producer. Original elementwise user masks still select the
  // Global-to-REG-to-LDS path.
  bool pipelineMask = load && load->hasAttr("amd.pipeliner_part") &&
                      load.getMask() &&
                      load.getMask().getDefiningOp<tt::SplatOp>();
  unsigned originalVec = 1;
  if (info && (!load.getMask() || pipelineMask)) {
    auto perThread = ttg::getContigPerThread(sourceType);
    unsigned fast = ttg::getOrder(sourceType)[0];
    unsigned bound = std::min<unsigned>(
        16, std::min<unsigned>(perThread[fast],
                               std::min(info->getContiguity(fast),
                                        info->getDivisibility(fast))));
    originalVec = llvm::bit_floor(std::max(1u, bound));
  }
  std::optional<ScaleProducer> producer;
  // Fix the producer/Shared policy first. Select b/g only within that
  // layout; the allocator revalidates the choice against the final MFMA.
  std::array<std::optional<bool>, 2> canCopy;
  auto select = [&](bool direct, unsigned order, unsigned vector,
                    RankedTensorType regType, RankedTensorType ptrType) {
    auto shared = ttg::SwizzledSharedEncodingAttr::get(
        ctx, 4, 1, 1, {order, 1 - order}, ctaLayout);
    auto memType =
        existingMemory
            ? existingMemory
            : ttg::MemDescType::get(shape, rewriter.getI8Type(), shared,
                                    ttg::SharedMemorySpaceAttr::get(ctx), true);
    if (direct) {
      auto cvt = ttg::toLinearLayout(ptrType).invertAndCompose(
          ttg::toLinearLayout(memType));
      if (!LLVM::AMD::canCoalesceWriteIntoSharedMemory(rewriter, cvt, 64,
                                                       vector))
        return false;
    }
    unsigned ldsOrder =
        cast<ttg::SwizzledSharedEncodingAttr>(memType.getEncoding())
            .getOrder()[0];
    if (!canCopy[ldsOrder])
      canCopy[ldsOrder] =
          HCU::ScaleMemoryPlan::select(
              scale.getType(), mma.getInstrShape()[2] / 32,
              mma.getTilesPerWarp()[side], mma.getWarpsPerCTA()[side], ldsOrder)
              .has_value();
    if (!*canCopy[ldsOrder])
      return false;
    producer = ScaleProducer{direct, vector, regType, memType};
    return true;
  };
  bool canRebuildLoad = load && exclusive &&
                        (!load.getMask() || pipelineMask) && !load.getOther() &&
                        !load.getIsVolatile() && info;
  auto selectLoader = [&](bool direct) {
    if (!canRebuildLoad || (direct && (!nativeDirectEnabled || shape[1] < 4)))
      return false;
    unsigned fast = globalOrder, slow = 1 - fast;
    // Use the widest legal, fully populated REG loader. Native copies
    // may repeat the tile across waves: standard lowering predicates them.
    for (unsigned vector : {16u, 8u, 4u, 2u, 1u}) {
      if (shape[fast] % vector || info->getContiguity(fast) < vector ||
          info->getDivisibility(fast) < vector ||
          shape[0] * shape[1] < 64 * (direct ? 1 : numWarps) * vector ||
          (direct && vector < 4))
        continue;
      unsigned fastThreads = std::min<int64_t>(64, shape[fast] / vector);
      SmallVector<unsigned> perThread(2, 1), threads(2, 1), warps(2, 1);
      perThread[fast] = vector;
      threads[fast] = fastThreads;
      threads[slow] = 64 / fastThreads;
      for (unsigned warpFast = 1; warpFast <= (direct ? 1 : numWarps);
           warpFast *= 2) {
        warps[fast] = warpFast;
        warps[slow] = numWarps / warpFast;
        auto loader = ttg::BlockedEncodingAttr::get(
            ctx, perThread, threads, warps, {fast, slow}, ctaLayout);
        auto ptrType = cast<RankedTensorType>(load.getPtr().getType())
                           .cloneWithEncoding(loader);
        auto regType = sourceType.cloneWithEncoding(loader);
        if (!direct) {
          auto free = ttg::toLinearLayout(regType).getFreeVariableMasks();
          if (free.lookup(StringAttr::get(ctx, "lane")) ||
              free.lookup(StringAttr::get(ctx, "warp")))
            continue;
        }
        if (select(direct, direct ? fast : 1, vector, regType, ptrType))
          return true;
      }
    }
    return false;
  };
  // MLS/async uses native copies whenever legal. All REG paths have
  // compact s-contiguous LDS, independently of Global order.
  if (!preferDirect || !selectLoader(true)) {
    if (!selectLoader(false))
      select(false, 1, originalVec, sourceType, {});
  }
  return producer;
}

static Value prepareScaleStorage(IRRewriter &rewriter, tt::DotScaledOp dotOp,
                                 TypedValue<RankedTensorType> scale,
                                 unsigned side) {
  auto producer =
      selectScaleProducer(rewriter, dotOp, scale, side, /*preferDirect=*/false);
  auto *ctx = dotOp.getContext();
  auto ctaLayout = ttg::getCTALayout(dotOp.getType().getEncoding());
  auto shape = scale.getType().getShape();
  Value source = scale;
  while (auto convert = source.getDefiningOp<ttg::ConvertLayoutOp>())
    source = convert.getSrc();
  auto load = source.getDefiningOp<tt::LoadOp>();
  if (!producer) {
    // Keep the late diagnostic for unsupported copy/MMAC geometries.
    auto shared =
        ttg::SwizzledSharedEncodingAttr::get(ctx, 4, 1, 1, {1, 0}, ctaLayout);
    auto memType =
        ttg::MemDescType::get(shape, rewriter.getI8Type(), shared,
                              ttg::SharedMemorySpaceAttr::get(ctx), true);
    auto alloc =
        rewriter.create<ttg::LocalAllocOp>(scale.getLoc(), memType, source);
    return rewriter.create<ttg::LocalLoadOp>(scale.getLoc(), scale.getType(),
                                             alloc);
  }
  const auto &selected = *producer;
  Value regSource = source;
  if (selected.regType != source.getType()) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(load);
    Value ptr = rewriter.create<ttg::ConvertLayoutOp>(
        load.getLoc(),
        cast<RankedTensorType>(load.getPtr().getType())
            .cloneWithEncoding(selected.regType.getEncoding()),
        load.getPtr());
    Operation *newLoad = rewriter.clone(*load);
    newLoad->setOperand(0, ptr);
    newLoad->getResult(0).setType(selected.regType);
    regSource = newLoad->getResult(0);
  }
  OpBuilder::InsertionGuard guard(rewriter);
  // Constant unit scales are loop invariant. Keep their allocation outside
  // the loop so the ordinary pipeliner need not predicate initialization.
  if (auto constant = regSource.getDefiningOp<arith::ConstantOp>()) {
    if (auto loop = dotOp->getParentOfType<scf::ForOp>()) {
      rewriter.setInsertionPoint(loop);
      regSource = rewriter.clone(*constant)->getResult(0);
    }
  }
  auto alloc = rewriter.create<ttg::LocalAllocOp>(scale.getLoc(),
                                                  selected.memType, regSource);
  rewriter.setInsertionPoint(dotOp);
  auto local =
      rewriter.create<ttg::LocalLoadOp>(scale.getLoc(), scale.getType(), alloc);
  return local;
}

// Batch ring reads before either EMPTY notification so one lgkm wait can
// retire both copies. Never cross another memory access or a READY wait.
static Operation *getScaleWriteAnchor(
    ttg::LocalLoadOp a, ttg::LocalLoadOp b, DominanceInfo &dominance,
    const llvm::SmallPtrSetImpl<Operation *> &prepared) {
  if (!a || !b || a->getBlock() != b->getBlock())
    return nullptr;
  if (a == b)
    return a;
  auto first = a->isBeforeInBlock(b) ? a : b;
  auto last = first == a ? b : a;
  // Private LDS chains must finish both initializations before either copy.
  if (prepared.contains(a) && prepared.contains(b))
    return last;
  for (Operation *op = first->getNextNode(); op != last;
       op = op->getNextNode())
    if (!isa<ROCDL::HCUAbarrierArriveOp>(op) && !isMemoryEffectFree(op))
      return nullptr;
  if (last.getToken() && !dominance.dominates(last.getToken(), first))
    return nullptr;
  if (!dominance.dominates(last.getSrc(), first)) {
    auto view = last.getSrc().getDefiningOp<ttg::MemDescIndexOp>();
    if (!view || !llvm::all_of(view->getOperands(), [&](Value operand) {
          return dominance.dominates(operand, first);
        }))
      return nullptr;
    view->moveBefore(first);
  }
  return first;
}

static LogicalResult
lowerScaledDots(ModuleOp module, ArrayRef<tt::DotScaledOp> dots,
                const llvm::SmallPtrSetImpl<Operation *> &prepared) {
  IRRewriter builder(module.getContext());
  DominanceInfo dominance(module);
  for (auto dot : dots) {
    auto aLoad = dot.getAScale().getDefiningOp<ttg::LocalLoadOp>();
    auto bLoad = dot.getBScale().getDefiningOp<ttg::LocalLoadOp>();
    Operation *anchor = getScaleWriteAnchor(aLoad, bLoad, dominance, prepared);
    SmallVector<Value> scales;
    llvm::SetVector<Operation *> oldLoads;
    for (Value scale : {dot.getAScale(), dot.getBScale()}) {
      auto load =
          scale ? scale.getDefiningOp<ttg::LocalLoadOp>() : ttg::LocalLoadOp{};
      if (!load) {
        dot.emitError(
            "HCU scale lowering requires standard LDS local_load operands");
        return failure();
      }
      Operation *position = anchor ? anchor : load.getOperation();
      builder.setInsertionPoint(position);
      Value pred;
      // The pipeliner leaves local_load outside a guarded epilogue dot.
      // Preserve that CTA scope while guarding the new ds2buf work.
      Operation *child = dot;
      for (Operation *parent = dot->getParentOp(); parent;
           child = parent, parent = parent->getParentOp()) {
        auto condition = dyn_cast<scf::IfOp>(parent);
        if (!condition || condition->isAncestor(load) ||
            !dominance.dominates(condition.getCondition(), position))
          continue;
        Value guard = condition.getCondition();
        if (child->getParentRegion() == &condition.getElseRegion()) {
          Value one =
              arith::ConstantIntOp::create(builder, load.getLoc(), 1, 1);
          guard = arith::XOrIOp::create(builder, load.getLoc(), guard, one);
        }
        pred = pred ? arith::AndIOp::create(builder, load.getLoc(), pred, guard)
                    : guard;
      }
      // The source memdesc fixes LDS order. Record the final copy geometry
      // only in the Scale Buffer encoding, shared by later passes.
      auto shared = dyn_cast<ttg::SwizzledSharedEncodingAttr>(
          load.getSrc().getType().getEncoding());
      if (!shared)
        return dot.emitError("scale copy requires a swizzled_shared LDS encoding");
      auto mma = cast<ttg::AMDMfmaEncodingAttr>(dot.getType().getEncoding());
      unsigned side = scales.size();
      auto plan = HCU::ScaleMemoryPlan::select(
          load.getType(), mma.getInstrShape()[2] / 32,
          mma.getTilesPerWarp()[side], mma.getWarpsPerCTA()[side],
          shared.getOrder()[0]);
      if (!plan)
        return dot.emitError("no legal Scale Buffer copy/MMAC layout");
      plan->bankXor = side;
      auto scaleType = ttg::MemDescType::get(
          load.getType().getShape(), builder.getI8Type(),
          plan->getEncoding(builder.getContext()),
          tta::ScaleMemorySpaceAttr::get(builder.getContext()), false);
      auto write = tta::ScaleWriteOp::create(
          builder, load.getLoc(), scaleType, load.getSrc(), pred,
          load.getToken(), builder.getI32IntegerAttr(plan->getCopyMode()),
          IntegerAttr{});
      scales.push_back(write);
      oldLoads.insert(load);
    }
    builder.setInsertionPoint(dot);
    auto converted = tta::DotScaledOp::create(
        builder, dot.getLoc(), dot.getType(), dot.getA(), dot.getB(),
        dot.getC(), scales[0], scales[1], dot.getAElemType(),
        dot.getBElemType(), dot.getFastMath(), dot.getLhsKPack(),
        dot.getRhsKPack());
    // The builder owns semantic attributes; retain only scheduling annotations.
    converted->setDiscardableAttrs(dot->getDiscardableAttrDictionary());
    builder.replaceOp(dot, converted.getResult());
    for (Operation *load : oldLoads)
      if (load->use_empty())
        builder.eraseOp(load);
  }
  return success();
}

// A scale's value is transported by the existing pipeliner. Follow only its
// standard conversions and for-loop SSA; computed scales keep the REG path.
struct ScaleValues {
  llvm::SetVector<Value> values;
  llvm::SetVector<Operation *> loads;
  scf::ForOp loop;
  bool collect(Value value) {
    if (!values.insert(value))
      return true;
    if (auto cvt = value.getDefiningOp<ttg::ConvertLayoutOp>())
      return collect(cvt.getSrc());
    if (auto load = value.getDefiningOp<tt::LoadOp>()) {
      loads.insert(load);
      return true;
    }
    scf::ForOp loop;
    unsigned index;
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      loop = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp());
      if (!loop || !arg.getArgNumber())
        return false;
      index = arg.getArgNumber() - 1;
    } else if (auto result = dyn_cast<OpResult>(value)) {
      loop = dyn_cast<scf::ForOp>(result.getOwner());
      if (!loop)
        return false;
      index = result.getResultNumber();
    } else {
      return false;
    }
    if (this->loop && this->loop != loop)
      return false;
    this->loop = loop;
    return collect(loop.getRegionIterArg(index)) &&
           collect(loop.getInitArgs()[index]) &&
           collect(loop.getBody()->getTerminator()->getOperand(index));
  }
};

static bool isMlsOperand(Value value) {
  while (auto cvt = value.getDefiningOp<ttg::ConvertLayoutOp>())
    value = cvt.getSrc();
  if (value.getDefiningOp<tt::MatrixLoadOp>())
    return true;
  if (auto load = value.getDefiningOp<ttg::LocalLoadOp>())
    return isa<ttg::HCUMlsSharedEncodingAttr>(
        load.getSrc().getType().getEncoding());
  return false;
}

// Resolve a WASP LDS view to the allocation captured by all partitions.
static Value getScaleAllocation(Value view) {
  if (auto index = view.getDefiningOp<ttg::MemDescIndexOp>())
    return getScaleAllocation(index.getSrc());
  if (auto slice = view.getDefiningOp<ttg::MemDescSubsliceOp>())
    return getScaleAllocation(slice.getSrc());
  if (auto arg = dyn_cast<BlockArgument>(view)) {
    if (auto partitions = dyn_cast<ttg::WarpSpecializePartitionsOp>(
            arg.getOwner()->getParentOp())) {
      auto ws = cast<ttg::WarpSpecializeOp>(partitions->getParentOp());
      return getScaleAllocation(ws.getExplicitCaptures()[arg.getArgNumber()]);
    }
  }
  return view;
}

// Put address calculations before the matrix load so its scale copy is the
// next memory operation. Never hoist a dependent memory access or cross blocks.
static bool placeScaleCopyAfter(ttg::AsyncCopyGlobalToLocalOp copy,
                                Operation *matrix) {
  llvm::SetVector<Operation *> dependencies;
  std::function<LogicalResult(Value)> collect = [&](Value value) {
    auto *op = value.getDefiningOp();
    if (!op || op == matrix || op->getBlock() != matrix->getBlock() ||
        op->isBeforeInBlock(matrix))
      return success();
    if (dependencies.contains(op))
      return success();
    if (!isPure(op))
      return failure();
    for (Value operand : op->getOperands())
      if (failed(collect(operand)))
        return failure();
    dependencies.insert(op);
    return success();
  };
  for (Value operand : copy->getOperands())
    if (failed(collect(operand)))
      return false;
  for (Operation *op : dependencies)
    op->moveBefore(matrix);
  copy->moveAfter(matrix);
  return true;
}

static bool isAbarrierTracked(Operation *op) {
  for (auto *previous = op->getPrevNode(); previous;
       previous = previous->getPrevNode()) {
    if (isa<ROCDL::HCUAbarrierArriveOp>(previous))
      return false; // Arrive closes the common hardware track interval.
    if (isa<ROCDL::HCUAbarrierSeqOp>(previous))
      return true;
  }
  return false;
}

// WASP has already placed the scale load/store in the producer partition.
// Replace only that transfer: keep its LDS layout, ring index and READY/EMPTY
// protocol. A copy immediately after its tracked MLS shares the same READY.
// Only untracked copies need an explicit completion wait before that arrival.
static void prepareWaspScaleProducer(IRRewriter &builder, tt::DotScaledOp dot,
                                     ttg::LocalLoadOp local, unsigned side) {
  auto ws = dot->getParentOfType<ttg::WarpSpecializeOp>();
  if (!ws)
    return;
  Value matrix = side == 0 ? dot.getA() : dot.getB();
  while (auto convert = matrix.getDefiningOp<ttg::ConvertLayoutOp>())
    matrix = convert.getSrc();
  auto matrixLoad = matrix.getDefiningOp<ttg::LocalLoadOp>();
  Value matrixAllocation =
      matrixLoad ? getScaleAllocation(matrixLoad.getSrc()) : Value{};
  Value allocation = getScaleAllocation(local.getSrc());
  SmallVector<ttg::LocalStoreOp> stores;
  ws.walk([&](ttg::LocalStoreOp store) {
    if (getScaleAllocation(store.getDst()) == allocation)
      stores.push_back(store);
  });
  for (auto store : stores) {
    auto load = store.getSrc().getDefiningOp<tt::LoadOp>();
    if (!load)
      continue;
    auto selected = selectScaleProducer(
        builder, dot, cast<TypedValue<RankedTensorType>>(load.getResult()),
        side,
        /*preferDirect=*/true, store.getDst().getType());
    if (!selected || !selected->direct)
      continue;
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(store);
    auto ptrType = cast<RankedTensorType>(load.getPtr().getType())
                       .cloneWithEncoding(selected->regType.getEncoding());
    Value ptr = ttg::ConvertLayoutOp::create(builder, load.getLoc(), ptrType,
                                             load.getPtr());
    Value mask = load.getMask();
    if (mask) {
      auto type = cast<RankedTensorType>(mask.getType())
                      .cloneWithEncoding(selected->regType.getEncoding());
      mask = ttg::ConvertLayoutOp::create(builder, load.getLoc(), type, mask);
    }
    auto copy = ttg::AsyncCopyGlobalToLocalOp::create(
        builder, load.getLoc(), ptr, store.getDst(), mask, Value{},
        load.getCache(), load.getEvict(), load.getIsVolatile());
    copy.setContiguity(selected->vector);
    // MLS has acquired its EMPTY slot before writing LDS. Follow that load,
    // rather than retaining the old register-to-LDS store's later position.
    Operation *anchor = nullptr;
    ws.walk([&](tta::MatrixLoadToLocalOp matrix) {
      if (matrix->getBlock() == store->getBlock() &&
          matrix->isBeforeInBlock(copy) &&
          getScaleAllocation(matrix.getDest()) == matrixAllocation)
        anchor = matrix;
    });
    bool tracked =
        anchor && placeScaleCopyAfter(copy, anchor) && isAbarrierTracked(copy);
    // seq/arrive tracks non-bypass buffer loads too. Do not drain that
    // asynchronous work before READY; consumers wait on the existing abarrier.
    if (!tracked)
      tta::MemoryCounterWaitOp::create(builder, store.getLoc(),
                                       /*load=*/builder.getI32IntegerAttr(0),
                                       /*store=*/nullptr, /*ds=*/nullptr);
    builder.eraseOp(store);
    if (load->use_empty())
      builder.eraseOp(load);
  }
}

struct ScaleStorage {
  Value view;
  Value token;
};

// Preserve the pipeliner's producer issue positions. Numeric loop parameters
// become matching LDS-view/token parameters only when native copies are legal.
class ScaleDataTransfer {
  IRRewriter &builder;
  DenseMap<Value, ScaleStorage> producers;
  DenseMap<Value, unsigned> carried;

  ScaleStorage getStorage(Value value) {
    if (auto it = producers.find(value); it != producers.end())
      return it->second;
    if (auto cvt = value.getDefiningOp<ttg::ConvertLayoutOp>())
      return getStorage(cvt.getSrc());
    bool result = false;
    BlockArgument arg;
    if (auto blockArg = dyn_cast<BlockArgument>(value)) {
      arg = blockArg;
    } else {
      auto opResult = cast<OpResult>(value);
      arg = cast<scf::ForOp>(opResult.getOwner())
                .getRegionIterArg(opResult.getResultNumber());
      result = true;
    }
    auto loop = cast<scf::ForOp>(arg.getOwner()->getParentOp());
    unsigned index = arg.getArgNumber() - 1;
    auto it = carried.find(arg);
    if (it == carried.end()) {
      auto initial = getStorage(loop.getInitArgs()[index]);
      unsigned count = loop.getNumRegionIterArgs();
      loop = addIterArgsToLoop(builder, loop, {initial.view, initial.token});
      carried[arg] = count;
      auto next =
          getStorage(loop.getBody()->getTerminator()->getOperand(index));
      appendToForOpYield(loop, {next.view, next.token});
      it = carried.find(arg);
    }
    // Other scale operands can extend the same loop later. Resolve its current
    // operation through the stable body argument instead of caching results.
    loop = cast<scf::ForOp>(arg.getOwner()->getParentOp());
    unsigned count = it->second;
    if (result)
      return {loop.getResult(count), loop.getResult(count + 1)};
    return {loop.getRegionIterArg(count), loop.getRegionIterArg(count + 1)};
  }

public:
  explicit ScaleDataTransfer(IRRewriter &builder) : builder(builder) {}

  Value prepare(tt::DotScaledOp dot, TypedValue<RankedTensorType> scale,
                unsigned side, bool preferDirect) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(dot);
    // Existing explicit LDS input already describes the LDS storage layout
    // and producer.
    if (auto local = scale.getDefiningOp<ttg::LocalLoadOp>()) {
      if (preferDirect)
        prepareWaspScaleProducer(builder, dot, local, side);
      return scale;
    }
    // An epilogue and loop body may consume the same prefetched scale chain.
    ScaleValues graph;
    bool transported = graph.collect(scale) && !graph.loads.empty();
    if (!transported ||
        llvm::any_of(
            graph.values,
            [](Value value) {
              return llvm::any_of(value.getUses(), [](OpOperand &use) {
                Operation *user = use.getOwner();
                if (isa<tt::DotScaledOp>(user))
                  return use.getOperandNumber() <
                         3; // A/B/C are not scale transport.
                return !isa<ttg::ConvertLayoutOp, scf::ForOp, scf::YieldOp>(
                    user);
              });
            }) ||
        (graph.loop && llvm::any_of(graph.loads, [&](Operation *load) {
           auto parent = load->getParentOfType<scf::ForOp>();
           return parent && parent != graph.loop;
         })))
      return prepareScaleStorage(builder, dot, scale, side);
    auto hasStorage = [&](Operation *load) {
      return producers.contains(load->getResult(0));
    };
    bool alreadyPrepared = llvm::all_of(graph.loads, hasStorage);
    // A partially shared transport graph has no single native ring contract.
    if (!alreadyPrepared && llvm::any_of(graph.loads, hasStorage))
      return prepareScaleStorage(builder, dot, scale, side);
    RankedTensorType loaderType;
    ttg::MemDescType memoryType;
    unsigned vector = 1;
    bool direct = false;
    if (!alreadyPrepared) {
      auto first = cast<tt::LoadOp>(graph.loads.front());
      // Reuse the same layout selector for static and loop-carried values.
      auto selected = selectScaleProducer(
          builder, dot, cast<TypedValue<RankedTensorType>>(first.getResult()),
          side, preferDirect);
      if (!selected)
        return prepareScaleStorage(builder, dot, scale, side);
      loaderType = selected->regType;
      memoryType = selected->memType;
      direct = selected->direct;
      vector = selected->vector;
      // Re-encode only a private scale transport graph. Never introduce a
      // register conversion of scale payloads to repair an existing schedule.
      for (Value value : graph.values)
        value.setType(cast<RankedTensorType>(value.getType())
                          .cloneWithEncoding(loaderType.getEncoding()));
      for (Operation *operation : graph.loads) {
        auto load = cast<tt::LoadOp>(operation);
        builder.setInsertionPoint(load);
        auto convert = [&](Value value) -> Value {
          if (!value)
            return {};
          auto type = cast<RankedTensorType>(value.getType())
                          .cloneWithEncoding(loaderType.getEncoding());
          if (value.getType() == type)
            return value;
          return ttg::ConvertLayoutOp::create(builder, load.getLoc(), type,
                                              value);
        };
        load.getPtrMutable().assign(convert(load.getPtr()));
        if (load.getMask())
          load.getMaskMutable().assign(convert(load.getMask()));
        if (load.getOther())
          load.getOtherMutable().assign(convert(load.getOther()));
      }
    } else {
      direct = true;
    }
    if (!direct) {
      builder.setInsertionPoint(dot);
      auto alloc =
          ttg::LocalAllocOp::create(builder, scale.getLoc(), memoryType, scale);
      auto local = ttg::LocalLoadOp::create(builder, scale.getLoc(),
                                            scale.getType(), alloc);
      return local;
    }
    if (!alreadyPrepared) {
      scf::ForOp loop;
      int64_t prologue = 0;
      for (Operation *load : graph.loads) {
        if (auto parent = load->getParentOfType<scf::ForOp>())
          loop = parent;
        else
          ++prologue;
      }
      int64_t depth = loop ? prologue + 1 : 1;
      SmallVector<int64_t> shape(memoryType.getShape());
      if (loop)
        shape.insert(shape.begin(), depth);
      auto ringType = ttg::MemDescType::get(shape, memoryType.getElementType(),
                                            memoryType.getEncoding(),
                                            memoryType.getMemorySpace(), true);
      builder.setInsertionPoint(
          graph.loads.front()->getParentOfType<scf::ForOp>()
              ? loop.getOperation()
              : graph.loads.front());
      auto ring = ttg::LocalAllocOp::create(builder, dot.getLoc(), ringType);
      int64_t initial = 0;
      for (Operation *operation : graph.loads) {
        auto load = cast<tt::LoadOp>(operation);
        builder.setInsertionPoint(load);
        Value view = ring;
        if (loop) {
          Value index;
          if (auto parent = load->getParentOfType<scf::ForOp>()) {
            auto type = parent.getInductionVar().getType();
            Value iteration = arith::SubIOp::create(builder, load.getLoc(),
                                                    parent.getInductionVar(),
                                                    parent.getLowerBound());
            iteration = arith::DivSIOp::create(builder, load.getLoc(),
                                               iteration, parent.getStep());
            Value offset = arith::ConstantOp::create(
                builder, load.getLoc(), builder.getIntegerAttr(type, prologue));
            Value count = arith::ConstantOp::create(
                builder, load.getLoc(), builder.getIntegerAttr(type, depth));
            index = arith::AddIOp::create(builder, load.getLoc(), iteration,
                                          offset);
            index =
                arith::RemUIOp::create(builder, load.getLoc(), index, count);
            if (type.isIndex())
              index = arith::IndexCastOp::create(builder, load.getLoc(),
                                                 builder.getI32Type(), index);
            else if (!type.isInteger(32))
              index = arith::TruncIOp::create(builder, load.getLoc(),
                                              builder.getI32Type(), index);
          } else {
            index = arith::ConstantIntOp::create(builder, load.getLoc(),
                                                 initial++, 32);
          }
          view = ttg::MemDescIndexOp::create(builder, load.getLoc(), memoryType,
                                             ring, index);
        }
        auto copy = ttg::AsyncCopyGlobalToLocalOp::create(
            builder, load.getLoc(), load.getPtr(), view, load.getMask(),
            load.getOther(), load.getCache(), load.getEvict(),
            load.getIsVolatile());
        copy.setContiguity(vector);
        auto group = ttg::AsyncCommitGroupOp::create(builder, load.getLoc(),
                                                     copy.getToken());
        producers[load.getResult()] = {view, group.getResult()};
      }
    }
    auto current = getStorage(scale);
    builder.setInsertionPoint(dot);
    auto inputLoad = [](Value operand) {
      while (auto cvt = operand.getDefiningOp<ttg::ConvertLayoutOp>())
        operand = cvt.getSrc();
      return operand.getDefiningOp<ttg::LocalLoadOp>();
    };
    auto aLoad = inputLoad(dot.getA()), bLoad = inputLoad(dot.getB());
    auto wait = aLoad && aLoad.getToken()
                    ? aLoad.getToken().getDefiningOp<ttg::AsyncWaitOp>()
                    : ttg::AsyncWaitOp{};
    DominanceInfo dominance(dot->getParentOfType<ModuleOp>());
    // Join current-tile scale readiness with the existing A/B wait. Keeping
    // separate waits adds another CTA barrier after the fragment reads. Only
    // merge when the scale token is already available at the common wait;
    // updateWaits() recomputes its pending-group count after both sides join.
    if (wait && bLoad && bLoad.getToken() == wait.getResult() &&
        wait->getBlock() == dot->getBlock() &&
        dominance.dominates(current.token, wait)) {
      if (!llvm::is_contained(wait.getAsyncToken(), current.token))
        wait.getAsyncTokenMutable().append(current.token);
      // Put the scale read just after its matrix fragment load, in the same
      // consumer cluster, so ds2buf can overlap the following prefetch work.
      auto anchor = side == 0 ? aLoad : bLoad;
      if (anchor->getBlock() == dot->getBlock() &&
          dominance.dominates(current.view, anchor))
        builder.setInsertionPointAfter(anchor);
    } else {
      wait = ttg::AsyncWaitOp::create(builder, dot.getLoc(), current.token, 0);
    }
    auto local =
        ttg::LocalLoadOp::create(builder, scale.getLoc(), scale.getType(),
                                 current.view, wait.getResult());
    return local;
  }
};

struct ScalePipelinePass : impl::TritonHCUScalePipelineBase<ScalePipelinePass> {
  using Base::Base;

  // Run after software pipelining, while dots still consume numeric scales.
  // Preserve the existing matrix schedule according to two rules:
  // 1. Each scale Global load follows its own A/B Global-load stage and
  //    cluster, immediately after that load; keep prefetch and ring rotation.
  // 2. Both scale_writes follow the current tile's local_load consumer stage
  //    and cluster, after BOTH A/B fragment loads have issued. Issue them early
  //    to overlap ds2buf with the following prefetch work, not next to the dot.
  // Reuse the current-tile readiness wait where its dependencies allow it.
  // Later synchronization analysis publishes ds2buf with lgkmcnt(0) + sync
  // before MMAC. Physical Scale Buffer rows are also allocated by a later pass.
  void runOnOperation() override {
    auto module = getOperation();
    IRRewriter builder(&getContext());
    ScaleDataTransfer scaleTransfer(builder);
    llvm::SmallPtrSet<Operation *, 16> prepared;
    SmallVector<tt::DotScaledOp> dots;
    // 1. Collect version-4 MFMA dots that have at least one scale. Dots with
    // no scales retain the numeric path and do not need Scale Buffer storage.
    module.walk([&](tt::DotScaledOp dot) {
      auto mma =
          dyn_cast<ttg::AMDMfmaEncodingAttr>(dot.getType().getEncoding());
      if (mma && mma.getVersion() == 4 && (dot.getAScale() || dot.getBScale()))
        dots.push_back(dot);
    });
    // Ordinary modules and other matrix families retain their existing waits.
    if (dots.empty())
      return;
    for (auto dot : dots) {
      builder.setInsertionPoint(dot);
      auto a = dot.getAScale(), b = dot.getBScale();
      // 2. Fill a missing side with E8M0 0x7f (numeric 1). Match the other
      // side's K-scale count/layout, but use this side's M or N dimension.
      auto unit = [&](unsigned side, TypedValue<RankedTensorType> other) {
        auto type = RankedTensorType::get(
            {dot.getType().getShape()[side], other.getType().getShape()[1]},
            builder.getI8Type(), other.getType().getEncoding());
        return cast<TypedValue<RankedTensorType>>(
            arith::ConstantOp::create(
                builder, dot.getLoc(), type,
                DenseElementsAttr::get(type, llvm::APInt(8, 0x7f)))
                .getResult());
      };
      if (!a)
        a = unit(0, b);
      if (!b)
        b = unit(1, a);
      // 3. Prefer Global -> LDS for explicit async requests or full MLS.
      // This is a preference: prepare() still checks whether direct copy is
      // legal and falls back to Global -> REG -> LDS when necessary.
      bool direct = useAsyncCopy ||
                    (isMlsOperand(dot.getA()) && isMlsOperand(dot.getB()));
      // 4. Prepare A/B LDS inputs at the existing producer positions. Reuse
      // explicit local_loads (including WASP); otherwise build the REG/direct
      // path and carry LDS views/tokens through existing loops as needed.
      // Record newly prepared loads for the later scale-write placement.
      auto prepare = [&](TypedValue<RankedTensorType> scale, unsigned side) {
        Value local = scaleTransfer.prepare(dot, scale, side, direct);
        if (local != scale)
          prepared.insert(local.getDefiningOp());
        return local;
      };
      dot.getAScaleMutable().assign(prepare(a, 0));
      // Extending loop results can replace a B operand captured above.
      b = dot.getBScale() ? dot.getBScale() : b;
      dot.getBScaleMutable().assign(prepare(b, 1));
    }
    // 5. Select ds2buf geometry from the final LDS layout, encode it in Scale
    // memdescs, and replace tt.dot_scaled with amdg.dot_scaled. Scale writes
    // retain the source token/predicate; unused numeric local_loads are erased.
    if (failed(lowerScaledDots(module, dots, prepared)))
      return signalPassFailure();
    // 6. Recompute async-group wait counts after adding/reusing scale copies.
    // This updates async waits; Scale Buffer barriers are analyzed later.
    tt::updateWaits(module);
  }
};
} // namespace
} // namespace mlir
