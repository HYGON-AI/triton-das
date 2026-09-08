#include "third_party/amd/lib/TritonAMDGPUToLLVM/TargetInfo.h"
#include "third_party/amd/lib/TritonAMDGPUToLLVM/BufferOpsEmitter.h"
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "third_party/amd/lib/TritonAMDGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "third_party/amd/lib/TritonAMDGPUToLLVM/TargetInfo.h"
#include "TritonHCU/MlsGroup.h"
#include "third_party/amd/lib/TritonAMDGPUToLLVM/Utility.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
#include "nvidia/backend/include/cuda.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "llvm/ADT/MapVector.h"

using namespace mlir;
using namespace mlir::triton::gpu;
using namespace mlir::triton::HCU;

using ::mlir::LLVM::delinearize;
using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::triton::amdgpu::MlsEncodingAttr;
using ::mlir::triton::amdgpu::MlsStoreEncodingAttr;
using ::mlir::triton::gpu::HCUMlsSharedEncodingAttr;

namespace {

// ===----------------------------------------------------------------------===//
// Utility functions
// ===----------------------------------------------------------------------===//

SmallVector<unsigned> getShapePerCTA(ArrayRef<unsigned> mlsTileG,
                                     ArrayRef<unsigned> warpsPerCTA) {
  assert(mlsTileG.size() == warpsPerCTA.size() && mlsTileG.size() == 2);

  return {warpsPerCTA[0] * mlsTileG[0], warpsPerCTA[1] * mlsTileG[1]};
}

SmallVector<unsigned> getNumReps(const MlsInsn &mlsInsn,
                                 ArrayRef<unsigned> warpsPerCTA,
                                 ArrayRef<int64_t> shapeG) {
  auto rank = shapeG.size();
  assert(rank == 2);

  SmallVector<unsigned> numReps(rank);
  auto shapePerCTA = getShapePerCTA(
          mlsInsn.getMlsTile(MlsTileKind::Global), warpsPerCTA);
  for (unsigned d = 0; d < rank; ++d) {
    numReps[d] = std::max<unsigned>(1, shapeG[d] / shapePerCTA[d]);
  }
  return numReps;
}

bool isKMajor(llvm::ArrayRef<unsigned> order, int opIdx) {
  auto rank = order.size();
  int kdim = opIdx == 0 ? rank - 1 : rank - 2;
  return order[0] == kdim;
}

Value getLinearWarpId(Location loc, RewriterBase &rewriter,
                      const AMD::TargetInfo &targetInfo) {
  unsigned warpSize = targetInfo.getWarpSize();
  assert(warpSize == 64);

  auto insertPt = rewriter.saveInsertionPoint();

  // Hoist warp-id to the LLVM func entry when possible. Do not hoist across
  // IsolatedFromAbove (ttg.warp_specialize partitions): entry SSA is invisible
  // inside those regions, which breaks MLS+WASP lowering.
  LLVM::LLVMFuncOp funcOp = nullptr;
  for (Operation *op = insertPt.getBlock()->getParentOp(); op;
       op = op->getParentOp()) {
    if (op->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
      funcOp = nullptr;
      break;
    }
    if ((funcOp = dyn_cast<LLVM::LLVMFuncOp>(op)))
      break;
  }
  if (funcOp)
    rewriter.setInsertionPointToStart(&funcOp.getBody().front());

  auto entryBuilder = TritonLLVMOpBuilder(loc, rewriter);
  Value threadId = getThreadId(rewriter, loc);
  Value warpId = entryBuilder.udiv(threadId, entryBuilder.i32_val(warpSize));
  auto call = LLVM::createLLVMIntrinsicCallOp(
      rewriter, loc, "llvm.amdgcn.readfirstlane", {i32_ty}, {warpId});
  rewriter.restoreInsertionPoint(insertPt);

  return call.getResult(0);
}

} // namespace

//--------------------------------------------------------------------------------------------------

namespace {

struct MLSMatrixLoadToLocalOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::MatrixLoadToLocalOp> {
  using ConvertOpToLLVMPattern<
      triton::amdgpu::MatrixLoadToLocalOp>::ConvertOpToLLVMPattern;

  MLSMatrixLoadToLocalOpConversion(LLVMTypeConverter &converter,
                                   const AMD::TargetInfo &targetInfo,
                                   ModuleAxisInfoAnalysis &axisAnalysisPass,
                                   PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::amdgpu::MatrixLoadToLocalOp>(converter,
                                                                    benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::amdgpu::MatrixLoadToLocalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    // AsyncTokenType converts to i32. This value only replaces the SSA token;
    // it is neither a predicate nor a count/completion state for the transfer.
    Value convertedToken = b.i32_val(0);
    // Pipeline predicates are uniform across the workgroup. Guard the actual
    // transfer, leaving async_commit_group outside: even an empty iteration
    // must keep the same mark history. Never speculate a short-loop prefetch.
    Block *continuation = nullptr;
    if (Value pred = adaptor.getPred()) {
      Block *before = rewriter.getInsertionBlock();
      continuation = rewriter.splitBlock(before, op->getIterator());
      Block *body = rewriter.createBlock(continuation);
      rewriter.setInsertionPointToEnd(before);
      LLVM::CondBrOp::create(rewriter, loc, pred, body, continuation);
      rewriter.setInsertionPointToStart(body);
    }
    bool mABIsNeed = targetInfo.getGPUKind() == llvm::AMDGPU::GPUKind::GK_GFX938;

    auto ctx = rewriter.getContext();

    auto dstTy = cast<MemDescType>(op.getDest().getType());
    auto sharedLayout =
        dyn_cast<HCUMlsSharedEncodingAttr>(dstTy.getEncoding());
    auto blockLayout = op->getAttrOfType<MlsEncodingAttr>(
        MlsEncodingAttr::getMnemonic());
    auto llvmElemTy = typeConverter->convertType(dstTy.getElementType());
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
         loc, adaptor.getDest(), llvmElemTy, rewriter);

    auto opIdx = blockLayout.getOpIdx();
    auto kDimIdx = opIdx == 0 ? 1 : 0;
    auto nonKDimIdx = opIdx == 0 ? 0 : 1;
    auto mlsInsn = MlsInsn::selectOrGetMlsInsn(
                               blockLayout.getMlsTile()[nonKDimIdx],
                               blockLayout.getMlsTile()[kDimIdx],
                               blockLayout.getElemBitWidth(),
                               static_cast<MlsElemBitTyKind>(blockLayout.getElemBitTyKind()),
                               opIdx,
                               blockLayout.getOrder()[0] == kDimIdx,
                               blockLayout.getVersion(),
                               static_cast<MlsInterleaveKind>(blockLayout.getAlt2Kind()));
    auto mlInsnAttr = mlsInsn->getMatrixLoadInsnAttr();

    auto allocOp = op.getResult().getDefiningOp<triton::gpu::LocalAllocOp>();
    auto elemByteWidth = blockLayout.getElemBitWidth() / 8;

    auto llBasePtr = adaptor.getBase();
    auto llShape = adaptor.getShape();
    auto llStrides = adaptor.getStrides();
    auto llOffsets = adaptor.getIndices();
    auto llBlockPos = adaptor.getIndices();

    auto ldBlkOff = dot64(loc, rewriter, llOffsets, llStrides);

    auto boundaryCheck = adaptor.getBoundaryCheck();
    bool needBoundaryCheck = boundaryCheck.size() > 0;
    SmallVector<bool> boundaryCheckInfo = {needBoundaryCheck && boundaryCheck[0] == 0,
                                           needBoundaryCheck && boundaryCheck[boundaryCheck.size() - 1] == 1};
    SmallVector<unsigned> numReps = getNumReps(*mlsInsn, blockLayout.getWarpsPerCTA(), dstTy.getShape());
    unsigned numRepsX = numReps[0] * mlInsnAttr.instrsPerWarp[0];
    unsigned numRepsY = numReps[1] * mlInsnAttr.instrsPerWarp[1];

    bool isBit4 = isMlsBit4ElemTyKind(static_cast<MlsElemBitTyKind>(blockLayout.getElemBitTyKind()));
    auto mlsTileE = mlsInsn->getMlsTile(MlsTileKind::Elems);
    auto mlsTileG = mlsInsn->getMlsTile(MlsTileKind::Global);
    unsigned dim0DivFactor = !isBit4 ? 1 : mlsTileE[0] / mlsTileG[0];
    unsigned dim1DivFactor = !isBit4 ? 1 : mlsTileE[1] / mlsTileG[1];


    bool useMatrixLoadStoreOffsetsWithMlOffs = true;
    if (boundaryCheckInfo[0] && boundaryCheckInfo[1]) {
      // Per mls inst need a rsrc desc for boundary padding.
      useMatrixLoadStoreOffsetsWithMlOffs = false;
    }

    if (!useMatrixLoadStoreOffsetsWithMlOffs) {
      SmallVector<std::pair<Value, Value>> ldInBlkOffCoordMappings;
      auto ldstInBlkMappings = computeMatrixLoadStoreOffsets(loc, rewriter,
                                              *mlsInsn,
                                              blockLayout, mlInsnAttr,
                                              sharedLayout,
                                              dstTy.getShape(),
                                              llStrides,
                                              mlsInsn->isRowMajor(),
                                              boundaryCheckInfo,
                                              ldInBlkOffCoordMappings);

      unsigned iterIdx = 0;
      for (auto &ldstInBlkGroup : ldstInBlkMappings) {
        Value ldInBlkOff64 = ldstInBlkGroup.first;

        // global mem address calc
        Value ldOffset = b.add(ldInBlkOff64, ldBlkOff);
        Value ldPtr = b.gep(ptr_ty(ctx, 1),
                          elemByteWidth == 1 ? i8_ty : i16_ty, llBasePtr, ldOffset);

        // create matrix load lds inst
        Value llStride = llStrides[sharedLayout.getOrder()[1]];
        llStride = b.trunc(i32_ty, llStride);

        Value paddingLenX = b.i32_val(0);
        Value paddingLenY = b.i32_val(0);
        if (boundaryCheckInfo[0]) {
          Value ldInBlockOffCoordX = ldInBlkOffCoordMappings[iterIdx].first;
          Value posEnd = b.add(b.add(llBlockPos[0], ldInBlockOffCoordX),
                               b.i32_val(mlInsnAttr.instrShape[0] / dim0DivFactor));
          Value posGt = b.icmp_sgt(posEnd, llShape[0]);
          paddingLenX = b.select(posGt, b.sub(posEnd, llShape[0]), b.i32_val(0));
        }
        if (boundaryCheckInfo[1]) {
          Value ldInBlockOffCoordY = ldInBlkOffCoordMappings[iterIdx].second;
          Value posEnd = b.add(b.add(llBlockPos[1], ldInBlockOffCoordY),
                              b.i32_val(mlInsnAttr.instrShape[1] / dim1DivFactor));
          Value posGt = b.icmp_sgt(posEnd, llShape[1]);
          paddingLenY = b.select(posGt, b.sub(posEnd, llShape[1]), b.i32_val(0));
        }

        if (isBit4) {
          paddingLenX = b.mul(paddingLenX, b.i32_val(dim0DivFactor)); // elem cnt
          paddingLenY = b.mul(paddingLenY, b.i32_val(dim1DivFactor)); // elem cnt
          llStride    = b.mul(llStride, b.i32_val(2));                // elem cnt
        }

        Value mPadding  = mlsInsn->getMajorDimIndex() == 0 ? paddingLenX : paddingLenY;
        Value nmPadding = mlsInsn->getMajorDimIndex() == 0 ? paddingLenY : paddingLenX;

        Value rsrcDesc = createRsrcDesc(loc, rewriter, mlsInsn->getMlsVersion(),
                                        ldPtr, llStride, blockLayout.getAlt2Kind(),
                                        mPadding, nmPadding);

        // share mem address calc
        Value stInBlkOff32 = ldstInBlkGroup.second;
        Value smemOff = stInBlkOff32;
        Value smemPtr = b.gep(ptr_ty(ctx, 3),
                            elemByteWidth == 1 ? i8_ty : i16_ty,
                            smemObj.getBase(),
                            smemOff);
        if (mlsInsn->isTranspose() && mABIsNeed) {
          Value stPtrInt = b.ptrtoint(i32_ty, smemPtr);
          stPtrInt = b.or_(stPtrInt, b.i32_val(0x80000000));
          smemPtr = b.inttoptr(ptr_ty(ctx, 3), stPtrInt);
        }
        bool t = mlsInsn->isRowMajor();
        bool r = false;
        generateMatrixLoadLDSOp(loc, rewriter, mlInsnAttr.insn,
                                rsrcDesc, smemPtr, t, 0, r, false, false, false);
        iterIdx++;
      }
    } else {
      // compute matrix load/store offsets with desc group and diff ml offset.
      bool mlOffsInY;
      DenseMap<Value, Value> ldInBlkOffCoordMappings;
      auto ldstInBlkMappings = computeMatrixLoadStoreOffsetsWithMlOffs(loc, rewriter,
                                              *mlsInsn,
                                              blockLayout, mlInsnAttr,
                                              sharedLayout,
                                              dstTy.getShape(),
                                              llStrides,
                                              mlsInsn->isRowMajor(),
                                              boundaryCheckInfo,
                                              mlOffsInY,
                                              ldInBlkOffCoordMappings);
      bool groupPaddingInY = !mlOffsInY;

      for (auto &ldstInBlkGroup : ldstInBlkMappings) {
        Value ldInBlkOff64 = ldstInBlkGroup.first;

        // global mem address calc
        Value ldOffset = b.add(ldInBlkOff64, ldBlkOff);
        Value ldPtr = b.gep(ptr_ty(ctx, 1),
                          elemByteWidth == 1 ? i8_ty : i16_ty, llBasePtr, ldOffset);

        // create matrix load lds inst
        Value llStride = llStrides[sharedLayout.getOrder()[1]];
        llStride = b.trunc(i32_ty, llStride);
        Value rsrcDesc;
        for (auto &ldstInBlkOff : ldstInBlkGroup.second) {
          unsigned ldInBlkOffC = ldstInBlkOff.first;
          Value stInBlkOff32   = ldstInBlkOff.second;

          if (!rsrcDesc) {
            Value paddingLen = b.i32_val(0);
            if (needBoundaryCheck) {
              int dimIdx = groupPaddingInY ? 1 : 0;
              unsigned dimDivFactor = !isBit4 ? 1 : (dimIdx == 0 ? dim0DivFactor : dim1DivFactor);
              Value ldInBlockOffCoordV = ldInBlkOffCoordMappings[ldInBlkOff64];
              Value posEnd = b.add(b.add(llBlockPos[dimIdx], ldInBlockOffCoordV),
                                   b.i32_val(mlInsnAttr.instrShape[dimIdx] / dimDivFactor));
              Value posGt = b.icmp_sgt(posEnd, llShape[dimIdx]);
              paddingLen = b.select(posGt, b.sub(posEnd, llShape[dimIdx]), b.i32_val(0));
            }

            Value mPadding  = ((groupPaddingInY && mlsInsn->isRowMajor()) ||
                               (!groupPaddingInY && !mlsInsn->isRowMajor())) ? paddingLen : b.i32_val(0);
            Value nmPadding = mPadding == paddingLen ? b.i32_val(0) : paddingLen;
            if (isBit4) {
              unsigned majorDimIdx = mlsInsn->getMajorDimIndex();
              unsigned mPaddingDivFactor = majorDimIdx == 0 ? dim0DivFactor : dim1DivFactor;
              unsigned nmPaddingDivFactor = majorDimIdx == 0 ? dim1DivFactor : dim0DivFactor;
              mPadding = b.mul(mPadding, b.i32_val(mPaddingDivFactor));    // elem cnt
              nmPadding = b.mul(nmPadding, b.i32_val(nmPaddingDivFactor)); // elem cnt
              llStride = b.mul(llStride, b.i32_val(2));                    // elem cnt
            }
            rsrcDesc = createRsrcDesc(loc, rewriter, mlsInsn->getMlsVersion(),
                                      ldPtr, llStride, blockLayout.getAlt2Kind(),
                                      mPadding, nmPadding);
          }

          // share mem address calc
          Value smemOff = stInBlkOff32;
          Value smemPtr = b.gep(ptr_ty(ctx, 3),
                              elemByteWidth == 1 ? i8_ty : i16_ty,
                              smemObj.getBase(),
                              smemOff);
          if (mlsInsn->isTranspose() && mABIsNeed) {
            Value stPtrInt = b.ptrtoint(i32_ty, smemPtr);
            stPtrInt = b.or_(stPtrInt, b.i32_val(0x80000000));
            smemPtr = b.inttoptr(ptr_ty(ctx, 3), stPtrInt);
          }

          bool t = mlsInsn->isRowMajor();
          bool r = mlOffsInY;

          if (isBit4 && (t == r))
            ldInBlkOffC = ldInBlkOffC * 2;

          assert(ldInBlkOffC < 1024 && "ldInBlkOffC out of range"); // 2^10
          generateMatrixLoadLDSOp(loc, rewriter, mlInsnAttr.insn,
                                  rsrcDesc, smemPtr, t, ldInBlkOffC, r, false, false, false);
        }
      }
    }

    if (continuation) {
      LLVM::BrOp::create(rewriter, loc, ValueRange{}, continuation);
      rewriter.setInsertionPoint(op);
    }
    rewriter.replaceOp(op, convertedToken);
    return success();
  }

private:
  friend struct MLSMatrixStoreFromRegOpConversion;

  const AMD::TargetInfo &targetInfo;

  /*
   * Calculate matrix load store offsets in blk: ldOffVal, stOffVal
   **/
  SmallVector<std::pair<Value, Value>>
  computeMatrixLoadStoreOffsets(Location loc, RewriterBase &rewriter,
                                const MlsInsn &mlsInsn,
                                const MlsEncodingAttr &blockLayout,
                                const MatrixLoadInsnAttr &mlInsnAttr,
                                const HCUMlsSharedEncodingAttr &sharedLayout,
                                ArrayRef<int64_t> tensorShape,
                                ValueRange llStrides,
                                bool mlsIsRowMajor,
                                SmallVector<bool> boundaryCheckInfo,
                                SmallVector<std::pair<Value, Value>> &ldInBlkOffCoordMappings) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto mlsTileG = mlsInsn.getMlsTile(MlsTileKind::Global);
    auto mlsTileS = mlsInsn.getMlsTile(MlsTileKind::Shared);
    auto mlsTileE = mlsInsn.getMlsTile(MlsTileKind::Elems);
    auto tensorShapeG = tensorShape;
    auto tensorShapeS =
        getMlsExpandedShape(sharedLayout, tensorShape, MlsTileKind::Shared);

    auto warpsPerCta = blockLayout.getWarpsPerCTA();
    auto shapePerCta = getShapePerCTA(mlsTileG, warpsPerCta);
    SmallVector<Value> multiDimWarpIds;
    auto dummyMapping = computeMatrixLoadStoreOffsetsMappingInCTA(
        loc, rewriter, mlsInsn, blockLayout, mlInsnAttr, tensorShape, multiDimWarpIds);

    auto dsByteOffsets = mlInsnAttr.dsByteOffsets;
    auto elemByteWidth = blockLayout.getElemBitWidth() / 8;
    SmallVector<unsigned> flattenedMlsShape(tensorShapeS.begin(), tensorShapeS.end());
    flattenedMlsShape[0] = tensorShapeS[0] / mlsTileS[0];
    flattenedMlsShape[1] = tensorShapeS[1] / mlsTileS[1];

    auto rank = tensorShape.size();
    Value offWarpX = b.mul(multiDimWarpIds[rank - 2], b.i32_val(mlsTileG[rank - 2]));
    Value offWarpY = b.mul(multiDimWarpIds[rank - 1], b.i32_val(mlsTileG[rank - 1]));

    SmallVector<unsigned> numReps = getNumReps(mlsInsn, blockLayout.getWarpsPerCTA(), tensorShapeG);
    unsigned numRepsX = numReps[0] * mlInsnAttr.instrsPerWarp[0];
    unsigned numRepsY = numReps[1] * mlInsnAttr.instrsPerWarp[1];
    bool needBoundaryCheck = boundaryCheckInfo[0] || boundaryCheckInfo[1];

    bool isBit4 = isMlsBit4ElemTyKind(static_cast<MlsElemBitTyKind>(mlsInsn.getElemBitTyKind()));
    unsigned dim0DivFactor = !isBit4 ? 1 : mlsTileE[0] / mlsTileG[0];
    unsigned dim1DivFactor = !isBit4 ? 1 : mlsTileE[1] / mlsTileG[1];

    SmallVector<std::pair<Value, Value>> loadStoreOffsets;

    for (unsigned ctaX = 0; ctaX < numReps[0]; ++ctaX) {
      for (unsigned ctaY = 0; ctaY < numReps[1]; ++ctaY) {
        for (unsigned instIdxX = 0; instIdxX < mlInsnAttr.instrsPerWarp[0]; ++instIdxX) {
          for (unsigned instIdxY = 0; instIdxY < mlInsnAttr.instrsPerWarp[1]; ++instIdxY) {
            unsigned ctaRow = ctaX * shapePerCta[0];
            unsigned ctaCol = ctaY * shapePerCta[1];
            Value ldRow = b.add(offWarpX, b.i32_val(ctaRow + instIdxX * mlInsnAttr.instrShape[0] / dim0DivFactor));
            Value ldCol = b.add(offWarpY, b.i32_val(ctaCol + instIdxY * mlInsnAttr.instrShape[1] / dim1DivFactor));

            Value ldOffV = dot64(loc, rewriter, {ldRow, ldCol}, llStrides);
            Value stMlsRow = b.add(b.i32_val(ctaX * warpsPerCta[0]), multiDimWarpIds[0]);
            Value stMlsCol = b.add(b.i32_val(ctaY * warpsPerCta[1]), multiDimWarpIds[1]);

            Value stMlsOff = b.mul(linearize(rewriter, loc, {stMlsRow, stMlsCol},
                                flattenedMlsShape, sharedLayout.getOrder()),
                                b.i32_val(product(mlsTileS)));
            unsigned instIdx = instIdxX * mlInsnAttr.instrsPerWarp[1] + instIdxY;
            Value stInMlsOff = b.i32_val(dsByteOffsets[instIdx]/elemByteWidth); // element offset in mls
            Value stOffV = b.add(stMlsOff, stInMlsOff);

            loadStoreOffsets.push_back(std::make_pair(ldOffV, stOffV));
            ldInBlkOffCoordMappings.push_back(std::make_pair(ldRow, ldCol));
          }
        }
      }
    }

    return loadStoreOffsets;
  }


  /*
   * Calculate matrix load with mls offset field.
   * For each vector, share same rsrc desc with ldOffValue, but has different ldOffConst offset & stOffValue.
   * key: ldOffValue, value: std::pair<ldOffConst, stOffValue>
   **/
  DenseMap<Value, SmallVector<std::pair<unsigned, Value>>>
  computeMatrixLoadStoreOffsetsWithMlOffs(Location loc, RewriterBase &rewriter,
                                const MlsInsn &mlsInsn,
                                const MlsEncodingAttr &blockLayout,
                                const MatrixLoadInsnAttr &mlInsnAttr,
                                const HCUMlsSharedEncodingAttr &sharedLayout,
                                ArrayRef<int64_t> tensorShape,
                                ValueRange llStrides,
                                bool mlsIsRowMajor,
                                SmallVector<bool> boundaryCheckInfo,
                                bool &mlOffsInY,     /* key: ldOffValue, value: ldCoordValue */
                                DenseMap<Value, Value> &ldInBlkOffCoordMappings) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    auto mlsTileG = mlsInsn.getMlsTile(MlsTileKind::Global);
    auto mlsTileS = mlsInsn.getMlsTile(MlsTileKind::Shared);
    auto mlsTileE = mlsInsn.getMlsTile(MlsTileKind::Elems);
    auto tensorShapeG = tensorShape;
    auto tensorShapeS =
        getMlsExpandedShape(sharedLayout, tensorShape, MlsTileKind::Shared);

    auto warpsPerCta = blockLayout.getWarpsPerCTA();
    auto shapePerCta = getShapePerCTA(mlsTileG, warpsPerCta);
    SmallVector<Value> multiDimWarpIds;
    auto dummyMapping = computeMatrixLoadStoreOffsetsMappingInCTA(
        loc, rewriter, mlsInsn, blockLayout, mlInsnAttr, tensorShape, multiDimWarpIds);

    auto dsByteOffsets = mlInsnAttr.dsByteOffsets;
    auto elemByteWidth = blockLayout.getElemBitWidth() / 8;
    SmallVector<unsigned> flattenedMlsShape(tensorShapeS.begin(), tensorShapeS.end());
    flattenedMlsShape[0] = tensorShapeS[0] / mlsTileS[0];
    flattenedMlsShape[1] = tensorShapeS[1] / mlsTileS[1];

    auto rank = tensorShape.size();
    Value offWarpX = b.mul(multiDimWarpIds[rank - 2], b.i32_val(mlsTileG[rank - 2]));
    Value offWarpY = b.mul(multiDimWarpIds[rank - 1], b.i32_val(mlsTileG[rank - 1]));

    SmallVector<unsigned> numReps = getNumReps(mlsInsn, warpsPerCta, tensorShapeG);
    unsigned numRepsX = numReps[0] * mlInsnAttr.instrsPerWarp[0];
    unsigned numRepsY = numReps[1] * mlInsnAttr.instrsPerWarp[1];
    bool membersPerGroupInY = (numRepsX == numRepsY) ? mlsIsRowMajor
                                 : numRepsX < numRepsY;  // per numRepsY mls insts share same coordX ?
    bool needBoundaryPadding = boundaryCheckInfo[0] || boundaryCheckInfo[1];
    if (needBoundaryPadding) {
      assert(!(boundaryCheckInfo[0] && boundaryCheckInfo[1]) && "both padding condition should not here!");
      if (boundaryCheckInfo[1] && membersPerGroupInY)
        membersPerGroupInY = false;
    }
    mlOffsInY = membersPerGroupInY;
    bool groupPaddingInY = !mlOffsInY;

    bool isBit4 = isMlsBit4ElemTyKind(static_cast<MlsElemBitTyKind>(mlsInsn.getElemBitTyKind()));
    unsigned dim0DivFactor = !isBit4 ? 1 : mlsTileE[0] / mlsTileG[0];
    unsigned dim1DivFactor = !isBit4 ? 1 : mlsTileE[1] / mlsTileG[1];

    SmallVector<SmallVector<std::tuple<Value, unsigned, Value, Value>>> loadStoreOffsets;
    unsigned numGroups = membersPerGroupInY ? numRepsX : numRepsY;     /* need create numGroups rsrc desc */
    unsigned membersPerGroup = membersPerGroupInY ? numRepsY : numRepsX; /* per group has membersPerGroup mls insts */
    loadStoreOffsets.resize(numGroups);
    for (auto &innerVec : loadStoreOffsets) {
        innerVec.resize(membersPerGroup);
    }

    for (unsigned ctaX = 0; ctaX < numReps[0]; ++ctaX) {
      for (unsigned ctaY = 0; ctaY < numReps[1]; ++ctaY) {
        for (unsigned instIdxX = 0; instIdxX < mlInsnAttr.instrsPerWarp[0]; ++instIdxX) {
          for (unsigned instIdxY = 0; instIdxY < mlInsnAttr.instrsPerWarp[1]; ++instIdxY) {
            unsigned groupIdx = membersPerGroupInY ? ctaX * mlInsnAttr.instrsPerWarp[0] + instIdxX
                                                   : ctaY * mlInsnAttr.instrsPerWarp[1] + instIdxY;
            unsigned memberIdx = membersPerGroupInY ? ctaY * mlInsnAttr.instrsPerWarp[1] + instIdxY
                                                    : ctaX * mlInsnAttr.instrsPerWarp[0] + instIdxX;

            unsigned ctaRow = ctaX * shapePerCta[0];
            unsigned ctaCol = ctaY * shapePerCta[1];
            /* Note: This fix need sync to other branches */
            Value ldRow = membersPerGroupInY
                              ? b.add(offWarpX, b.i32_val(ctaRow + instIdxX * mlInsnAttr.instrShape[0] / dim0DivFactor))
                              : offWarpX;
            Value ldCol = membersPerGroupInY
                              ? offWarpY
                              : b.add(offWarpY, b.i32_val(ctaCol + instIdxY * mlInsnAttr.instrShape[1] / dim1DivFactor));

            Value ldOffV = dot64(loc, rewriter, {ldRow, ldCol}, llStrides);
            unsigned ldOffC = membersPerGroupInY ? ctaCol + instIdxY * mlInsnAttr.instrShape[1] / dim1DivFactor
                                                 : ctaRow + instIdxX * mlInsnAttr.instrShape[0] / dim0DivFactor;

            Value stMlsRow = b.add(b.i32_val(ctaX * warpsPerCta[0]), multiDimWarpIds[0]);
            Value stMlsCol = b.add(b.i32_val(ctaY * warpsPerCta[1]), multiDimWarpIds[1]);

            Value stMlsOff = b.mul(linearize(rewriter, loc, {stMlsRow, stMlsCol},
                                flattenedMlsShape, sharedLayout.getOrder()),
                                b.i32_val(product(mlsTileS)));
            unsigned instIdx = instIdxX * mlInsnAttr.instrsPerWarp[1] + instIdxY;
            Value stInMlsOff = b.i32_val(dsByteOffsets[instIdx]/elemByteWidth); // element offset in mls
            Value stOffset = b.add(stMlsOff, stInMlsOff);

            Value ldInBlockOffVal = groupPaddingInY ? ldCol : ldRow;
            loadStoreOffsets[groupIdx][memberIdx] = std::make_tuple(ldOffV, ldOffC, stOffset, ldInBlockOffVal);
          }
        }
      }
    }

    DenseMap<Value, SmallVector<std::pair<unsigned, Value>>> loadStoreOffsetsMap;
    for (auto &group : loadStoreOffsets) {
      Value ldOffV = std::get<0>(group[0]);
      auto &curMap = loadStoreOffsetsMap[ldOffV];
      curMap.reserve(group.size());
      for (auto &member : group) {
        unsigned ldOffC = std::get<1>(member);
        Value stOffset = std::get<2>(member);
        curMap.emplace_back(ldOffC, stOffset);
      }

      Value ldInBlockOffVal = std::get<3>(group[0]);
      ldInBlkOffCoordMappings[ldOffV] = ldInBlockOffVal;
    }

    return loadStoreOffsetsMap;
  }

  // reference: emitBaseIndexForMfmaLayout, emitMfmaOffsetForCTA
  SmallVector<SmallVector<Value>>
  computeMatrixLoadStoreOffsetsMappingInCTA(Location loc, RewriterBase &rewriter,
                                       const MlsInsn &mlsInsn,
                                       const MlsEncodingAttr &blockLayout,
                                       const MatrixLoadInsnAttr &mlInsnAttr,
                                       ArrayRef<int64_t> shape,
                                       SmallVector<Value> &multiDimWarpIds) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto rank = shape.size();
    assert(rank == 2);

    auto warpOrder = blockLayout.getOrder();
    auto _warpsPerCTA = blockLayout.getWarpsPerCTA();
    SmallVector<Value> warpsPerCTA;
    for (unsigned i = 0; i < rank; ++i)
      warpsPerCTA.push_back(b.i32_val(_warpsPerCTA[i]));

    auto mlsTileG = mlsInsn.getMlsTile(MlsTileKind::Global);
    auto mlsTileE = mlsInsn.getMlsTile(MlsTileKind::Elems);
    bool isBit4 = isMlsBit4ElemTyKind(static_cast<MlsElemBitTyKind>(mlsInsn.getElemBitTyKind()));
    unsigned dim0DivFactor = !isBit4 ? 1 : mlsTileE[0] / mlsTileG[0];
    unsigned dim1DivFactor = !isBit4 ? 1 : mlsTileE[1] / mlsTileG[1];

    Value warpId = getLinearWarpId(loc, rewriter, targetInfo);
    SmallVector<Value> multiDimWarpId =
        delinearize(rewriter, loc, warpId, _warpsPerCTA, warpOrder);

    if (shape[rank - 2] >= mlsTileG[rank - 2]) {
      assert(shape[rank - 2] % mlsTileG[rank - 2] == 0);
      multiDimWarpId[rank - 2] =
          b.urem(multiDimWarpId[rank - 2],
               b.i32_val(ceil<unsigned>(shape[rank - 2], mlsTileG[rank - 2])));
    }
    if (shape[rank - 1] >= mlsTileG[rank - 1]) {
      assert(shape[rank - 1] % mlsTileG[rank - 1] == 0);
      multiDimWarpId[rank - 1] =
          b.urem(multiDimWarpId[rank - 1],
               b.i32_val(ceil<unsigned>(shape[rank - 1], mlsTileG[rank - 1])));
    }

    multiDimWarpIds.push_back(multiDimWarpId[rank - 2]);
    multiDimWarpIds.push_back(multiDimWarpId[rank - 1]);

    Value offWarpX = b.mul(multiDimWarpId[rank - 2], b.i32_val(mlsTileG[rank - 2]));
    Value offWarpY = b.mul(multiDimWarpId[rank - 1], b.i32_val(mlsTileG[rank - 1]));

    SmallVector<SmallVector<Value>> multiDimOffsets;
    for (unsigned i = 0; i < mlInsnAttr.instrsPerWarp[0]; i++) {
      for (unsigned j = 0; j < mlInsnAttr.instrsPerWarp[1]; j++) {
        assert((mlInsnAttr.instrsPerWarp[0] == 1 || mlInsnAttr.instrsPerWarp[1] == 1) &&
               "add more code support!");
        Value offElemX =
            b.add(offWarpX, b.mul(b.i32_val(i), b.i32_val(mlInsnAttr.instrShape[0] / dim0DivFactor)));
        Value offElemY =
            b.add(offWarpY, b.mul(b.i32_val(j), b.i32_val(mlInsnAttr.instrShape[1] / dim1DivFactor)));

        multiDimOffsets.push_back({offElemX, offElemY});
      }
    }

    return multiDimOffsets;
  }

  Value createRsrcDesc(Location loc, RewriterBase &rewriter,
                       unsigned mlsVersion,
                       Value llBasePtr, Value llStride,
                       unsigned alt2Kind, Value mPadding, Value nmPadding)  const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    // 1. Create the resource descriptor
    // version1:
    // dword 0: base_addr_lo, bit pos: [31:0], Byte base address 32 LSBs for global adressing
    // dwrod 1: base_addr_hi, bit pos: [15:0], Byte base address 16 MSBs for global adressing
    // dword 2: stride,       bit pos: [31:0], The unit is matrix element.
    // dword 3:
    //   mfilter,      bit pos: [7:0]
    //                 The number of row/column needed to be zero-padded in matrix major direction.
    //   nfilter,      bit pos: [15:8]
    //                 The number of row/column needed to be zero-padded ...
    //   cache_swizzle_enable, bit pos: [16]
    //                 1: enable cache swizzle operation in L1 cache.
    //   mfmt,         bit pos: [18:17], The matrix layout format in memory.
    //                 0: non-interleaved
    //                 1: 2-interleaved
    //                 2: 4-interleaved
    //                 3: 8-interleaved
    // version2:
    //   1. mfmt no need to set
    // version3:
    //   2. mfmt no need to set
    //   3. mfilter_hi, nfilter_hi new add:
    //   mfilter_hi,   bit pos: [20:20]
    //   nfilter_hi,   bit pos: [22:22]
    // Note: from Spec, it requires the base ptr to be 16-byte aligned !!!

    // dword 0: base_addr_lo [31:0]
    Value baseLow = b.ptrtoint(i32_ty, llBasePtr);
    // dword 1: base_addr_hi [15:0]
    Value baseHigh = b.and_(b.trunc(i32_ty, b.lshr(b.ptrtoint(i64_ty, llBasePtr),
                                          b.i64_val(32))),
                          b.i32_val(0xFFFF));

    // dword 2: stride [31:0]
    Value stride = llStride;

    // dword 3:
    Value mfilter = b.and_(mPadding, b.i32_val(0xFF));
    Value nfilter = b.and_(nmPadding, b.i32_val(0xFF));
    bool enableCacheSwizzle =
        triton::tools::isEnvValueBool(
            triton::tools::getStrEnv("TRITON_BUFFER_CACHE_SWIZZLE"))
            .value_or(true);
    Value cacheSwizzle = b.i32_val(enableCacheSwizzle ? 1 : 0);
    Value mfmt = b.i32_val(alt2Kind);

    Value controlField = mfilter;                                                    // [7:0]
    controlField = b.or_(controlField, b.shl(nfilter, b.i32_val(8)));       // [15:8]
    controlField = b.or_(controlField, b.shl(cacheSwizzle, b.i32_val(16))); // [16]

    if (mlsVersion == 1)
      controlField = b.or_(controlField, b.shl(mfmt, b.i32_val(17)));         // [18:17]
    else if (mlsVersion == 3) {
      Value mfilter_hi = b.and_(b.lshr(mPadding, b.i32_val(8)), b.i32_val(0x1));
      Value nfilter_hi = b.and_(b.lshr(nmPadding, b.i32_val(8)), b.i32_val(0x1));
      controlField = b.or_(controlField, b.shl(mfilter_hi, b.i32_val(20))); // [20:20]
      controlField = b.or_(controlField, b.shl(nfilter_hi, b.i32_val(22))); // [22:22]
    }

    Value resource = b.undef(vec_ty(i32_ty, 4));
    resource = b.insert_element(vec_ty(i32_ty, 4), resource, baseLow, b.i32_val(0));
    resource = b.insert_element(vec_ty(i32_ty, 4), resource, baseHigh, b.i32_val(1));
    resource = b.insert_element(vec_ty(i32_ty, 4), resource, stride, b.i32_val(2));
    resource = b.insert_element(vec_ty(i32_ty, 4), resource, controlField, b.i32_val(3));

    return resource;
  }

  void generateMatrixLoadLDSOp(Location loc, RewriterBase &rewriter,
                               StringRef mlInsnOpName,
                               Value rsrcDesc,
                               Value soffset,
                               bool t,            // Transposition. 0: column major, 1: row major
                               int32_t offset,    // Global address offset [9:0], elem number offset
                               bool r,            // Golbal address offset is in row direction ?
                               bool glc = false,
                               bool slc = false,
                               bool bps = false) const {
    OperationState loweredOp(loc, mlInsnOpName);
    loweredOp.addOperands(rsrcDesc);
    loweredOp.addOperands(soffset);

    loweredOp.addAttribute("offset", rewriter.getI32IntegerAttr(offset));
    loweredOp.addAttribute("t", rewriter.getBoolAttr(t));
    loweredOp.addAttribute("r", rewriter.getBoolAttr(r));
    loweredOp.addAttribute("glc", rewriter.getBoolAttr(glc));
    loweredOp.addAttribute("slc", rewriter.getBoolAttr(slc));
    loweredOp.addAttribute("bps", rewriter.getBoolAttr(bps));

    rewriter.create(loweredOp);
  }


  Value dot64(Location loc, RewriterBase &rewriter,
            ValueRange offsets32, ValueRange strides64) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    assert(offsets32.size() == strides64.size());
    Value ret = b.i64_val(0);
    for (auto [offset, stride] : llvm::zip(offsets32, strides64)) {
      ret = b.add(ret, b.mul(b.zext(i64_ty, offset), stride));
    }
    return ret;
  }
};

//--------------------------------------------------------------------------------------------------


struct MLSMatrixStoreFromRegOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::MatrixStoreFromRegOp> {
  MLSMatrixStoreFromRegOpConversion(LLVMTypeConverter &converter,
                                    const AMD::TargetInfo &targetInfo,
                                    ModuleAxisInfoAnalysis &axisAnalysisPass,
                                    PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::amdgpu::MatrixStoreFromRegOp>(
            converter, benefit),
        targetInfo(targetInfo),
        loadHelper(converter, targetInfo, axisAnalysisPass, benefit) {}

  LogicalResult
  matchAndRewrite(triton::amdgpu::MatrixStoreFromRegOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto valueTy = cast<RankedTensorType>(op.getValue().getType());
    auto storeEncoding = op->getAttrOfType<MlsStoreEncodingAttr>(
        MlsStoreEncodingAttr::getMnemonic());
    if (!storeEncoding)
      return failure();
    return lowerMatrixStoreFromReg(
        op.getLoc(), rewriter, valueTy, adaptor.getValue(), adaptor.getBase(),
        adaptor.getShape(), adaptor.getStrides(), adaptor.getIndices(),
        adaptor.getBoundaryCheck(), storeEncoding, op);
  }

  /// VGPR→GMEM matrix_store for fp16 MMAC C layout 3.
  LogicalResult lowerMatrixStoreFromReg(
      Location loc, ConversionPatternRewriter &rewriter,
      RankedTensorType valueTy, Value llValue, Value llBasePtr,
      ValueRange llShape, ValueRange llStrides, ValueRange llOffsets,
      ArrayRef<int32_t> boundaryCheck, MlsStoreEncodingAttr storeEncoding,
      Operation *opToErase) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto ctx = rewriter.getContext();
    auto *typeConverter = getTypeConverter();

    auto mfmaEnc = dyn_cast<AMDMfmaEncodingAttr>(valueTy.getEncoding());
    if (!valueTy.getElementType().isF16() || !mfmaEnc ||
        mfmaEnc.getElementBitWidth() != 32 ||
        mfmaEnc.getMmacLayout() != MmacLayout::TRANSPOSE ||
        storeEncoding.getInterleaveKind() !=
            static_cast<unsigned>(MlsInterleaveKind::Interleave2)) {
      return rewriter.notifyMatchFailure(
          opToErase,
          "matrix_store_from_reg requires fp16 MMAC C layout 3 with interleave 2");
    }

    auto storeTile = storeEncoding.getStoreTile();
    bool transpose = storeEncoding.getOrder()[0] == 1;
    auto mlsInsn = MlsInsn::selectOrGetMatrixStoreInsn(
        storeTile[0], storeTile[1], storeEncoding.getElemBitWidth(),
        storeEncoding.getVersion(), static_cast<MlsInterleaveKind>(
                                        storeEncoding.getInterleaveKind()),
        transpose);
    if (failed(mlsInsn))
      return failure();
    auto msInsnAttr = mlsInsn->getMatrixStoreInsnAttr();
    if (msInsnAttr.insn.empty())
      return rewriter.notifyMatchFailure(
          opToErase, "no matrix-store instruction for the selected MLS layout");
    MatrixLoadInsnAttr traversalAttr{};
    traversalAttr.instrShape = msInsnAttr.instrShape;
    traversalAttr.instrsPerWarp = msInsnAttr.instrsPerWarp;
    traversalAttr.instrOrder = msInsnAttr.instrOrder;
    constexpr unsigned interleaveFactor = 2;

    // The existing address helpers are shared with matrix-load and consume
    // MlsEncodingAttr. Keep this compatibility representation local to LLVM
    // lowering; the store IR itself uses M/N-only MlsStoreEncodingAttr.
    auto blockLayout = MlsEncodingAttr::get(
        ctx, /*opIdx=*/0, mlsInsn->getMlsTile(),
        storeEncoding.getElemBitWidth(),
        static_cast<unsigned>(MlsElemBitTyKind::None),
        storeEncoding.getInterleaveKind(), storeEncoding.getVersion(),
        storeEncoding.getOrder(), storeEncoding.getWarpsPerCTA());

    // Synthetic shared layout so GMEM address helpers match the LDS store path.
    auto ctaLayout = getCTALayout(valueTy.getEncoding());
    auto sharedLayout = HCUMlsSharedEncodingAttr::get(
        ctx, blockLayout.getOpIdx(), blockLayout.getMlsTile(),
        blockLayout.getElemBitWidth(),
        static_cast<MlsElemBitTyKind>(blockLayout.getElemBitTyKind()),
        blockLayout.getAlt2Kind(), blockLayout.getVersion(),
        blockLayout.getOrder(), ctaLayout);

    auto inVals = unpackLLElements(loc, llValue, rewriter);
    auto elemByteWidth = blockLayout.getElemBitWidth() / 8;
    auto llBlockPos = llOffsets;
    auto ldBlkOff = loadHelper.dot64(loc, rewriter, llOffsets, llStrides);

    // MFMA-C: each warp only owns shape[d]/warpsPerCTA[d] of the CTA tile.
    // Offset helpers that iterate the full tensor shape would visit other warps'
    // regions while consuming this warp's registers → scrambled stores.
    // Restrict address iteration to the per-warp tile and add warp base.
    //
    // Do NOT use tilesPerWarp*instr alone: when MLS A/B tiles are smaller than
    // the C block (e.g. BM=256 with mlsTile=64), MFMA layouts absorb the
    // leftover as extra register dims (shape/(warps*tiles*instr)). Per-warp
    // store size must follow the tensor shape so VGPR count matches.
    MlsEncodingAttr storeLayout = blockLayout;
    SmallVector<int64_t> storeShape(valueTy.getShape().begin(),
                                    valueTy.getShape().end());
    auto mfmaWarps = mfmaEnc.getWarpsPerCTA();
    assert(mfmaWarps.size() >= 2);
    assert(storeShape[0] % mfmaWarps[0] == 0 &&
           storeShape[1] % mfmaWarps[1] == 0);
    storeShape[0] = storeShape[0] / mfmaWarps[0];
    storeShape[1] = storeShape[1] / mfmaWarps[1];
    SmallVector<unsigned> oneWarp = {1, 1};
    storeLayout = MlsEncodingAttr::get(
        ctx, blockLayout.getOpIdx(), blockLayout.getMlsTile(),
        blockLayout.getElemBitWidth(), blockLayout.getElemBitTyKind(),
        blockLayout.getAlt2Kind(), blockLayout.getVersion(),
        blockLayout.getOrder(), oneWarp);

    // MFMA warps are delinearized in MMA order (M,N), not MLS tensor order.
    SmallVector<unsigned> warpOrder =
        triton::gpu::getMatrixOrder(/*rank=*/2, /*rowMajor=*/true);
    Value linearWarpId = getLinearWarpId(loc, rewriter, targetInfo);
    SmallVector<Value> multiDimWarpId =
        delinearize(rewriter, loc, linearWarpId, mfmaWarps, warpOrder);
    Value warpRow =
        b.mul(multiDimWarpId[0], b.i32_val(static_cast<int32_t>(storeShape[0])));
    // Interleave2 keeps each pair of instruction-width N stores together and
    // distributes consecutive pairs across N warps.  Express that ownership
    // directly instead of first building a contiguous per-warp coordinate and
    // exchanging address bits afterwards.  This also handles the degenerate
    // 32x32 case where groupWidth == storeShape[1] (there is no local group to
    // exchange with the warp coordinate).
    unsigned storeNGroupWidth =
        interleaveFactor * msInsnAttr.instrShape[1];
    unsigned numStoreNWarps = mfmaWarps[1];
    Value warpCol = b.mul(
        multiDimWarpId[1], b.i32_val(static_cast<int32_t>(storeNGroupWidth)));
    Value warpBaseOff = loadHelper.dot64(loc, rewriter, {warpRow, warpCol}, llStrides);
    ldBlkOff = b.add(ldBlkOff, warpBaseOff);
    llBlockPos = {b.add(llBlockPos[0], warpRow),
                  b.add(llBlockPos[1], warpCol)};

    bool needBoundaryCheck = boundaryCheck.size() > 0;
    SmallVector<bool> boundaryCheckInfo = {
        needBoundaryCheck && boundaryCheck[0] == 0,
        needBoundaryCheck &&
            boundaryCheck[boundaryCheck.size() - 1] == 1};
    unsigned iWarpSize = targetInfo.getWarpSize();
    assert(iWarpSize == 64);
    unsigned numOfElemsPerInsn =
        msInsnAttr.instrShape[0] * msInsnAttr.instrShape[1] / iWarpSize;

    unsigned valIdx = 0;
    unsigned wideStoreIdx = 0;
    auto packNext = [&]() -> FailureOr<Value> {
      if (valIdx + numOfElemsPerInsn > inVals.size())
        return failure();
      SmallVector<Value> chunk;
      chunk.reserve(numOfElemsPerInsn);
      if (msInsnAttr.instrShape == std::array<unsigned, 2>{32, 32}) {
        // Layout-3's flat order is [M-half][N-half].  Each wide store needs
        // both MMAC M subgroups for one (storeM, storeN) tile.  The subgroup
        // stride is one complete row of N stores, not half of all input
        // values (the latter only happened to work when storeM == 1).
        unsigned half = numOfElemsPerInsn / 2;
        unsigned numStoresM =
            storeShape[0] / msInsnAttr.instrShape[0];
        unsigned numStoresN =
            storeShape[1] / msInsnAttr.instrShape[1];
        unsigned numWideStores = numStoresM * numStoresN;
        if (half * 2 != numOfElemsPerInsn ||
            wideStoreIdx >= numWideStores ||
            inVals.size() != numWideStores * numOfElemsPerInsn)
          return failure();
        unsigned storeM = wideStoreIdx / numStoresN;
        unsigned storeN = wideStoreIdx % numStoresN;
        unsigned subgroupStride = numStoresN * half;
        unsigned storeMStride = 2 * subgroupStride;
        unsigned base = storeM * storeMStride + storeN * half;
        for (unsigned mSubgroup = 0; mSubgroup < 2; ++mSubgroup)
          for (unsigned i = 0; i < half; ++i)
            chunk.push_back(
                inVals[base + mSubgroup * subgroupStride + i]);
        ++wideStoreIdx;
        valIdx += numOfElemsPerInsn;
      } else {
        for (unsigned i = 0; i < numOfElemsPerInsn; ++i)
          chunk.push_back(inVals[valIdx++]);
      }
      unsigned elemsPerVreg128 = 128 / storeEncoding.getElemBitWidth();
      if (chunk.size() % interleaveFactor != 0)
        return failure();
      if (chunk.size() % elemsPerVreg128 != 0)
        return failure();
      SmallVector<Value> interleaved;
      interleaved.reserve(chunk.size());
      for (unsigned group = 0; group < chunk.size();
           group += elemsPerVreg128) {
        unsigned elemsPerInterleave = elemsPerVreg128 / interleaveFactor;
        for (unsigned elem = 0; elem < elemsPerInterleave; ++elem)
          for (unsigned part = 0; part < interleaveFactor; ++part)
            interleaved.push_back(
                chunk[group + part * elemsPerInterleave + elem]);
      }
      chunk = std::move(interleaved);
      return packMatrixStoreVGPRArgs(loc, rewriter, valueTy, msInsnAttr,
                                     chunk);
    };

    // Build every address/descriptor and pack every accumulator before emitting
    // matrix stores.  Keeping address arithmetic out of the store sequence gives
    // the backend one contiguous group that it can issue with maximum ILP.
    struct PendingMatrixStore {
      Value packed;
      Value rsrcDesc;
      bool t;
      int32_t offset;
      bool r;
    };
    SmallVector<PendingMatrixStore> pendingStores;
    auto queueOne = [&](Value rsrcDesc, bool t, int32_t offset,
                        bool r) -> LogicalResult {
      auto packed = packNext();
      if (failed(packed))
        return failure();
      pendingStores.push_back({*packed, rsrcDesc, t, offset, r});
      return success();
    };

    bool useMatrixLoadStoreOffsetsWithMlOffs = true;
    if (boundaryCheckInfo[0] && boundaryCheckInfo[1])
      useMatrixLoadStoreOffsetsWithMlOffs = false;

    // Rebuild shared layout against the (possibly warp-local) store layout.
    sharedLayout = HCUMlsSharedEncodingAttr::get(
        ctx, storeLayout.getOpIdx(), storeLayout.getMlsTile(),
        storeLayout.getElemBitWidth(),
        static_cast<MlsElemBitTyKind>(storeLayout.getElemBitTyKind()),
        storeLayout.getAlt2Kind(), storeLayout.getVersion(),
        storeLayout.getOrder(), ctaLayout);

    if (!useMatrixLoadStoreOffsetsWithMlOffs) {
      SmallVector<std::pair<Value, Value>> ldInBlkOffCoordMappings;
      auto ldstInBlkMappings = loadHelper.computeMatrixLoadStoreOffsets(
          loc, rewriter, *mlsInsn, storeLayout, traversalAttr, sharedLayout,
          storeShape, llStrides, mlsInsn->isRowMajor(),
          boundaryCheckInfo, ldInBlkOffCoordMappings);

      unsigned iterIdx = 0;
      for (auto &ldstInBlkGroup : ldstInBlkMappings) {
        Value ldOffset = b.add(ldstInBlkGroup.first, ldBlkOff);
        Value logicalCol = ldInBlkOffCoordMappings[iterIdx].second;
        Value group = b.udiv(logicalCol, b.i32_val(storeNGroupWidth));
        Value inner = b.urem(logicalCol, b.i32_val(storeNGroupWidth));
        Value physicalCol = b.add(
            b.mul(group,
                  b.i32_val(storeNGroupWidth * numStoreNWarps)),
            inner);
        Value colDelta = b.sub(physicalCol, logicalCol);
        ldOffset = b.add(
            ldOffset, b.mul(b.sext(i64_ty, colDelta), llStrides[1]));
        Value ldPtr = b.gep(ptr_ty(ctx, 1), elemByteWidth == 1 ? i8_ty : i16_ty,
                            llBasePtr, ldOffset);
        Value llStride = b.trunc(i32_ty, llStrides[sharedLayout.getOrder()[1]]);

        Value paddingLenX = b.i32_val(0);
        Value paddingLenY = b.i32_val(0);
        if (boundaryCheckInfo[0]) {
          Value ldInBlockOffCoordX = ldInBlkOffCoordMappings[iterIdx].first;
          Value posEnd =
              b.add(b.add(llBlockPos[0], ldInBlockOffCoordX),
                    b.i32_val(msInsnAttr.instrShape[0]));
          Value posGt = b.icmp_sgt(posEnd, llShape[0]);
          paddingLenX =
              b.select(posGt, b.sub(posEnd, llShape[0]), b.i32_val(0));
        }
        if (boundaryCheckInfo[1]) {
          Value ldInBlockOffCoordY = physicalCol;
          Value posEnd =
              b.add(b.add(llBlockPos[1], ldInBlockOffCoordY),
                    b.i32_val(msInsnAttr.instrShape[1]));
          Value posGt = b.icmp_sgt(posEnd, llShape[1]);
          paddingLenY =
              b.select(posGt, b.sub(posEnd, llShape[1]), b.i32_val(0));
        }
        Value mPadding =
            mlsInsn->getMajorDimIndex() == 0 ? paddingLenX : paddingLenY;
        Value nmPadding =
            mlsInsn->getMajorDimIndex() == 0 ? paddingLenY : paddingLenX;
        Value rsrcDesc = loadHelper.createRsrcDesc(
            loc, rewriter, mlsInsn->getMlsVersion(), ldPtr, llStride,
            storeLayout.getAlt2Kind(), mPadding, nmPadding);
        if (failed(queueOne(rsrcDesc, mlsInsn->isRowMajor(), 0, false)))
          return failure();
        iterIdx++;
      }
    } else {
      bool mlOffsInY;
      DenseMap<Value, Value> ldInBlkOffCoordMappings;
      auto ldstInBlkMappings = computeMatrixStoreOffsetsWithMlOffs(
          loc, rewriter, *mlsInsn, storeLayout, msInsnAttr, storeShape,
          llStrides, mlsInsn->isRowMajor(),
          boundaryCheckInfo, storeNGroupWidth, numStoreNWarps, mlOffsInY,
          ldInBlkOffCoordMappings);
      bool groupPaddingInY = !mlOffsInY;

      for (auto &ldstInBlkGroup : ldstInBlkMappings) {
        Value ldOffset = b.add(ldstInBlkGroup.first, ldBlkOff);
        Value ldPtr = b.gep(ptr_ty(ctx, 1), elemByteWidth == 1 ? i8_ty : i16_ty,
                            llBasePtr, ldOffset);
        Value llStride = b.trunc(i32_ty, llStrides[sharedLayout.getOrder()[1]]);
        Value rsrcDesc;
        for (unsigned ldInBlkOffC : ldstInBlkGroup.second) {
          if (!rsrcDesc) {
            Value paddingLen = b.i32_val(0);
            if (needBoundaryCheck) {
              int dimIdx = groupPaddingInY ? 1 : 0;
              Value ldInBlockOffCoordV =
                  ldInBlkOffCoordMappings[ldstInBlkGroup.first];
              Value posEnd = b.add(
                  b.add(llBlockPos[dimIdx], ldInBlockOffCoordV),
                  b.i32_val(msInsnAttr.instrShape[dimIdx]));
              Value posGt = b.icmp_sgt(posEnd, llShape[dimIdx]);
              paddingLen = b.select(posGt, b.sub(posEnd, llShape[dimIdx]),
                                    b.i32_val(0));
            }
            Value mPadding =
                ((groupPaddingInY && mlsInsn->isRowMajor()) ||
                 (!groupPaddingInY && !mlsInsn->isRowMajor()))
                    ? paddingLen
                    : b.i32_val(0);
            Value nmPadding =
                mPadding == paddingLen ? b.i32_val(0) : paddingLen;
            rsrcDesc = loadHelper.createRsrcDesc(
                loc, rewriter, mlsInsn->getMlsVersion(), ldPtr, llStride,
                storeLayout.getAlt2Kind(), mPadding, nmPadding);
          }
          bool t = mlsInsn->isRowMajor();
          bool r = mlOffsInY;
          assert(ldInBlkOffC < 1024 && "ldInBlkOffC out of range");
          if (failed(queueOne(rsrcDesc, t, ldInBlkOffC, r)))
            return failure();
        }
      }
    }

    if (valIdx != inVals.size())
      return rewriter.notifyMatchFailure(
          opToErase, "matrix_store_from_reg value/instruction count mismatch");

    for (const auto &store : pendingStores)
      generateMatrixStoreVGPROp(loc, rewriter, msInsnAttr.insn, store.packed,
                                store.rsrcDesc, store.t, store.offset, store.r,
                                numOfElemsPerInsn *
                                    storeEncoding.getElemBitWidth() / 32,
                                false, false);

    rewriter.eraseOp(opToErase);
    return success();
  }


private:
  /*
   * Calculate matrix-store offsets while preserving accumulator order.
   * For each vector, share same rsrc desc with ldOffValue, but has different ldOffConst offset & stOffValue.
   * key: ldOffValue, value: std::pair<ldOffConst, stOffValue>
   **/
  llvm::MapVector<Value, SmallVector<unsigned>>
  computeMatrixStoreOffsetsWithMlOffs(Location loc, RewriterBase &rewriter,
                                const MlsInsn &mlsInsn,
                                const MlsEncodingAttr &blockLayout,
                                const MatrixStoreInsnAttr &msInsnAttr,
                                ArrayRef<int64_t> tensorShape,
                                ValueRange llStrides,
                                bool mlsIsRowMajor,
                                SmallVector<bool> boundaryCheckInfo,
                                unsigned storeNGroupWidth,
                                unsigned numStoreNWarps,
                                bool &mlOffsInY,     /* key: ldOffValue, value: ldCoordValue */
                                DenseMap<Value, Value> &ldInBlkOffCoordMappings) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    auto mlsTileG = mlsInsn.getMlsTile(MlsTileKind::Global);
    auto mlsTileE = mlsInsn.getMlsTile(MlsTileKind::Elems);
    auto tensorShapeG = tensorShape;

    auto warpsPerCta = blockLayout.getWarpsPerCTA();
    auto shapePerCta = getShapePerCTA(mlsTileG, warpsPerCta);
    SmallVector<Value> multiDimWarpIds;
    MatrixLoadInsnAttr mappingAttr{};
    mappingAttr.instrShape = msInsnAttr.instrShape;
    mappingAttr.instrsPerWarp = msInsnAttr.instrsPerWarp;
    mappingAttr.instrOrder = msInsnAttr.instrOrder;
    auto dummyMapping = loadHelper.computeMatrixLoadStoreOffsetsMappingInCTA(
        loc, rewriter, mlsInsn, blockLayout, mappingAttr, tensorShape,
        multiDimWarpIds);

    auto rank = tensorShape.size();
    Value offWarpX = b.mul(multiDimWarpIds[rank - 2], b.i32_val(mlsTileG[rank - 2]));
    Value offWarpY = b.mul(multiDimWarpIds[rank - 1], b.i32_val(mlsTileG[rank - 1]));

    SmallVector<unsigned> numReps = getNumReps(mlsInsn, warpsPerCta, tensorShapeG);
    unsigned numRepsX = numReps[0] * msInsnAttr.instrsPerWarp[0];
    unsigned numRepsY = numReps[1] * msInsnAttr.instrsPerWarp[1];
    bool membersPerGroupInY = (numRepsX == numRepsY) ? mlsIsRowMajor
                                 : numRepsX < numRepsY;  // per numRepsY mls insts share same coordX ?
    bool needBoundaryPadding = boundaryCheckInfo[0] || boundaryCheckInfo[1];
    if (needBoundaryPadding) {
      assert(!(boundaryCheckInfo[0] && boundaryCheckInfo[1]) && "both padding condition should not here!");
      if (boundaryCheckInfo[1] && membersPerGroupInY)
        membersPerGroupInY = false;
    }
    mlOffsInY = membersPerGroupInY;
    bool groupPaddingInY = !mlOffsInY;

    bool isBit4 = isMlsBit4ElemTyKind(static_cast<MlsElemBitTyKind>(mlsInsn.getElemBitTyKind()));
    unsigned dim0DivFactor = !isBit4 ? 1 : mlsTileE[0] / mlsTileG[0];
    unsigned dim1DivFactor = !isBit4 ? 1 : mlsTileE[1] / mlsTileG[1];

    SmallVector<SmallVector<std::tuple<Value, unsigned, Value>>> storeOffsets;
    auto mapStoreNOffset = [&](unsigned logicalCol) {
      unsigned group = logicalCol / storeNGroupWidth;
      unsigned inner = logicalCol % storeNGroupWidth;
      return group * numStoreNWarps * storeNGroupWidth + inner;
    };
    unsigned numGroups = membersPerGroupInY ? numRepsX : numRepsY;     /* need create numGroups rsrc desc */
    unsigned membersPerGroup = membersPerGroupInY ? numRepsY : numRepsX; /* per group has membersPerGroup mls insts */
    storeOffsets.resize(numGroups);
    for (auto &innerVec : storeOffsets) {
        innerVec.resize(membersPerGroup);
    }

    for (unsigned ctaX = 0; ctaX < numReps[0]; ++ctaX) {
      for (unsigned ctaY = 0; ctaY < numReps[1]; ++ctaY) {
        for (unsigned instIdxX = 0; instIdxX < msInsnAttr.instrsPerWarp[0]; ++instIdxX) {
          for (unsigned instIdxY = 0; instIdxY < msInsnAttr.instrsPerWarp[1]; ++instIdxY) {
            unsigned groupIdx = membersPerGroupInY ? ctaX * msInsnAttr.instrsPerWarp[0] + instIdxX
                                                   : ctaY * msInsnAttr.instrsPerWarp[1] + instIdxY;
            unsigned memberIdx = membersPerGroupInY ? ctaY * msInsnAttr.instrsPerWarp[1] + instIdxY
                                                    : ctaX * msInsnAttr.instrsPerWarp[0] + instIdxX;

            unsigned ctaRow = ctaX * shapePerCta[0];
            unsigned ctaCol = ctaY * shapePerCta[1];
            unsigned logicalCol =
                ctaCol + instIdxY * msInsnAttr.instrShape[1] / dim1DivFactor;
            unsigned physicalCol = mapStoreNOffset(logicalCol);
            /* Note: This fix need sync to other branches */
            Value ldRow = membersPerGroupInY
                              ? b.add(offWarpX, b.i32_val(ctaRow + instIdxX * msInsnAttr.instrShape[0] / dim0DivFactor))
                              : offWarpX;
            Value ldCol = membersPerGroupInY
                              ? offWarpY
                              : b.add(offWarpY, b.i32_val(physicalCol));

            Value ldOffV = loadHelper.dot64(loc, rewriter, {ldRow, ldCol}, llStrides);
            unsigned ldOffC = membersPerGroupInY ? physicalCol
                                                 : ctaRow + instIdxX * msInsnAttr.instrShape[0] / dim0DivFactor;

            Value ldInBlockOffVal = groupPaddingInY ? ldCol : ldRow;
            storeOffsets[groupIdx][memberIdx] =
                std::make_tuple(ldOffV, ldOffC, ldInBlockOffVal);
          }
        }
      }
    }

    llvm::MapVector<Value, SmallVector<unsigned>> storeOffsetsMap;
    for (auto &group : storeOffsets) {
      Value ldOffV = std::get<0>(group[0]);
      auto &curMap = storeOffsetsMap[ldOffV];
      curMap.reserve(group.size());
      for (auto &member : group) {
        unsigned ldOffC = std::get<1>(member);
        curMap.push_back(ldOffC);
      }

      Value ldInBlockOffVal = std::get<2>(group[0]);
      ldInBlkOffCoordMappings[ldOffV] = ldInBlockOffVal;
    }

    return storeOffsetsMap;

  }


  /// Emit matrix_store_* VGPR→GMEM.
  /// Builtin/intrinsic: __builtin_hcu_matrix_store_* /
  /// llvm.hcu.matrix.store.* (vdata:i32xN, rsrc:i32x4, moffset, t, r, glc, slc)
  void generateMatrixStoreVGPROp(Location loc, RewriterBase &rewriter,
                                 StringRef matrixStoreInsnName, Value vdata,
                                 Value rsrcDesc, bool t, int32_t offset, bool r,
                                 unsigned vdataDwords,
                                 bool glc = false, bool slc = false) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Type vdataTy = vec_ty(i32_ty, vdataDwords);
    Value vdataI32 = vdata;
    if (vdata.getType() != vdataTy)
      vdataI32 = b.bitcast(vdata, vdataTy);

    OperationState loweredOp(loc, matrixStoreInsnName);
    loweredOp.addOperands(vdataI32);
    loweredOp.addOperands(rsrcDesc);
    loweredOp.addAttribute("offset", rewriter.getI32IntegerAttr(offset));
    loweredOp.addAttribute("t", rewriter.getBoolAttr(t));
    loweredOp.addAttribute("r", rewriter.getBoolAttr(r));
    loweredOp.addAttribute("glc", rewriter.getBoolAttr(glc));
    loweredOp.addAttribute("slc", rewriter.getBoolAttr(slc));
    rewriter.create(loweredOp);
  }

  Value packMatrixStoreVGPRArgs(Location loc, RewriterBase &rewriter,
                                RankedTensorType tensorTy,
                                const MatrixStoreInsnAttr &msInsnAttr,
                                ArrayRef<Value> elems) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto *typeConverter = getTypeConverter();
    auto msInsnElemTy = typeConverter->convertType(tensorTy.getElementType());
    unsigned iWarpSize = 64;
    auto msInsnNumOfElems =
        msInsnAttr.instrShape[0] * msInsnAttr.instrShape[1] / iWarpSize;
    Type vecTy = vec_ty(msInsnElemTy, msInsnNumOfElems);
    Value packed = b.undef(vecTy);
    for (unsigned elemId = 0; elemId < msInsnNumOfElems; ++elemId) {
      packed =
          b.insert_element(packed, elems[elemId], b.i32_val(elemId));
    }
    return packed;
  }

  const AMD::TargetInfo &targetInfo;
  MLSMatrixLoadToLocalOpConversion loadHelper;
};

//--------------------------------------------------------------------------------------------------

struct MLSLocalAllocOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp> {
  MLSLocalAllocOpConversion(LLVMTypeConverter &converter,
                            const AMD::TargetInfo &targetInfo,
                            ModuleAxisInfoAnalysis &axisAnalysisPass,
                            PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.isSharedMemoryAlloc())
      return failure();
    Location loc = op->getLoc();
    Value smemBase =
        LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
    auto resultTy = cast<MemDescType>(op.getType());
    auto typeConverter = getTypeConverter();
    auto sharedLayout =
        dyn_cast<HCUMlsSharedEncodingAttr>(resultTy.getEncoding());
    if (!sharedLayout)
      return failure();

    auto llvmElemTy = typeConverter->convertType(resultTy.getElementType());
    auto smemObj = LLVM::SharedMemoryObject(smemBase,
                            llvmElemTy, resultTy.getRank(), loc, rewriter);
    assert(!op.getSrc());
    auto retVal = LLVM::getStructFromSharedMemoryObject(loc, smemObj, rewriter);
    rewriter.replaceOp(op, retVal);

    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

//--------------------------------------------------------------------------------------------------

// Get warpId inside block of warps.
Value getWarpIdInBlock(ConversionPatternRewriter &rewriter, Location loc,
                       Value warpId, const ArrayRef<unsigned int> &wpt,
                       int elemPerInstrNonK, int tensorSizeNonK, int nonKIdx,
                       const ArrayRef<unsigned int> &order) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  SmallVector<Value> multiDimWarpId =
      delinearize(rewriter, loc, warpId, wpt, order);

  return b.urem(multiDimWarpId[nonKIdx],
              b.i32_val(tensorSizeNonK / elemPerInstrNonK));
}

struct MLSLocalLoadOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp> {
public:
  using ConvertOpToLLVMPattern<
      triton::gpu::LocalLoadOp>::ConvertOpToLLVMPattern;

  MLSLocalLoadOpConversion(LLVMTypeConverter &converter,
                           const AMD::TargetInfo &targetInfo,
                           ModuleAxisInfoAnalysis &axisAnalysisPass,
                           PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemDescType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();
    if (isa<triton::gpu::HCUMlsSharedEncodingAttr>(srcLayout)) {
      if (isa<DotOperandEncodingAttr>(dstLayout) &&
          isa<AMDMfmaEncodingAttr>(
              cast<DotOperandEncodingAttr>(dstLayout).getParent())) {
        return lowerMLSSharedToDotOperand(op, adaptor, getTypeConverter(),
                                        rewriter);
      } else {
        assert(false && "should been canonicalized and not reach here!");
        return failure();
      }
    }

    return failure();
  }

private:
  const AMD::TargetInfo &targetInfo;

  // shared -> matrix_core_dot_operand
  // reference: SharedToDotOperandMFMA::convertLayout in SharedToDotOperandMFMA.cpp
  LogicalResult
  lowerMLSSharedToDotOperand(triton::gpu::LocalLoadOp op,
                             triton::gpu::LocalLoadOpAdaptor adaptor,
                             const LLVMTypeConverter *typeConverter,
                             ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    Value src = op.getSrc();
    Value dst = op.getResult();
    auto dstTensorTy = cast<RankedTensorType>(dst.getType());
    auto srcTensorTy = cast<MemDescType>(src.getType());
    auto dotOperandLayout =
        cast<DotOperandEncodingAttr>(dstTensorTy.getEncoding());

    auto llvmElemTy = typeConverter->convertType(
        cast<MemDescType>(src.getType()).getElementType());

    auto smemObj = getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                   llvmElemTy, rewriter);

    Value res;
    auto opIdx = dotOperandLayout.getOpIdx();
    auto mfmaLayout = cast<AMDMfmaEncodingAttr>(dotOperandLayout.getParent());
    Value linearWarpId = getLinearWarpId(loc, rewriter, targetInfo);
    bool matrixStorePath = false;
    for (Operation *parent = op; parent; parent = parent->getParentOp()) {
      if (parent->hasAttr("triton.hcu.has_matrix_store")) {
        matrixStorePath = true;
        break;
      }
    }

    if ((opIdx == 0 && mfmaLayout.getMfmaTile()[0] == mfmaLayout.getInstrShape()[0]) ||
        (opIdx == 1 && mfmaLayout.getMfmaTile()[1] == mfmaLayout.getInstrShape()[1])) {
      res = convertLayoutUnitTilesPerWarp(dotOperandLayout.getOpIdx(), rewriter, loc, src,
                                          dotOperandLayout, smemObj, typeConverter,
                                          linearWarpId, matrixStorePath);
    } else {
        res = convertLayoutMultiTilesPerWarp(dotOperandLayout.getOpIdx(), rewriter, loc, src,
                                             dotOperandLayout, smemObj, typeConverter,
                                             linearWarpId, matrixStorePath);
    }

    if (!res)
      return failure();
    rewriter.replaceOp(op, res);
    return success();
  }

  Value
  convertLayoutUnitTilesPerWarp(int opIdx, ConversionPatternRewriter &rewriter,
                                Location loc, Value tensor,
                                DotOperandEncodingAttr encoding,
                                const SharedMemoryObject &smemObj,
                                const LLVMTypeConverter *typeConverter,
                                Value linearWarpId,
                                bool matrixStorePath) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    assert((opIdx == 0 || opIdx == 1) && "unexpected operand idx");
    matrixStorePath = matrixStorePath && opIdx == 0;
    auto tensorTy = cast<MemDescType>(tensor.getType());
    ArrayRef<int64_t> shape = tensorTy.getShape();
    auto rank = shape.size();
    int kDimIdx = opIdx == 0 ? rank - 1 : rank - 2;
    int nonKDimIdx = opIdx == 0 ? rank - 2 : rank - 1;
    int kDimIdx2D = opIdx == 0 ? 1 : 0;
    int nonKDimIdx2D = opIdx == 0 ? 0 : 1;

    auto sharedLayout = cast<HCUMlsSharedEncodingAttr>(tensorTy.getEncoding());
    auto shapeE = getMlsExpandedShape(sharedLayout, shape, MlsTileKind::Elems);
    auto shapeS = getMlsExpandedShape(sharedLayout, shape, MlsTileKind::Shared);

    auto mfmaLayout = cast<AMDMfmaEncodingAttr>(encoding.getParent());
    assert(((opIdx==0 && mfmaLayout.getInstrsPerWarp()[0] == 1) ||
            (opIdx==1 && mfmaLayout.getInstrsPerWarp()[1] == 1)) &&
            "only support unit tiles per warp mfma layout!");
    auto warpsPerCTA = mfmaLayout.getWarpsPerCTA();

    auto mlsElemBitTyKind = getMlsElemBitTyKind(encoding.getMlsScaledExt());
    auto flow = getMlsDataFlowInfo(mlsElemBitTyKind);
    bool isB4PackedDotOperand = flow.usesPackedShape(MlsDataFlowView::DotOperand);
    auto elemTy = tensorTy.getElementType();
    auto kWidth = encoding.getKWidth();
    auto elemsPerInstr =
        mfmaLayout.getInstrShapeForScaledOperand(kWidth, mlsElemBitTyKind, opIdx);

    int64_t mfmaInstrNonK;
    int64_t mfmaInstrK;
    // TODO(Lixun): make it simpler
    // getInstrShapeForOperand always returns a 2D vector
    if (rank == 3) {
      mfmaInstrNonK = elemsPerInstr[nonKDimIdx - 1];
      mfmaInstrK = elemsPerInstr[kDimIdx - 1];
    } else {
      mfmaInstrNonK = elemsPerInstr[nonKDimIdx];
      mfmaInstrK = elemsPerInstr[kDimIdx];
    }

    if (mfmaInstrNonK > shapeE[nonKDimIdx] || mfmaInstrK > shapeE[kDimIdx]) {
      // This pattern does not support cases tensor shape is smaller than
      // one instruction size, it will be processed by LinearLayout converter
      return Value();
    }

    auto numReps =
        mfmaLayout.getRepForScaledOperand(shapeE, kWidth, mlsElemBitTyKind, opIdx);
    auto numRepNonK = numReps[nonKDimIdx];
    auto numRepK = numReps[kDimIdx];
    auto repB = numReps[0];
    // TODO(Lixun): make it simpler
    // getRepForOperand always returns a 3D vector
    if (rank == 2) {
      numRepNonK = numReps[nonKDimIdx + 1];
      numRepK = numReps[kDimIdx + 1];
    }

    unsigned iWarpSize = targetInfo.getWarpSize();
    assert(iWarpSize == 64);

    auto warpOrder = triton::gpu::getMatrixOrder(rank, /*rowMajor*/ true);
    Value spatialWarpId = getWarpIdInBlock(
        rewriter, loc, linearWarpId, warpsPerCTA, mfmaInstrNonK,
        shapeE[nonKDimIdx], nonKDimIdx, warpOrder);

    int numSubBlocks = 1;
    // numOfElemsPerThreadPerMfmaInstr
    int numOfElems =
        (mfmaInstrNonK * mfmaInstrK * numSubBlocks / iWarpSize) /
        (isB4PackedDotOperand ? 2 : 1);
    assert(numOfElems >= 1);

    unsigned int maxNumWarps = shapeE[nonKDimIdx] / mfmaInstrNonK;
    int warpsPerBlockNonK = std::min(warpsPerCTA[nonKDimIdx], maxNumWarps);
    int warpsPerBatch =
        rank == 3 ? std::min<unsigned>(shapeE[0], warpsPerCTA[0]) : 1;
    Value warpIdInBatch = b.urem(linearWarpId, b.i32_val(warpsPerBatch));
    elemTy = typeConverter->convertType(elemTy);

    // 1. get ds_read_matrix inst info
    auto mlsTile = sharedLayout.getMlsTile();
    bool kMajor = sharedLayout.getOrder()[0] == kDimIdx;
    assert(kMajor && "m16n16 mfma & mls only support kMajor layout!");

    auto mlsInsn = MlsInsn::selectOrGetMlsInsn(
                               mlsTile[nonKDimIdx2D], mlsTile[kDimIdx2D],
                               sharedLayout.getElemBitWidth(),
                               sharedLayout.getElemBitTyKind(),
                               opIdx, kMajor,
                               sharedLayout.getVersion(),
                               static_cast<MlsInterleaveKind>(sharedLayout.getAlt2Kind()));

    auto dsInsnAttr = mlsInsn->getDsReadMatrixInsnAttr();

    assert(mlsTile[kDimIdx2D] % mfmaInstrK == 0 &&
           shapeE[kDimIdx] % mlsTile[kDimIdx2D] == 0 &&
           "mls tile kdim and shape kdim should be divisible!");

    unsigned mlsNumRepK = shapeE[kDimIdx] / mlsTile[kDimIdx2D];
    unsigned mlsNumRepNonK = numRepNonK;
    SmallVector<unsigned> mlsNumReps{mlsNumRepNonK, mlsNumRepK};
    if (opIdx == 1) {
      std::swap(mlsNumReps[0], mlsNumReps[1]);
    }

    // 2. compute ds_read_matrix offsets into the full parent MLS buffer
    auto parentShape = getMlsParentMatrixShape(tensorTy);
    auto offsets = computeDsReadMatrixOffsets(rewriter, loc,
                               sharedLayout, dsInsnAttr,
                               spatialWarpId, warpsPerBlockNonK, mlsNumReps,
                               smemObj, parentShape, shape, matrixStorePath);
    Value smemBase = smemObj.getBase();
    Type smemPtrTy = ptr_ty(rewriter.getContext(), 3);

    assert(mlsInsn->getElemBitWidth() == 16 || mlsInsn->getElemBitWidth() == 8);
    unsigned dsRepPerWarpNonK = dsInsnAttr.instrsPerWarp[nonKDimIdx2D];
    unsigned dsRepPerWarpK = dsInsnAttr.instrsPerWarp[kDimIdx2D];
    unsigned dsLoadsPerK = dsRepPerWarpNonK * dsRepPerWarpK;
    unsigned numOfElemsPerDsInsn =
        (dsInsnAttr.instrShape[kDimIdx2D] *
         dsInsnAttr.instrShape[nonKDimIdx2D] / iWarpSize) /
        (isB4PackedDotOperand ? 2 : 1);

    // Within-tile half view of a full-tile MLS write: trim nonK ds reps and
    // fold parent dsByteOffsets delta into the GEP (offset attr is immediate).
    Value nonKLoadBaseVal = b.i32_val(0);
    bool withinTileHalfView = false;
    unsigned elemByteWidth =
        std::max<unsigned>(1, elemTy.getIntOrFloatBitWidth() / 8);
    {
      auto offs = smemObj.getOffsets();
      if (offs.size() >= 2 && parentShape.size() >= 2) {
        unsigned offBase = offs.size() - 2;
        int64_t viewNonK = shape[nonKDimIdx];
        int64_t parentNonK = parentShape[nonKDimIdx];
        if (viewNonK > 0 && parentNonK > viewNonK &&
            parentNonK % viewNonK == 0 &&
            (!matrixStorePath ||
             viewNonK / warpsPerBlockNonK <
                 static_cast<int64_t>(mlsTile[nonKDimIdx2D]))) {
          unsigned parts = static_cast<unsigned>(parentNonK / viewNonK);
          if (dsRepPerWarpNonK % parts == 0) {
            dsRepPerWarpNonK /= parts;
            dsLoadsPerK = dsRepPerWarpNonK * dsRepPerWarpK;
            withinTileHalfView = true;
            // The subslice selects the parent MLS tile.  Within that tile,
            // the spatial warp selects its ds-read half.
            nonKLoadBaseVal = matrixStorePath
                ? b.mul(spatialWarpId, b.i32_val(dsRepPerWarpNonK))
                : b.mul(b.udiv(offs[offBase + nonKDimIdx2D],
                               b.i32_val(viewNonK)),
                        b.i32_val(dsRepPerWarpNonK));
          }
        }
      }
    }
    auto selectDsByteOffsetUnit = [&](Value parentLoadIdxVal) -> Value {
      Value selected = b.i32_val(dsInsnAttr.dsByteOffsets[0]);
      for (unsigned i = 1; i < dsInsnAttr.dsByteOffsets.size(); ++i) {
        selected =
            b.select(b.icmp_eq(parentLoadIdxVal, b.i32_val(i)),
                     b.i32_val(dsInsnAttr.dsByteOffsets[i]), selected);
      }
      return selected;
    };

    // 3. create ds_read_matrix ops
    SmallVector<Value> loadedValues;
    for (int batchIdx = 0; batchIdx < repB; ++batchIdx) {
      int operandSize = shapeS[rank - 1] * shapeS[rank - 2];
      Value batchOffset = b.mul(b.i32_val(operandSize),
                              b.add(warpIdInBatch, b.i32_val(batchIdx * warpsPerBatch)));
      for (int mlsNonK = 0; mlsNonK < mlsNumRepNonK; ++mlsNonK) {
        for (int mlsKIdx = 0; mlsKIdx < mlsNumRepK; ++mlsKIdx) {
          auto loadOffset = offsets[mlsNonK * mlsNumRepK + mlsKIdx];
          loadOffset = b.add(loadOffset, batchOffset);
          for (int loadIdx = 0; loadIdx < dsLoadsPerK; ++loadIdx) {
            unsigned loadNonKIdx = loadIdx / dsRepPerWarpK;
            unsigned loadKIdx = loadIdx % dsRepPerWarpK;
            unsigned localByteOff = dsInsnAttr.dsByteOffsets[loadIdx];
            Value gepElems = loadOffset;
            if (withinTileHalfView) {
              Value parentLoadIdxVal = b.add(
                  b.mul(b.add(nonKLoadBaseVal, b.i32_val(loadNonKIdx)),
                        b.i32_val(dsRepPerWarpK)),
                  b.i32_val(loadKIdx));
              Value parentByteOff = selectDsByteOffsetUnit(parentLoadIdxVal);
              Value adjElems =
                  b.sdiv(b.sub(parentByteOff, b.i32_val(localByteOff)),
                         b.i32_val(elemByteWidth));
              gepElems = b.add(loadOffset, adjElems);
            }
            Value loadAddress = b.gep(smemPtrTy, elemTy, smemBase, gepElems);

            auto dsInsnResTy = getDsReadMatrixInsnResType(loc, rewriter, tensorTy,
                                                                dsInsnAttr,
                                                                mlsInsn->getElemBitWidth());
            Value loadedValue = generateDsReadMatrixOp(loc, rewriter, dsInsnAttr.insn,
                                                       dsInsnResTy,
                                                       loadAddress,
                                                       localByteOff,
                                                       dsInsnAttr.flags);
            auto unpackedValues = unpackDsReadMatrixInsnRes(loc, rewriter, tensorTy,
                                                            dsInsnAttr,
                                                            mlsInsn->getElemBitWidth(),
                                                            loadedValue);
            loadedValues.append(unpackedValues);
          }
        }
      }
    }

    assert(loadedValues.size() ==
           repB * mlsNumRepNonK * mlsNumRepK * dsLoadsPerK *
               numOfElemsPerDsInsn);

    MLIRContext *ctx = mfmaLayout.getContext();
    Type structTy = LLVM::LLVMStructType::getLiteral(
        ctx, SmallVector<Type>(loadedValues.size(), loadedValues[0].getType()));
    auto result =
        packLLElements(loc, typeConverter, loadedValues, rewriter, structTy);
    return result;
  }

  Value
  convertLayoutMultiTilesPerWarp(int opIdx, ConversionPatternRewriter &rewriter,
                                 Location loc, Value tensor,
                                 DotOperandEncodingAttr encoding,
                                const SharedMemoryObject &smemObj,
                                const LLVMTypeConverter *typeConverter,
                                Value linearWarpId,
                                bool matrixStorePath) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    assert((opIdx == 0 || opIdx == 1) && "unexpected operand idx");
    matrixStorePath = matrixStorePath && opIdx == 0;
    auto tensorTy = cast<MemDescType>(tensor.getType());
    ArrayRef<int64_t> shape = tensorTy.getShape();
    auto rank = shape.size();
    int kDimIdx = opIdx == 0 ? rank - 1 : rank - 2;
    int nonKDimIdx = opIdx == 0 ? rank - 2 : rank - 1;
    int kDimIdx2D = opIdx == 0 ? 1 : 0;
    int nonKDimIdx2D = opIdx == 0 ? 0 : 1;

    auto mfmaLayout = cast<AMDMfmaEncodingAttr>(encoding.getParent());
    auto warpsPerCTA = mfmaLayout.getWarpsPerCTA();

    auto sharedLayout = cast<HCUMlsSharedEncodingAttr>(tensorTy.getEncoding());
    auto shapeE = getMlsExpandedShape(sharedLayout, shape, MlsTileKind::Elems);
    auto shapeS = getMlsExpandedShape(sharedLayout, shape, MlsTileKind::Shared);

    auto mlsElemBitTyKind = getMlsElemBitTyKind(encoding.getMlsScaledExt());
    auto flow = getMlsDataFlowInfo(mlsElemBitTyKind);
    bool isB4PackedDotOperand = flow.usesPackedShape(MlsDataFlowView::DotOperand);
    auto elemTy = tensorTy.getElementType();
    auto kWidth = encoding.getKWidth();
    auto elemsPerInstr =
        mfmaLayout.getInstrShapeForScaledOperand(kWidth, mlsElemBitTyKind, opIdx);

    int64_t mfmaInstrNonK;
    int64_t mfmaInstrK;
    // TODO(Lixun): make it simpler
    // getInstrShapeForOperand always returns a 2D vector
    if (rank == 3) {
      mfmaInstrNonK = elemsPerInstr[nonKDimIdx - 1];
      mfmaInstrK = elemsPerInstr[kDimIdx - 1];
    } else {
      mfmaInstrNonK = elemsPerInstr[nonKDimIdx];
      mfmaInstrK = elemsPerInstr[kDimIdx];
    }

    if (mfmaInstrNonK > shapeE[nonKDimIdx] || mfmaInstrK > shapeE[kDimIdx]) {
      // This pattern does not support cases tensor shape is smaller than
      // one instruction size, it will be processed by LinearLayout converter
      return Value();
    }


    auto numReps =
        mfmaLayout.getRepForScaledOperand(shapeE, kWidth, mlsElemBitTyKind, opIdx);
    auto numRepNonK = numReps[nonKDimIdx];
    auto numRepK = numReps[kDimIdx];
    auto repB = numReps[0];
    // TODO(Lixun): make it simpler
    // getRepForOperand always returns a 3D vector
    if (rank == 2) {
      numRepNonK = numReps[nonKDimIdx + 1];
      numRepK = numReps[kDimIdx + 1];
    }

    auto mfmaTileNonK = opIdx == 0 ? mfmaLayout.getMfmaTile()[0] : mfmaLayout.getMfmaTile()[1];
    auto mfmaTileK = mfmaInstrK;
    auto mfmaInstrsPerWarpNonK = mfmaLayout.getInstrsPerWarp()[nonKDimIdx2D];
    auto mfmaTileNumReps =
        mfmaLayout.getMfmaTileRepForScaledOperand(shapeE, kWidth, mlsElemBitTyKind, opIdx);
    auto mfmaTileNumRepNonK = mfmaTileNumReps[nonKDimIdx];
    if (rank == 2) {
      mfmaTileNumRepNonK = mfmaTileNumReps[nonKDimIdx + 1];
    }

    unsigned iWarpSize = targetInfo.getWarpSize();
    assert(iWarpSize == 64);

    auto warpOrder = triton::gpu::getMatrixOrder(rank, /*rowMajor*/ true);
    Value spatialWarpId = getWarpIdInBlock(
        rewriter, loc, linearWarpId, warpsPerCTA, mfmaTileNonK,
        shapeE[nonKDimIdx], nonKDimIdx, warpOrder);

    int numSubBlocks = 1;
    // numOfElemsPerThreadPerMfmaInstr
    int numOfElems =
        (mfmaInstrNonK * mfmaInstrK * numSubBlocks / iWarpSize) /
        (isB4PackedDotOperand ? 2 : 1);
    assert(numOfElems >= 1);

    unsigned int maxNumWarps = shapeE[nonKDimIdx] / mfmaTileNonK;
    int warpsPerBlockNonK = std::min(warpsPerCTA[nonKDimIdx], maxNumWarps);
    int warpsPerBatch =
        rank == 3 ? std::min<unsigned>(shapeE[0], warpsPerCTA[0]) : 1;
    Value warpIdInBatch = b.urem(linearWarpId, b.i32_val(warpsPerBatch));
    elemTy = typeConverter->convertType(elemTy);

    // 1. get ds_read_matrix inst info
    auto mlsTile = sharedLayout.getMlsTile();
    bool kMajor = sharedLayout.getOrder()[0] == kDimIdx;

    auto mlsInsn = MlsInsn::selectOrGetMlsInsn(
                               mlsTile[nonKDimIdx2D], mlsTile[kDimIdx2D],
                               sharedLayout.getElemBitWidth(),
                               sharedLayout.getElemBitTyKind(),
                               opIdx, kMajor,
                               sharedLayout.getVersion(),
                               static_cast<MlsInterleaveKind>(sharedLayout.getAlt2Kind()));

    auto dsInsnAttr = mlsInsn->getDsReadMatrixInsnAttr();

    assert(mlsTile[kDimIdx2D] % mfmaInstrK == 0 &&
           shapeE[kDimIdx] % mlsTile[kDimIdx2D] == 0 &&
           "mls tile kdim and shape kdim should be divisible!");

    unsigned mlsNumRepK = shapeE[kDimIdx] / mlsTile[kDimIdx2D];
    unsigned mlsNumRepNonK = mfmaTileNumRepNonK;
    SmallVector<unsigned> mlsNumReps{mlsNumRepNonK, mlsNumRepK};
    if (opIdx == 1) {
      std::swap(mlsNumReps[0], mlsNumReps[1]);
    }

    // 2. compute ds_read_matrix offsets into the full parent MLS buffer
    auto parentShape = getMlsParentMatrixShape(tensorTy);
    auto offsets = computeDsReadMatrixOffsets(rewriter, loc,
                               sharedLayout, dsInsnAttr,
                               spatialWarpId, warpsPerBlockNonK, mlsNumReps,
                               smemObj, parentShape, shape, matrixStorePath);
    Value smemBase = smemObj.getBase();
    Type smemPtrTy = ptr_ty(rewriter.getContext(), 3);

    assert(mlsInsn->getElemBitWidth() == 16 || mlsInsn->getElemBitWidth() == 8);

    // 3. create ds_read_matrix ops
    unsigned dsRepPerWarpNonK = dsInsnAttr.instrsPerWarp[nonKDimIdx2D];
    unsigned dsRepPerWarpK = dsInsnAttr.instrsPerWarp[kDimIdx2D];

    unsigned mfmaGroupPerDsInsn = dsInsnAttr.instrShape[nonKDimIdx2D] / mfmaInstrNonK;
    assert(mfmaGroupPerDsInsn >= 1);
    unsigned mfmaKStridePerDsInsn = dsInsnAttr.instrShape[kDimIdx2D] / mfmaTileK;

    // Half-view of a full-tile MLS write: encoding still describes the parent
    // tile (e.g. mlsTile 64x64 → 8 ds_reads) but this LocalLoad only needs the
    // MFMA coverage for the view (e.g. 4 ds_reads → 32 elems).
    unsigned nonKDsNeeded =
        (mfmaInstrsPerWarpNonK + mfmaGroupPerDsInsn - 1) / mfmaGroupPerDsInsn;
    if (nonKDsNeeded < dsRepPerWarpNonK)
      dsRepPerWarpNonK = nonKDsNeeded;
    unsigned dsLoadsPerK = dsRepPerWarpNonK * dsRepPerWarpK;

    unsigned numOfElemsPerDsInsn = (dsInsnAttr.instrShape[kDimIdx2D] *
                                    dsInsnAttr.instrShape[nonKDimIdx2D] / iWarpSize) / (isB4PackedDotOperand ? 2 : 1);
    unsigned totalElemsMfma = repB * (mfmaTileNumRepNonK * mfmaInstrsPerWarpNonK) * numRepK * numOfElems;
    unsigned totalElemsMls  = repB * mlsNumRepNonK * mlsNumRepK * dsLoadsPerK * numOfElemsPerDsInsn;
    assert(totalElemsMfma == totalElemsMls &&
           "MLS LocalLoad elem count must match MFMA operand");

    // Within-tile half of a full-tile MLS write: pick the matching half of
    // dsByteOffsets. Offset comes from SharedMemoryObject (often an
    // extractvalue, not a foldable constant). ds_read_matrix takes an immediate
    // offset attr, so fold the parent-vs-local byte delta into the GEP.
    Value nonKLoadBaseVal = b.i32_val(0);
    bool withinTileHalfView = false;
    unsigned elemByteWidth =
        std::max<unsigned>(1, elemTy.getIntOrFloatBitWidth() / 8);
    {
      auto offs = smemObj.getOffsets();
      if (offs.size() >= 2 && parentShape.size() >= 2) {
        unsigned offBase = offs.size() - 2;
        int64_t viewNonK = shape[nonKDimIdx];
        int64_t parentNonK = parentShape[nonKDimIdx];
        if (viewNonK > 0 && parentNonK > viewNonK &&
            parentNonK % viewNonK == 0 &&
            (!matrixStorePath ||
             viewNonK / warpsPerBlockNonK <
                 static_cast<int64_t>(mlsTile[nonKDimIdx2D]))) {
          unsigned parts = static_cast<unsigned>(parentNonK / viewNonK);
          unsigned dsRepFull = dsInsnAttr.instrsPerWarp[nonKDimIdx2D];
          if (dsRepFull == dsRepPerWarpNonK * parts) {
            withinTileHalfView = true;
            // The subslice selects the parent MLS tile.  Within that tile,
            // the spatial warp selects its ds-read half.
            nonKLoadBaseVal = matrixStorePath
                ? b.mul(spatialWarpId, b.i32_val(dsRepPerWarpNonK))
                : b.mul(b.udiv(offs[offBase + nonKDimIdx2D],
                               b.i32_val(viewNonK)),
                        b.i32_val(dsRepPerWarpNonK));
          }
        }
      }
    }
    auto selectDsByteOffset = [&](Value parentLoadIdxVal) -> Value {
      Value selected = b.i32_val(dsInsnAttr.dsByteOffsets[0]);
      for (unsigned i = 1; i < dsInsnAttr.dsByteOffsets.size(); ++i) {
        selected =
            b.select(b.icmp_eq(parentLoadIdxVal, b.i32_val(i)),
                     b.i32_val(dsInsnAttr.dsByteOffsets[i]), selected);
      }
      return selected;
    };

    SmallVector<Value> loadedValues(totalElemsMfma);
    for (int batchIdx = 0; batchIdx < repB; ++batchIdx) {
      int operandSize = shapeS[rank - 1] * shapeS[rank - 2];
      Value batchOffset = b.mul(b.i32_val(operandSize),
                              b.add(warpIdInBatch, b.i32_val(batchIdx * warpsPerBatch)));
      for (int mlsNonK = 0; mlsNonK < mlsNumRepNonK; ++mlsNonK) {
        int mfmaTileNonKIdx = mlsNonK;
        for (int mlsKIdx = 0; mlsKIdx < mlsNumRepK; ++mlsKIdx) {
          int mfmaTileKIdx = mlsKIdx * (mlsTile[kDimIdx2D] / mfmaTileK);

          auto loadOffset = offsets[mlsNonK * mlsNumRepK + mlsKIdx];
          loadOffset = b.add(loadOffset, batchOffset);
          for (int loadIdx = 0; loadIdx < dsLoadsPerK; ++loadIdx) {
            unsigned loadNonKIdx = loadIdx / dsRepPerWarpK;
            unsigned loadKIdx = loadIdx % dsRepPerWarpK;
            unsigned localByteOff = dsInsnAttr.dsByteOffsets[loadIdx];
            Value gepElems = loadOffset;
            if (withinTileHalfView) {
              Value parentLoadIdxVal = b.add(
                  b.mul(b.add(nonKLoadBaseVal, b.i32_val(loadNonKIdx)),
                        b.i32_val(dsRepPerWarpK)),
                  b.i32_val(loadKIdx));
              Value parentByteOff = selectDsByteOffset(parentLoadIdxVal);
              Value adjElems =
                  b.sdiv(b.sub(parentByteOff, b.i32_val(localByteOff)),
                         b.i32_val(elemByteWidth));
              gepElems = b.add(loadOffset, adjElems);
            }
            Value loadAddress = b.gep(smemPtrTy, elemTy, smemBase, gepElems);

            auto dsInsnResTy = getDsReadMatrixInsnResType(loc, rewriter, tensorTy,
                                                                dsInsnAttr,
                                                                mlsInsn->getElemBitWidth());
            Value loadedValue = generateDsReadMatrixOp(loc, rewriter, dsInsnAttr.insn,
                                              dsInsnResTy,
                                                       loadAddress,
                                                       localByteOff,
                                                       dsInsnAttr.flags);
            auto unpackedValues = unpackDsReadMatrixInsnRes(loc, rewriter, tensorTy,
                                                            dsInsnAttr, mlsInsn->getElemBitWidth(),
                                                            loadedValue);

            assert(unpackedValues.size() % mfmaGroupPerDsInsn == 0 &&
                  "unpackedValues should be divisible by dsGroupSize");
            unsigned elemsPerGroup = unpackedValues.size() / mfmaGroupPerDsInsn;

            for (int i = 0; i < mfmaGroupPerDsInsn; ++i) {
              int mfmaInsnNonKIdxInTile = loadNonKIdx * mfmaGroupPerDsInsn + i;
              int mfmaInsnKIdxInTile = loadKIdx * mfmaKStridePerDsInsn;
              unsigned groupOffset = batchIdx * (mfmaTileNumRepNonK * mfmaInstrsPerWarpNonK) * numRepK * numOfElems /* batch idx */ +
                                     mfmaTileNonKIdx * mfmaInstrsPerWarpNonK * numRepK * numOfElems /* block idx*/ +
                                     mfmaInsnNonKIdxInTile * numRepK * numOfElems + /* inblock idx */
                                     (mfmaTileKIdx + mfmaInsnKIdxInTile) * numOfElems;

              for (int j = 0; j < elemsPerGroup; ++j) {
                loadedValues[groupOffset + j] = unpackedValues[i * elemsPerGroup + j];
              }
            }
          }
        }
      }
    }

    assert(loadedValues.size() == totalElemsMfma);

    MLIRContext *ctx = mfmaLayout.getContext();
    Type structTy = LLVM::LLVMStructType::getLiteral(
        ctx, SmallVector<Type>(loadedValues.size(), loadedValues[0].getType()));
    auto result =
        packLLElements(loc, typeConverter, loadedValues, rewriter, structTy);
    return result;
  }

  llvm::SmallVector<Value>
  computeDsReadMatrixOffsets(ConversionPatternRewriter &rewriter, Location loc,
                             const HCUMlsSharedEncodingAttr &sharedLayout,
                             const DsReadMatrixInsnAttr &dsInsnAttr,
                             Value warpNonKId, int warpsPerBlockNonK,
                             ArrayRef<unsigned> mlsReps, SharedMemoryObject smemObj,
                             ArrayRef<int64_t> parentShape,
                             ArrayRef<int64_t> viewShape,
                             bool matrixStorePath) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto opIdx = sharedLayout.getOpIdx();
    auto mlsTile = sharedLayout.getMlsTile();
    auto kDimIdx = opIdx == 0 ? 1 : 0;
    auto nonKDimIdx = opIdx == 0 ? 0 : 1;

    assert(mlsReps.size() == 2);
    auto numRepK = mlsReps[kDimIdx];
    auto numRepNonK = mlsReps[nonKDimIdx];

    const auto numBlocks = numRepNonK;
    const auto numMlsTilesPerBlock = numRepK;
    const auto blockSize = numMlsTilesPerBlock;

    // Layout geometry is always the full parent MLS buffer (allocShape).
    // mlsReps / warps cover only this LocalLoad's view (possibly a half).
    auto shapeE =
        getMlsExpandedShape(sharedLayout, parentShape, MlsTileKind::Elems);
    auto shapeS =
        getMlsExpandedShape(sharedLayout, parentShape, MlsTileKind::Shared);
    auto divFactor = product(shapeE) / product(shapeS);
    auto divFactor0 = shapeE[0] / shapeS[0];
    auto divFactor1 = shapeE[1] / shapeS[1];

    SmallVector<unsigned> flattenedMlsShape{parentShape.begin(),
                                            parentShape.end()};
    flattenedMlsShape[0] =
        shapeS[0] * divFactor0 / sharedLayout.getMlsTile()[0];
    flattenedMlsShape[1] =
        shapeS[1] * divFactor1 / sharedLayout.getMlsTile()[1];

    // memdesc_subslice logical offsets → parent MLS coordinates.
    // Within-tile halves (OnePTwoC A: mlsTile covers parent, view is half) are
    // handled by selecting the matching dsByteOffsets half in the caller.
    // Here handle tile-aligned subslices (offset is a multiple of mlsTile).
    Value subsliceNonKTileOff = b.i32_val(0);
    Value subsliceKTileOff = b.i32_val(0);
    auto offs = smemObj.getOffsets();
    if (offs.size() >= 2 && viewShape.size() >= 2) {
      unsigned offBase = offs.size() - 2;
      unsigned tileNonK = mlsTile[nonKDimIdx];
      unsigned tileK = mlsTile[kDimIdx];
      assert(tileNonK > 0 && tileK > 0);
      int64_t viewNonK = viewShape[nonKDimIdx];
      // Only apply tile-index offsets when the view itself is tile-sized (or
      // larger). Within-tile half views leave tile udiv at 0.
      if (viewNonK >= static_cast<int64_t>(tileNonK)) {
        subsliceNonKTileOff =
            b.udiv(offs[offBase + nonKDimIdx], b.i32_val(tileNonK));
      }
      subsliceKTileOff =
          b.udiv(offs[offBase + kDimIdx], b.i32_val(tileK));
    }

    llvm::SmallVector<Value> offsets(numBlocks * numMlsTilesPerBlock);
    bool nonKSlicedView = matrixStorePath &&
        viewShape[nonKDimIdx] < parentShape[nonKDimIdx];
    bool warpWithinNonKTile =
        viewShape[nonKDimIdx] / warpsPerBlockNonK <
        static_cast<int64_t>(mlsTile[nonKDimIdx]);
    for (int block = 0; block < numBlocks; ++block) {
      int blockNonKOffset = block * warpsPerBlockNonK;

      for (int mlsTileIdx = 0; mlsTileIdx < numMlsTilesPerBlock; ++mlsTileIdx) {
        // A within-tile WDRA subslice uses the parent tile plus a caller-side
        // dsByteOffsets half.  Unsliced operands (notably B) retain the
        // [block][warp] ownership matched by the direct non-contiguous store.
        Value mlsNonKOff = nonKSlicedView && warpWithinNonKTile
            ? b.add(b.i32_val(block), subsliceNonKTileOff)
            : b.add(b.add(b.i32_val(blockNonKOffset), warpNonKId),
                    subsliceNonKTileOff);
        Value mlsKOff = b.add(b.i32_val(mlsTileIdx), subsliceKTileOff);

        std::array<Value, 2> mlsCoords =
            opIdx == 0 ? std::array<Value, 2>{mlsNonKOff, mlsKOff}
                       : std::array<Value, 2>{mlsKOff, mlsNonKOff};

        Value mlsOff = b.mul(linearize(rewriter, loc, mlsCoords,
                                        flattenedMlsShape, sharedLayout.getOrder()),
                            b.i32_val(product(sharedLayout.getMlsTile())/divFactor));
        offsets[block * blockSize + mlsTileIdx] = mlsOff;
      }
    }

    return offsets;
  }

  // Last `rank` dims of allocShape when the memdesc is a subview of a larger
  // MLS buffer; otherwise the view shape itself.
  static SmallVector<int64_t>
  getMlsParentMatrixShape(MemDescType tensorTy) {
    ArrayRef<int64_t> shape = tensorTy.getShape();
    ArrayRef<int64_t> allocShape = tensorTy.getAllocShape();
    SmallVector<int64_t> parent(shape.begin(), shape.end());
    if (allocShape.size() >= shape.size()) {
      parent.assign(allocShape.end() - shape.size(), allocShape.end());
      for (size_t i = 0; i < parent.size(); ++i)
        if (parent[i] < shape[i])
          parent[i] = shape[i];
    }
    return parent;
  }

  Type getDsReadMatrixInsnResType(Location loc, RewriterBase &rewriter,
                              const MemDescType &tensorTy,
                              const DsReadMatrixInsnAttr &dsInsnAttr,
                              unsigned elemBitWidth,
                              unsigned iWarpSize = 64) const {
    auto dsInsnElemTy = tensorTy.getElementType();
    auto dsInsnNumOfElems = dsInsnAttr.instrShape[0] * dsInsnAttr.instrShape[1] / iWarpSize;

    if (elemBitWidth == 8) {
      dsInsnElemTy = i32_ty;
      dsInsnNumOfElems = 4;
    }
    return vec_ty(dsInsnElemTy, dsInsnNumOfElems);
  }

  Value
  generateDsReadMatrixOp(Location loc, RewriterBase &rewriter,
                         StringRef dsOpName,
                         Type resType,
                         Value soffset,
                         unsigned offset,
                         unsigned flags) const {
    OperationState loweredOp(loc, dsOpName);
    loweredOp.addTypes(resType);
    loweredOp.addOperands(soffset);
    loweredOp.addAttribute("offset", rewriter.getI32IntegerAttr(offset));

    int8_t elem3 = MLS_DS_FLAGS_GET_ELEM3(flags);
    int8_t row3 = MLS_DS_FLAGS_GET_ROW3(flags);
    int8_t col3 = MLS_DS_FLAGS_GET_COL3(flags);
    int8_t alt2 = MLS_DS_FLAGS_GET_ALT2(flags);

    loweredOp.addAttribute("element", rewriter.getI8IntegerAttr(elem3));
    loweredOp.addAttribute("row", rewriter.getI8IntegerAttr(row3));
    loweredOp.addAttribute("col", rewriter.getI8IntegerAttr(col3));
    loweredOp.addAttribute("alt", rewriter.getI8IntegerAttr(alt2));

    return rewriter.create(loweredOp)->getResult(0);
  }

  SmallVector<Value>
  unpackDsReadMatrixInsnRes(Location loc, RewriterBase &rewriter,
                           const MemDescType &tensorTy,
                           const DsReadMatrixInsnAttr &dsInsnAttr,
                           unsigned elemBitWidth,
                           Value dsLoadedValue) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto dsInsnRetTy = cast<VectorType>(dsLoadedValue.getType());
    auto dsInsnElemTy = dsInsnRetTy.getElementType();
    auto dsInsnNumOfElems = dsInsnRetTy.getNumElements();
    bool needCvt = false;
    unsigned numExtracts = 1;
    if (elemBitWidth == 8) {
      numExtracts = dsInsnElemTy.getIntOrFloatBitWidth() / elemBitWidth;
    }

    SmallVector<Value> retValues;
    auto elemTy = typeConverter->convertType(tensorTy.getElementType());
    for (int elemId = 0; elemId < dsInsnNumOfElems; ++elemId) {
      Value elemVal = b.extract_element(dsInsnElemTy, dsLoadedValue, b.i32_val(elemId));

      if (numExtracts != 1) {
        auto vecVal = b.bitcast(elemVal, vec_ty(elemTy, numExtracts));
        for (int extractId = 0; extractId < numExtracts; ++extractId) {
          Value subElemVal = b.extract_element(elemTy, vecVal, b.i32_val(extractId));
          retValues.push_back(subElemVal);
        }
      } else {
        retValues.push_back(elemVal);
      }
    }
    return retValues;
  }
};

//--------------------------------------------------------------------------------------------------
// DotOperand VGPR → MLS shared via ds_write_matrix_format (reverse of LocalLoad)
//--------------------------------------------------------------------------------------------------


} // namespace

namespace mlir::triton::AMD {
void populateMLSOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                 const TargetInfo &targetInfo,
                                 RewritePatternSet &patterns,
                                 ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                 PatternBenefit benefit) {
  patterns.add<MLSLocalAllocOpConversion,
               MLSMatrixLoadToLocalOpConversion,
               MLSMatrixStoreFromRegOpConversion,
               MLSLocalLoadOpConversion>(typeConverter, targetInfo,
                                         axisInfoAnalysis, benefit);
}
} // namespace mlir::triton::AMD
