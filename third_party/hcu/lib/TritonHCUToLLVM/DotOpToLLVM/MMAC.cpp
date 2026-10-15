/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
// Modified by Hygon Information Technology Co., Ltd., 2026.

#include "third_party/amd/lib/TritonAMDGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "TritonAMDGPUTransforms/MfmaGroup.h"
#include "third_party/amd/lib/TritonAMDGPUToLLVM/Utility.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "TritonHCU/MlsGroup.h"
#include "TritonHCU/ScaleMemoryPlan.h"
#include "TritonHCU/WaspSplitPlan.h"
#include "TritonHCU/Utility.h"

using namespace mlir;
using namespace mlir::triton::HCU;

namespace {

Value getUniformWarpId(Operation *op, ConversionPatternRewriter &rewriter,
                       Location loc) {
  OpBuilder::InsertionGuard guard(rewriter);
  if (!op->getParentOfType<triton::gpu::WarpSpecializeOp>()) {
    auto func = op->getParentOfType<LLVM::LLVMFuncOp>();
    assert(func && "native scale instruction must be inside an LLVM function");
    rewriter.setInsertionPointToStart(&func.getBody().front());
  }
  Value warp = getLaneAndWarpId(rewriter, loc).second;
  return LLVM::createLLVMIntrinsicCallOp(
             rewriter, loc, "llvm.amdgcn.readfirstlane", {i32_ty}, {warp})
      .getResult(0);
}

using ::mlir::LLVM::AMD::scaleDotElemTypeToMLIRType;
using ::mlir::LLVM::AMD::shuffleXor;
using ::mlir::triton::gpu::AMDMfmaEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::LinearEncodingAttr;
using ::mlir::triton::gpu::MmacLayout;

using ValueTable = std::map<std::array<int, 3>, Value>;

/// Get matrix format flag passed through BLGP/CBSZ args in V_MFMA_*_F8F6F4
/// instructions.
///
/// Values:
/// - 0: E4M3(FP8)
/// - 1: E5M2(BF8)
/// - 2: E2M3(FP6)
/// - 3: E3M2(BF6)
/// - 4: E2M1(FP4)
static inline int32_t getMfmaF8F6F4MatrixFormat(Type t) {
  return llvm::TypeSwitch<Type, int32_t>(t)
      .Case<Float8E4M3FNType>([](Type) { return 0; })
      .Case<Float8E5M2Type>([](Type) { return 1; })
      .Case<Float6E3M2FNType>([](Type) { return 2; })
      .Case<Float6E2M3FNType>([](Type) { return 3; })
      .Case<Float4E2M1FNType>([](Type) { return 4; })
      .Default([](Type) { return -1; });
}

struct DotOpMFMAConversionHelper {
  AMDMfmaEncodingAttr mfmaLayout;

  ConversionPatternRewriter &rewriter;
  const LLVMTypeConverter *typeConverter;
  Location loc;
  MLIRContext *ctx{};

  virtual ~DotOpMFMAConversionHelper() = default;

  explicit DotOpMFMAConversionHelper(AMDMfmaEncodingAttr mfmaLayout,
                                     ConversionPatternRewriter &rewriter,
                                     const LLVMTypeConverter *typeConverter,
                                     Location loc)
      : mfmaLayout(mfmaLayout), rewriter(rewriter),
        typeConverter(typeConverter), loc(loc), ctx(mfmaLayout.getContext()) {}

  Value generateMFMAOp(StringRef intrinsicName, Value valA, Value valB,
                       Value valC, MmacLayout mmacLayout) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value zeroFlag = b.i32_val(0);
    auto resType = valC.getType();
    OperationState loweredOp(loc, intrinsicName);
    loweredOp.addTypes(resType);
    Value i32Flag = b.i32_val(0);
    Value i1Flag = b.false_val();
    bool isMmacLTS =
        mmacLayout == MmacLayout::TRANSPOSE ||
        mmacLayout == MmacLayout::INTERLEAVE_TRANSPOSE;
    bool isMmacLIT =
        mmacLayout == MmacLayout::INTERLEAVE ||
        mmacLayout == MmacLayout::INTERLEAVE_TRANSPOSE;
    Value ltsFlag = isMmacLTS ? b.true_val() : b.false_val();
    Value litFlag = isMmacLIT ? b.true_val() : b.false_val();
    if (intrinsicName.compare("rocdl.mmac.16x16x4.f32") == 0 ||
        intrinsicName.compare("rocdl.mmac.16x16x4.f64") == 0) {
      assert(!isMmacLIT && "16x16x4 FP32/FP64 MMAC does not support LIT");
      loweredOp.addOperands({valA, valB, valC, i32Flag, ltsFlag});
    } else if (intrinsicName.compare("rocdl.mmac.f32.16x16x16.bf16") == 0) {
      // LLVM expects packed BF16 MMAC inputs as <4 x i16>. Older layouts
      // already provide i16 vectors; bitcast only BF16 vectors preserved
      // until this intrinsic boundary.
      auto toI16Vec = [&](Value v) -> Value {
        auto vt = dyn_cast<VectorType>(v.getType());
        if (!vt || !vt.getElementType().isBF16())
          return v;
        return b.bitcast(v, vec_ty(i16_ty, vt.getNumElements()));
      };
      loweredOp.addOperands(
          {toI16Vec(valA), toI16Vec(valB), valC, litFlag, ltsFlag});
    } else if (intrinsicName.compare("rocdl.mmac.16x16x8.f32") == 0 ||
               intrinsicName.compare("rocdl.mmac.f32.16x16x16.f16") == 0 ||
               intrinsicName.compare("rocdl.mmac.16x16x16.f16") == 0 ||
               intrinsicName.compare("rocdl.mmac.f32.16x16x32.bf8.bf8") == 0 ||
               intrinsicName.compare("rocdl.mmac.f32.16x16x32.bf8.fp8") == 0 ||
               intrinsicName.compare("rocdl.mmac.f32.16x16x32.fp8.bf8") == 0 ||
               intrinsicName.compare("rocdl.mmac.f32.16x16x32.fp8.fp8") == 0) {
      loweredOp.addOperands({valA, valB, valC, litFlag, ltsFlag});
    } else if (intrinsicName.compare("rocdl.mmac.f32.16x16x8.tf32") == 0) {
      auto v2i32Ty = vec_ty(i32_ty, 2);
      Value valAi32 = b.bitcast(valA, v2i32Ty);
      Value valBi32 = b.bitcast(valB, v2i32Ty);
      loweredOp.addOperands({valAi32, valBi32, valC, litFlag, ltsFlag});
    } else if (intrinsicName.compare("rocdl.mmac.bf16.16x16x16.f16") == 0 ||
               intrinsicName.compare("rocdl.mmac.f16.16x16x16.bf16") == 0 ||
               intrinsicName.compare("rocdl.mmac.16x16x16.bf16") == 0) {
      auto toI16Vec = [&](Value v) -> Value {
        auto vt = dyn_cast<VectorType>(v.getType());
        if (!vt || !vt.getElementType().isBF16())
          return v;
        return b.bitcast(v, vec_ty(i16_ty, vt.getNumElements()));
      };
      Value ai = toI16Vec(valA), bi = toI16Vec(valB), ci = toI16Vec(valC);
      OperationState bf16Op(loc, intrinsicName);
      bf16Op.addTypes(ci.getType());
      bf16Op.addOperands({ai, bi, ci, litFlag, ltsFlag});
      Value raw = rewriter.create(bf16Op)->getResult(0);
      return raw.getType() == resType ? raw : b.bitcast(raw, resType);
    } else if (intrinsicName.compare("rocdl.mmac.i32.16x16x32.i8") == 0 ||
               intrinsicName.compare("rocdl.mmac.i32.16x16x32.u8") == 0) {
      loweredOp.addOperands({valA, valB, valC, litFlag, i1Flag, ltsFlag});
    } else {
      loweredOp.addOperands({valA, valB, valC});
    }

    return rewriter.create(loweredOp)->getResult(0);
  }

  int getNumSubmatrices(Type elementType, int mDim, int nDim) const {
    if (mDim == 4 || nDim == 4)
      return 16;
    return 1;
  }

  Value processSubBlocks(int numSubBlocks, Value acc, bool reduceSubBlocks,
                         bool zeroSubBlocks) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    assert((numSubBlocks & (numSubBlocks - 1)) == 0 &&
           "numSubBlocks in not pow 2!");
    if (numSubBlocks == 1)
      return acc;
    constexpr int warpSize = 64;
    int subBlockSize = warpSize / numSubBlocks;
    Value laneId = getLaneId(rewriter, loc);
    auto vecTy = dyn_cast<VectorType>(acc.getType());
    auto elemType = vecTy.getElementType();
    assert(elemType.getIntOrFloatBitWidth() == 32);
    int numScalars = vecTy.getNumElements();
    std::vector<Value> accScalar(numScalars);
    for (int i = 0; i < numScalars; ++i)
      accScalar[i] = b.extract_element(elemType, acc, b.i32_val(i));

    if (reduceSubBlocks) {
      while (subBlockSize < warpSize) {
        for (int i = 0; i < numScalars; ++i) {
          Value other_acc =
              shuffleXor(loc, rewriter, accScalar[i], subBlockSize);
          if (elemType.isInteger(32))
            accScalar[i] = b.add(accScalar[i], other_acc);
          else
            accScalar[i] = b.fadd(accScalar[i], other_acc);
        }
        subBlockSize *= 2;
      }
    }
    if (zeroSubBlocks) {
      Value zero;
      if (elemType.isInteger(32))
        zero = b.i32_val(0);
      else if (elemType.isF64())
        zero = b.f64_val(0.0);
      else
        zero = b.f32_val(0.0);
      auto cond = b.icmp_ult(laneId, b.i32_val(subBlockSize));
      for (int i = 0; i < numScalars; ++i)
        accScalar[i] = b.select(cond, accScalar[i], zero);
    }

    Value reducedAcc = b.undef(vecTy);
    for (int i = 0; i < numScalars; ++i)
      reducedAcc =
          b.insert_element(vecTy, reducedAcc, accScalar[i], b.i32_val(i));
    return reducedAcc;
  }

  /// @brief MFMA 4x4 is computes 16 matrix multiplications, this functions adds
  /// these 16 matrices to get final 4x4 matrix
  /// @param numSubBlocks
  /// @param acc
  /// @return
  Value reduceSubBlocks(int numSubBlocks, Value acc) const {
    return processSubBlocks(numSubBlocks, acc, true, false);
  }

  /// @brief Zeroes out redundant values in all sub-blocks except first one
  ///
  /// Every warp in mfma 4x4 layout holds only 4 unique values(scalar or
  /// vectors) in blocks of 4 consecutive threads, There are 16 copies of these
  /// 4 values across all threads of the warp. Need to zero out 15 copies to use
  /// accumulator between dot operations.
  /// @param numSubBlocks
  /// @param acc
  /// @return
  Value zeroAuxiliarBlocks(int numSubBlocks, Value acc) const {
    return processSubBlocks(numSubBlocks, acc, false, true);
  }

  /// Dot operand layout minimal tile is kDimInstrSize elements across
  /// K dimension. If dot operand K dimension is smaller, layout
  /// assigns tensor elements to multiple different hardware locations.
  /// In this case mfma instruction adds elements in accumulator
  /// multiple times.
  ///
  /// Let say A=[1,2]; B=[3,4], C = A*B = 1*3+2*4 = 11
  /// Consider instruction K size is 4,
  /// in this case operands will be duplicated:
  /// A' = [1,2,1,2] B' = [3,4,3,4]
  /// C' = (1*3+2*4) + (1*3+2*4) = 22
  ///
  /// Following code adjusts accumulator values in such cases. Only the dot
  /// contribution (acc - carriedInC) is duplicated; the carried-in
  /// accumulator itself must be preserved as-is. If the dot contribution
  /// is integer, shift it right by log2(duplicationRate); if float,
  /// multiply it with the 1/duplicationRate constant. Adding the
  /// contribution back to the carried-in accumulator last keeps earlier
  /// K-loop iterations from being diluted.
  void adjustAccForSmallKDim(SmallVector<Value> &fc, Value &acc, Type dstElemTy,
                             int b, int m, int n, int64_t numRepM,
                             int64_t numRepN, int64_t kDimInstrSize,
                             int64_t kDimOperandSize,
                             unsigned elemsPerVec) const {
    auto tb = TritonLLVMOpBuilder(loc, rewriter);
    for (unsigned v = 0; v < elemsPerVec; ++v) {
      Value accElem = tb.extract_element(dstElemTy, acc, tb.i32_val(v));
      auto linearIdx = b * numRepM * numRepN * elemsPerVec +
                       m * numRepN * elemsPerVec + n * elemsPerVec + v;
      if (kDimInstrSize > kDimOperandSize) {
        assert(kDimInstrSize % kDimOperandSize == 0);
        int duplicationRate = kDimInstrSize / kDimOperandSize;
        assert(llvm::isPowerOf2_32(duplicationRate));
        // Only the dot contribution (acc - c) is duplicated along K. The
        // carried-in accumulator c (still in fc[linearIdx]) must be kept
        // intact: rescaling the whole accumulator would dilute the partial
        // sums accumulated by previous K-loop iterations.
        if (dstElemTy.isInteger()) {
          Value dotAcc = tb.sub(accElem, fc[linearIdx]);
          auto shiftSize = llvm::Log2_32(duplicationRate);
          assert(!dotAcc.getType().isUnsignedInteger() &&
                 "MFMA uses signed accumulator");
          dotAcc = tb.ashr(dotAcc, tb.i32_val(shiftSize));
          accElem = tb.add(dotAcc, fc[linearIdx]);
        } else {
          Value dotAcc = LLVM::FSubOp::create(rewriter, loc, dstElemTy,
                                              accElem, fc[linearIdx]);
          auto multiplierAttr =
              rewriter.getFloatAttr(dstElemTy, 1.0 / duplicationRate);
          auto multiplierVal = LLVM::ConstantOp::create(
              rewriter, loc, dstElemTy, multiplierAttr);
          dotAcc = tb.fmul(dotAcc, multiplierVal);
          accElem = tb.fadd(dotAcc, fc[linearIdx]);
        }
      }
      fc[linearIdx] = accElem;
    }
  }

  template <typename T>
  void packAndReplaceResult(T &op, SmallVector<Value> &fc,
                            const FailureOr<MfmaIntrinsic> &maybeMfmaIntrinsic,
                            Type dstElemTy, Type elemtTy,
                            size_t mmaCount) const {
    Type structTy = LLVM::LLVMStructType::getLiteral(
        ctx, SmallVector<Type>(fc.size(), dstElemTy));
    Value res = packLLElements(loc, typeConverter, fc, rewriter, structTy);

    rewriter.replaceOp(op, res);
  }

  // Conduct the Dot conversion.
  LogicalResult convertDot(DotOp op, DotOpAdaptor adaptor) const {
    auto tb = TritonLLVMOpBuilder(loc, rewriter);
    // Check if this dot has come with priority set by setprio.
    auto setPrioOp = dyn_cast_or_null<ROCDL::SetPrioOp>(op->getPrevNode());

    auto warpsPerCTA = mfmaLayout.getWarpsPerCTA();
    auto mnkDim = mfmaLayout.getInstrShape();
    auto mDim = mnkDim[0];
    auto nDim = mnkDim[1];
    auto kDim = mnkDim[2];
    auto mfmaVersion = mfmaLayout.getVersion();

    Value a = op.getA();
    Value b = op.getB();
    Value d = op.getD();
    auto aTensorTy = cast<RankedTensorType>(a.getType());
    auto bTensorTy = cast<RankedTensorType>(b.getType());
    auto dTensorTy = cast<RankedTensorType>(d.getType());
    auto elemTyA = aTensorTy.getElementType();
    auto elemTyB = bTensorTy.getElementType();

    const auto kDimOperandSize = aTensorTy.getShape().back();

    bool allowXF32 =
        op.getInputPrecision() == InputPrecision::TF32 && mfmaVersion == 3;
    StringRef intrinsicName;
    auto arch = getAMDArch(op->getParentOfType<ModuleOp>());
    assert(arch.has_value() && "expected arch");
    auto intrinsicFeatures = mlir::triton::HCU::deduceHCUISAFeature(*arch);
    Type elemTyD = dTensorTy.getElementType();
    FailureOr<MfmaIntrinsic> maybeMfmaIntrinsic = MfmaIntrinsic::selectFor(
        op.getLoc(), mfmaVersion, mDim, nDim, kDim, elemTyA, elemTyB, elemTyD,
        /*withScale=*/false, allowXF32,
        intrinsicFeatures);
    if (failed(maybeMfmaIntrinsic))
      return op.emitError(
          "no matching matrix core intrinsic due to unsupported element type");

    unsigned kBase = maybeMfmaIntrinsic->kBase;

    auto aEncoding = cast<DotOperandEncodingAttr>(aTensorTy.getEncoding());
    auto bEncoding = cast<DotOperandEncodingAttr>(bTensorTy.getEncoding());
    int kWidth = aEncoding.getKWidth();

    intrinsicName = maybeMfmaIntrinsic->name;
    // Do not discard modifier bits that the selected intrinsic cannot express,
    // including when an encoded dot bypasses the acceleration pass.
    if (failed(mlir::triton::HCU::validateMmacLayout(
            op, mfmaLayout.getMmacLayout(), *maybeMfmaIntrinsic)))
      return failure();

    // Upstream doubles kWidth here for the XF32 instruction (16x16x32: 2x K
    // and 2x per-lane fragment of F32). HCU f32 MMAC (mmac_f32_16x16x8tf32/
    // 8f32) is K=8, kBase=2, so the doubling does not apply:
    // if (aTensorTy.getElementType().isF32() && allowXF32)
    //   kWidth *= 2;

    const auto kDimInstrSize = mfmaLayout.getInstrShapeForOperand(kWidth, 0)[1];

    auto repA = mfmaLayout.getRepForOperand(aTensorTy.getShape(), kWidth, 0);
    auto repB = mfmaLayout.getRepForOperand(bTensorTy.getShape(), kWidth, 1);

    assert(repA[2] == repB[1]);

    Value loadedA = adaptor.getA();
    Value loadedB = adaptor.getB();
    Value loadedC = adaptor.getC();

    auto numRepM = repA[1];
    auto numRepN = repB[2];
    auto numRepK = repA[2];
    auto numRepB = repA[0];
    assert(repA[0] == repB[0]);

    auto mmacLayout = mfmaLayout.getMmacLayout();
    bool isHCUMmac = mfmaLayout.isHCUMmac();
    int numBroadcastA = 1;
    int numBroadcastB = 1;
    int numRepKA = numRepK;
    int numRepKB = numRepK;

    numRepK = std::max(numRepKA, numRepKB);
    int numBroadcast = std::max(numBroadcastA, numBroadcastB);

    bool preserveBF16 = intrinsicName.contains(".bf16") && mfmaVersion >= 4;
    auto operandA = getValuesFromDotOperandLayoutStruct(
        loadedA, numRepB, numRepM, numRepKA, kWidth, kBase,
        aTensorTy.getElementType(), allowXF32, preserveBF16, false,
        isHCUMmac);
    auto operandB = getValuesFromDotOperandLayoutStruct(
        loadedB, numRepB, numRepN, numRepKB, kWidth, kBase,
        aTensorTy.getElementType(), allowXF32, preserveBF16, false,
        isHCUMmac);

    int warpSize = triton::gpu::lookupThreadsPerWarp(rewriter);
    int elemsPerVec = mDim * nDim / warpSize;
    int numVecInKBase = numRepK * kWidth / kBase;

    auto dstElemTy = dTensorTy.getElementType();
    auto fc = unpackLLElements(loc, loadedC, rewriter);
    SmallVector<int64_t> fcStrides =
        computeStrides({numRepB, numRepM, numRepN, elemsPerVec});

    // Optional fence: keep preceding local_load (ds_read*) before any mmac.
    if (getSchedBarrierBeforeMmacAttr(op))
      ROCDL::SchedBarrier::create(rewriter, loc, /*mask=*/0);

    Value firstMfma;
    auto vecTy = vec_ty(dstElemTy, elemsPerVec);
    for (int b = 0; b < numRepB; ++b) {
      for (int m = 0; m < numRepM; ++m) {
        for (int n = 0; n < numRepN; ++n) {
          Value acc = tb.undef(vecTy);

          for (int v = 0; v < elemsPerVec; ++v) {
            int linearIdx = linearize({b, m, n, v}, fcStrides);
            Value c = fc[linearIdx];
            acc = tb.insert_element(vecTy, acc, c, tb.i32_val(v));
          }

          for (int k = 0; k < numVecInKBase; ++k) {
            Value op1 = operandA[{b, m, k}];
            Value op2 = operandB[{b, n, k}];

            if (mfmaLayout.getIsTransposed())
              std::swap(op1, op2);

            acc = generateMFMAOp(intrinsicName, op1, op2, acc, mmacLayout);

            if (!firstMfma)
              firstMfma = acc;
          }

          adjustAccForSmallKDim(fc, acc, dstElemTy, b, m, n, numRepM, numRepN,
                                kDimInstrSize, kDimOperandSize, elemsPerVec);
        }
      }
    }

    // Optional fence: keep empty abarrier.arrive after the mmac cluster when
    // empty_arrive_after_mmac places arrive after tt.dot.
    if (getEmptyArriveAfterMmacAttr(op))
      ROCDL::SchedBarrier::create(rewriter, loc, /*mask=*/0);

    // Originally, setprio (high) is set to the high-level dot op. After dot is
    // being lowered to the series of mfma operations, it should be moved next
    // to the first mfma leaving the first mfma staying at the low priority. In
    // this way, incoming warp can be effectively waiting on the first mfma
    // instruction (low priority) while the other warp is executing mfma with
    // high priority. Otherwise, incoming warp can break the cluster.
    if (setPrioOp && firstMfma)
      setPrioOp->moveAfter(firstMfma.getDefiningOp());

    const size_t mmaCount = numRepB * numRepM * numRepN * numVecInKBase;
    packAndReplaceResult(op, fc, maybeMfmaIntrinsic, dstElemTy, elemTyA,
                         mmaCount);

    return success();
  }

  /// Adapt packed FP4 fragments to the HCU F8F6F4 register ABI.
  /// Its two VGPRs hold one E2M1 code per byte, in bits [5:2], matching MLS
  /// B4Mix padding. Already-padded operands and packed FP4-only instructions
  /// need no adaptation. Both scaled and unscaled MMAC use this byte layout.
  void prepareMmacOperands(ValueTable &operands, StringRef intrinsicName,
                           bool isPackedFp4) const {
    if (!isPackedFp4 || !mfmaLayout.isHCUMmac() ||
        !intrinsicName.ends_with(".f8f6f4"))
      return;

    constexpr unsigned vgprCount = 2;
    constexpr unsigned bytesPerVgpr = 4;
    constexpr unsigned fp4Bits = 4;
    constexpr unsigned fp4ByteOffset = 2;
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto resultTy = vec_ty(i32_ty, vgprCount);
    for (auto &entry : operands) {
      Value packed = entry.second;
      assert(packed.getType().isInteger(32) &&
             "K32 packed FP4 requires one i32 per lane");
      Value words = b.undef(resultTy);
      for (unsigned word = 0; word < vgprCount; ++word) {
        Value bytes = b.i32_val(0);
        for (unsigned byte = 0; byte < bytesPerVgpr; ++byte) {
          unsigned code = word * bytesPerVgpr + byte;
          Value nibble = b.and_(
              b.lshr(packed, b.i32_val(code * fp4Bits)),
              b.i32_val((1u << fp4Bits) - 1));
          bytes = b.or_(
              bytes, b.shl(nibble, b.i32_val(byte * 8 + fp4ByteOffset)));
        }
        words = b.insert_element(resultTy, words, bytes, b.i32_val(word));
      }
      entry.second = words;
    }
  }

  /// Process the elements in rawElems and prepare a vector for mfma input.
  /// rawElems is a vector of kBase elements. Each element is of the raw
  /// element type from the input. We need to prepare a vector of kBase
  /// elements of appropriate element type required by mfma instructions.
  /// FIXME: isHCUMmac is a special handling for HCUs and should be eliminated
  ///        once HCU backend is implemented in Triton.
  Value prepareOperands(Value rawElems, int kBase, Type type, bool preserveBF16, bool isHCUMmac,
    bool isConstantScale = false) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value results;

    // Construct a vector type of kBase elements with desired type
    auto vecTy = vec_ty(type, kBase);
    auto pack8BitToI32 = [&](Value vec, int m) -> Value {
      // Pack m 8bit elements to n I32, e.g. 4xi8 -> 1xi32, 8xi8 -> 2xi32
      assert(m % 4 == 0 && "m should be multiple of 4 for packing");
      auto v4x8BitTy = vec_ty(type, 4);
      auto n = std::ceil(m / 4.0);
      auto vNxI32Ty = vec_ty(i32_ty, n);
      Value vNxI32 = b.undef(vNxI32Ty);
      for (int i = 0; i < n; ++i) {
        Value v4x8Bit = b.undef(v4x8BitTy);
        for (int elemId = 0; elemId < 4; ++elemId) {
          auto val = b.extract_element(type, vec, b.i32_val(elemId + i * 4));
          v4x8Bit = b.insert_element(v4x8BitTy, v4x8Bit, val, b.i32_val(elemId));
        }
        Value toI32 = b.bitcast(v4x8Bit, i32_ty);
        vNxI32 = b.insert_element(vNxI32Ty, vNxI32, toI32, b.i32_val(i));
      }
      return vNxI32;
    };
    if (type.isBF16() && !preserveBF16)
      vecTy = vec_ty(i16_ty, kBase);
    Value vec = b.undef(vecTy);

    // For each element in rawElems, extract the element as the desired type,
    // bitcast it if needed, and insert it into vec.
    for (int elemId = 0; elemId < kBase; ++elemId) {
      auto val = b.extract_element(type, rawElems, b.i32_val(elemId));
      if (type.isBF16() && !preserveBF16) {
        // rocdl.mfma.f32.32x32x8bf16.1k calls for input of i16 type
        auto cast = b.bitcast(val, i16_ty);
        vec = b.insert_element(vecTy, vec, cast, b.i32_val(elemId));
      } else {
        vec = b.insert_element(vecTy, vec, val, b.i32_val(elemId));
      }
    }

    // Now we have a vector of kBase elements of desired type.
    // Then we need to prepare vec for results.
    if (type.getIntOrFloatBitWidth() == 8) {
      if (1 == kBase)
        // This is only for the scale operands of scaled mfma on CDNA4
          results = b.zext(i32_ty, b.bitcast(vec, i8_ty));
      if (2 == kBase)
        // This case can occur during scale tensor packing when there aren't
        // enough elements to fill all 4 opSel slots. For example, with an A
        // tensor of size 16x256 and using 16x16x128 block sizes, we end up with
        // only 2 elements to pack,  resulting in a kBase of 2.
        results = b.zext(i32_ty, b.bitcast(vec, i16_ty));
      if (4 == kBase)
        // This is for int8 on pre- CDNA3 GPUs and scale tensors on CDNA4 GPUs
        results = b.bitcast(vec, i32_ty);
      if (8 == kBase)
        results = isHCUMmac ? pack8BitToI32(vec, 8) : b.bitcast(vec, i64_ty);
      if (16 == kBase)
        // This is only for the operands of scaled mfma on CDNA4
        results = b.bitcast(vec, vec_ty(i32_ty, 4));
      if (32 == kBase)
        results = b.bitcast(vec, vec_ty(i32_ty, 8));
    } else if (type.getIntOrFloatBitWidth() == 32 && kBase == 1) {
      if (vec.getType() == i32_ty)
        results = vec;
      else
        results = b.bitcast(vec, i32_ty);
    } else {
      results = vec;
    }
    return results;
  }

  /// Converts dot operand structure to value table and converts types
  /// appropriate for mfma instructions
  virtual ValueTable getValuesFromDotOperandLayoutStruct(
      Value value, int batch, int nonKRep, int kRepInKWidth, int kWidth,
      int kBase, Type type, bool allowXF32, bool preserveBF16,
      bool isConstantScale = false, bool isHCUMmac = false) const {
    auto tb = TritonLLVMOpBuilder(loc, rewriter);
    auto elems = unpackLLElements(loc, value, rewriter);
    // number of kBase-element vectors
    int numVecInKBase = kRepInKWidth * kWidth / kBase;
    if (numVecInKBase == 0) {
      numVecInKBase = 1;
      nonKRep /= kBase / (kRepInKWidth * kWidth);
      assert(nonKRep > 0 && "nonKrep too small");
    }
    ValueTable dotOpVals;

    SmallVector<int64_t> strides =
        computeStrides({batch, nonKRep, numVecInKBase, kBase});
    for (int b = 0; b < batch; ++b) {
      for (int nonK = 0; nonK < nonKRep; nonK++) {
        for (int kBaseVec = 0; kBaseVec < numVecInKBase; kBaseVec++) {
          // For each kBase-element vector

          // Step 1: construct each kBase-element vector by
          //         - extracting kBase elements from elems and
          //         - putting them into a kBase-element vector, i.e. rawElems
          Type elemTy = typeConverter->convertType(type);
          Type ty = vec_ty(elemTy, kBase);
          Value rawElems = tb.undef(ty);
          for (int k = 0; k < kBase; ++k) {
            auto index = linearize({b, nonK, kBaseVec, k}, strides);
            rawElems =
                tb.insert_element(ty, rawElems, elems[index], tb.i32_val(k));
          }

          // Step 2: process rawElems based on element type
          // Note that for f32/fp64 input and XF32 is not allowed, nothing needs
          // to be done and rawElems is inserted into the ValueTable directly
          if ((type.isF32() || type.isF64()) && !allowXF32) {
          #if 0
            dotOpVals[{b, nonK, kBaseVec}] =
                tb.extract_element(type, rawElems, tb.i32_val(0));
          #endif
            /* HCU support: add mmac_f32_m16n16k8_f32 support */
            if (kBase == 1) {
              // Scalar-operand MMACs (mmac_f32_16x16x4f32 /
              // mmac_f64_16x16x4f64) consume one element per lane per
              // instruction. kPack only widens kWidth, not kBase, so the
              // scalar case must be keyed on kBase, not on kWidth.
              dotOpVals[{b, nonK, kBaseVec}] =
                tb.extract_element(type, rawElems, tb.i32_val(0));
            } else {
              // mmac_f32_m16n16k8_f32 support
              dotOpVals[{b, nonK, kBaseVec}] =
                prepareOperands(rawElems, kBase, type, preserveBF16, isHCUMmac);
            }
          } else {
            Value vals;
            if (type.isF32() && allowXF32) {
              vals = prepareOperands(rawElems, kBase, f32_ty, preserveBF16, isHCUMmac);
            } else if (type.getIntOrFloatBitWidth() == 8) {
              vals = prepareOperands(rawElems, kBase, i8_ty, preserveBF16, isHCUMmac,
                                     isConstantScale);
            } else if (type.getIntOrFloatBitWidth() == 32) {
              vals = prepareOperands(rawElems, kBase, i32_ty, preserveBF16, isHCUMmac,
                                     isConstantScale);
            } else if (type.isBF16()) {
              vals = prepareOperands(rawElems, kBase, bf16_ty, preserveBF16, isHCUMmac);
            } else {
              assert(type.isF16() && "Unsupported data type");
              vals = prepareOperands(rawElems, kBase, f16_ty, preserveBF16, isHCUMmac);
            }

            // Step 3: Insert the processed vals into the ValueTable
            dotOpVals[{b, nonK, kBaseVec}] = vals;
          }
        }
      }
    }
    return dotOpVals;
  }
};

struct ScaledDotOpMFMAConversionHelper : DotOpMFMAConversionHelper {
  // All addresses and offsets below are in Scale Buffer rows, not bytes.
  // waveBase is the allocated base plus this wave's starting-row displacement;
  // rowOffset is the compile-time displacement for one MMAC's scale operand.
  //
  //   physicalRow = waveBase + rowOffset = m0Base + offset
  //   m0Base      = waveBase + (rowOffset / 32) * 32
  //   offset      = rowOffset % 32  // 5-bit instruction immediate
  //
  // m0 packs the A/B m0Base values; each operand has its own immediate offset.
  struct ScaleAddress {
    Value waveBase;
    int64_t rowOffset;
  };
  virtual ~ScaledDotOpMFMAConversionHelper() = default;

  ScaledDotOpMFMAConversionHelper(AMDMfmaEncodingAttr mfmaLayout,
                                  ConversionPatternRewriter &rewriter,
                                  const LLVMTypeConverter *typeConverter,
                                  Location loc)
      : DotOpMFMAConversionHelper(mfmaLayout, rewriter, typeConverter, loc) {}

  /// Tensor shapes remain in logic space. This helper derives the expanded
  /// elem-space shape used by MFMA/MMAC operand lowering.
  template <typename ScaledDotOp>
  SmallVector<int64_t> getScaledDotOpElemsShape(ScaledDotOp op, int opIdx) const {
    auto tensorTy = cast<RankedTensorType>(
        op->getOperand(opIdx).getType());
    SmallVector<int64_t> shape(tensorTy.getShape().begin(),
                               tensorTy.getShape().end());
    if (opIdx == 0) {
      auto elemType = op.getAElemType();
      if (elemType == ScaleDotElemType::E2M1) {
        int rank = shape.size();
        if (op.getLhsKPack())
          shape[rank - 1] *= 2; // K-pack: unpack doubles K
        else
          shape[rank - 2] *= 2; // M-pack: unpack doubles M
      }
    } else {
      assert(opIdx == 1 && "opIdx must be 0 or 1");
      auto elemType = op.getBElemType();
      if (elemType == ScaleDotElemType::E2M1) {
        int rank = shape.size();
        if (op.getRhsKPack())
          shape[rank - 2] *= 2; // K-pack: unpack doubles K
        else
          shape[rank - 1] *= 2; // N-pack: unpack doubles N
      }
    }
    return shape;
  }

  Value generateScaledMFMAOp(StringRef intrinsicName, Value valA, Value valB,
                             Value valC, Type elemTypeA, Type elemTypeB,
                             ArrayRef<ScaleAddress> scales) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto resType = valC.getType();
    auto mmacLayout = mfmaLayout.getMmacLayout();
    bool isLTS = mmacLayout == MmacLayout::TRANSPOSE ||
                 mmacLayout == MmacLayout::INTERLEAVE_TRANSPOSE;
    bool isLIT = mmacLayout == MmacLayout::INTERLEAVE ||
                 mmacLayout == MmacLayout::INTERLEAVE_TRANSPOSE;

    SmallVector<Value> args{valA, valB, valC};
    if (intrinsicName.ends_with(".f8f6f4")) {
      int aFormat = getMfmaF8F6F4MatrixFormat(elemTypeA);
      int bFormat = getMfmaF8F6F4MatrixFormat(elemTypeB);
      assert(aFormat >= 0 && bFormat >= 0);
      int realABTypeID = 5 * aFormat + bFormat;
      args.push_back(b.i32_val(realABTypeID));
    }
    args.append({b.i1_val(isLIT), b.i1_val(isLTS)});

    if (scales.empty()) {
      // Keep the selected geometry and format, but use the unscaled ABI.
      if (intrinsicName ==
          ROCDL::mmac_scale_f32_16x16x64_fp4::getOperationName())
        intrinsicName = ROCDL::mmac_f32_16x16x64_fp4::getOperationName();
      else if (intrinsicName ==
               ROCDL::mmac_scale_f32_16x16x32_f8f6f4::getOperationName())
        intrinsicName = ROCDL::mmac_f32_16x16x32_f8f6f4::getOperationName();
      OperationState loweredOp(loc, intrinsicName);
      loweredOp.addTypes(resType);
      loweredOp.addOperands(args);
      return rewriter.create(loweredOp)->getResult(0);
    }

    assert(scales.size() == 2 && "scaled MMAC requires A/B scale addresses");
    const auto &scaleA = scales[0], &scaleB = scales[1];
    // Resolve instruction windows only after selecting register fragments.
    auto baseA = b.add(scaleA.waveBase, b.i32_val((scaleA.rowOffset / 32) * 32));
    auto baseB = b.add(scaleB.waveBase, b.i32_val((scaleB.rowOffset / 32) * 32));
    Value m0 = b.or_(baseA, b.shl(baseB, b.i32_val(16)));
    args.append(
        {b.i32_val(scaleA.rowOffset % 32), b.i32_val(scaleB.rowOffset % 32)});
    std::string name =
        "llvm.hcu." +
        intrinsicName.drop_front(StringRef("rocdl.").size()).str();
    auto call =
        LLVM::createLLVMIntrinsicCallOp(rewriter, loc, name, {resType}, args);
    // Standard operand bundles carry the SSA dependency without changing the
    // intrinsic ABI. LLVM export only materializes the named-register write.
    call.getOpBundleOperandsMutable().join().append(ValueRange{m0});
    call.setOpBundleSizesAttr(rewriter.getDenseI32ArrayAttr({1}));
    call.setOpBundleTagsAttr(
        rewriter.getArrayAttr({rewriter.getStringAttr("amdg.m0")}));
    // Export finishes the whole function before replacing bundled calls, so
    // MLIR's write-once value mapping never needs to be overwritten.
    call->getParentOfType<LLVM::LLVMFuncOp>()->setAttr("amdg.m0",
                                                       rewriter.getUnitAttr());
    return call.getResult(0);
  }

  LogicalResult convertScaledDot(triton::DotScaledOp op,
                                 triton::DotScaledOp::Adaptor adaptor) const {
    return convertMatrix(
        op, adaptor, /*withScale=*/false, [] {},
        [](int, int, int) { return SmallVector<ScaleAddress, 2>{}; });
  }

  LogicalResult
  convertScaledDot(triton::amdgpu::DotScaledOp op,
                   triton::amdgpu::DotScaledOp::Adaptor adaptor) const {
    auto tb = TritonLLVMOpBuilder(loc, rewriter);
    auto warpsPerCTA = mfmaLayout.getWarpsPerCTA();
    int64_t kDim = mfmaLayout.getInstrShape()[2];
    Value aScale = op.getAScale();
    Value bScale = op.getBScale();
    // Late planning has proved that each consumer row is a wave base plus a
    // static offset. Only materialize the two bases, not a table of dynamic
    // rows for every batch/replica/K atom whose entries are never consumed.
    std::array<Value, 2> scaleWaveBases;
    std::array<ScaleMemoryPlan, 2> scalePlans;
    int64_t scaleColumns = 0;
    auto tpw = mfmaLayout.getTilesPerWarp();
    auto prepareScales = [&] {
      scaleColumns =
          cast<triton::gpu::MemDescType>(aScale.getType()).getShape()[1];
      Value warp = getUniformWarpId(op, rewriter, loc);
      auto buildBase = [&](Value scale, Value base, unsigned idx) {
        auto type = cast<triton::gpu::MemDescType>(scale.getType());
        auto plan = ScaleMemoryPlan::fromMemDesc(type);
        scalePlans[idx] = plan;
        Value waveCoordinate =
            idx == 0 ? Value(tb.udiv(warp, tb.i32_val(warpsPerCTA[1])))
                     : Value(tb.urem(warp, tb.i32_val(warpsPerCTA[1])));
        // Shape reduction broadcasts surplus wave bits for small tiles.
        int64_t activeWaves = std::min<int64_t>(
            warpsPerCTA[idx],
            llvm::divideCeil(type.getShape()[0], 16 * tpw[idx]));
        waveCoordinate = tb.urem(waveCoordinate, tb.i32_val(activeWaves));
        Value firstGroup = tb.mul(waveCoordinate, tb.i32_val(tpw[idx]));
        auto instTilesAlongS = plan.getInstsAlongS(scaleColumns);
        Value instTileId = tb.mul(tb.udiv(firstGroup, tb.i32_val(plan.g)),
                                  tb.i32_val(instTilesAlongS));
        Value inner = tb.mul(tb.urem(firstGroup, tb.i32_val(plan.g)),
                             tb.i32_val(plan.instCols));
        return tb.add(
            base,
            tb.add(tb.mul(instTileId, tb.i32_val(plan.rowStride)), inner));
      };
      scaleWaveBases = {buildBase(aScale, adaptor.getAScale(), 0),
                        buildBase(bScale, adaptor.getBScale(), 1)};
    };
    auto scaleAddress = [&](unsigned idx, int mn, int k) {
      int64_t nonK = 16 * (mn % tpw[idx]) +
                     16 * tpw[idx] * warpsPerCTA[idx] * (mn / tpw[idx]);
      return ScaleAddress{
          scaleWaveBases[idx],
          scalePlans[idx].getRow(nonK, k * kDim / 32, scaleColumns)};
    };

    return convertMatrix(op, adaptor, /*withScale=*/true, prepareScales,
                         [&](int m, int n, int k) {
                           return SmallVector<ScaleAddress, 2>{
                               scaleAddress(0, m, k), scaleAddress(1, n, k)};
                         });
  }

  template <typename ScaledDotOp, typename Adaptor>
  LogicalResult
  convertMatrix(ScaledDotOp op, Adaptor adaptor, bool withScale,
                llvm::function_ref<void()> prepareScales,
                llvm::function_ref<SmallVector<ScaleAddress, 2>(int, int, int)>
                    scaleAddresses) const {
    auto tb = TritonLLVMOpBuilder(loc, rewriter);
    // Check if this dot has come with priority set by setprio.
    auto setPrioOp = dyn_cast_or_null<ROCDL::SetPrioOp>(op->getPrevNode());

    auto mnkDim = mfmaLayout.getInstrShape();
    auto mDim = mnkDim[0];
    auto nDim = mnkDim[1];
    auto mfmaVersion = mfmaLayout.getVersion();

    Value a = op.getA();
    Value b = op.getB();
    Value d = op.getD();
    auto aTensorTy = cast<RankedTensorType>(a.getType());
    auto bTensorTy = cast<RankedTensorType>(b.getType());
    auto dTensorTy = cast<RankedTensorType>(d.getType());
    auto elemTyA = aTensorTy.getElementType();
    auto elemTyB = bTensorTy.getElementType();
    auto elemTyD = dTensorTy.getElementType();
    ScaleDotElemType aElemType = op.getAElemType();
    ScaleDotElemType bElemType = op.getBElemType();
    auto aEncoding = cast<DotOperandEncodingAttr>(aTensorTy.getEncoding());
    auto bEncoding = cast<DotOperandEncodingAttr>(bTensorTy.getEncoding());

    auto supportsTypes = [](ScaleDotElemType elemType) {
      return elemType == ScaleDotElemType::E2M1 ||
             elemType == ScaleDotElemType::E4M3 ||
             elemType == ScaleDotElemType::E5M2;
    };

    if (!supportsTypes(aElemType) || !supportsTypes(bElemType)) {
      llvm::report_fatal_error("NYI: mxfp6\n");
    }

    bool isMlsA = aEncoding.getMlsScaledExt();
    bool isMlsB = bEncoding.getMlsScaledExt();
    auto aShapeE = getScaledDotOpElemsShape(op, 0);
    auto bShapeE = getScaledDotOpElemsShape(op, 1);
    auto aMlsElemBitTyKind = getMlsElemBitTyKind(aEncoding.getMlsScaledExt());
    auto bMlsElemBitTyKind = getMlsElemBitTyKind(bEncoding.getMlsScaledExt());
    // At a dot operand, ordinary FP4 loads still contain two K-packed
    // values per byte. MLS may keep that representation (B4) or already
    // pad each code into its own byte (B4Mix/F4). Resolve that per operand
    // instead of inferring packing from a particular kWidth or from the
    // presence of MLS on the other operand.
    auto getDotOperandFlow = [](ScaleDotElemType elemType,
                                MlsElemBitTyKind mlsKind) {
      if (mlsKind == MlsElemBitTyKind::None &&
          elemType == ScaleDotElemType::E2M1)
        mlsKind = MlsElemBitTyKind::B4;
      return getMlsDataFlowInfo(mlsKind);
    };
    auto aFlow = getDotOperandFlow(aElemType, aMlsElemBitTyKind);
    auto bFlow = getDotOperandFlow(bElemType, bMlsElemBitTyKind);
    bool isAElemB4Packed = aFlow.usesPackedShape(MlsDataFlowView::DotOperand);
    bool isBElemB4Packed = bFlow.usesPackedShape(MlsDataFlowView::DotOperand);

    int aKWidth = aEncoding.getKWidth();
    int bKWidth = bEncoding.getKWidth();
    unsigned aLogicalKWidth = aFlow.getMfmaKWidth(aKWidth);
    unsigned bLogicalKWidth = bFlow.getMfmaKWidth(bKWidth);
    auto kTileForDotOperand = [&](unsigned opIdx,
                                 unsigned logicalKWidth) -> int64_t {
      auto shape = mfmaLayout.getInstrShapeForOperand(logicalKWidth, opIdx);
      return shape[opIdx == 0 ? 1 : 0];
    };
    int64_t kDimOperandSizeA = kTileForDotOperand(0, aLogicalKWidth);
    int64_t kDimOperandSizeB = kTileForDotOperand(1, bLogicalKWidth);
    assert(kDimOperandSizeA == kDimOperandSizeB &&
           "A/B dot operand K tile sizes must match");
    int64_t kDimOperandSize = kDimOperandSizeA;

    auto ctx = op.getContext();
    constexpr bool allowXF32 = false;

    auto arch = getAMDArch(op->template getParentOfType<ModuleOp>());
    assert(arch.has_value() && "expected arch");
    auto features = withScale
        ? mlir::triton::HCU::deduceHCUISAFeature(*arch) | HCUISAFeature::MMAC_WITH_SCALE
        : mlir::triton::HCU::deduceHCUISAFeature(*arch);
    // Acceleration has already selected the instruction and its layout.
    // Re-selecting from packed register widths can silently change K32 to K64.
    FailureOr<MfmaIntrinsic> maybeMfmaIntrinsic = MfmaIntrinsic::get(
        op.getLoc(), mfmaVersion, mDim, nDim, mnkDim[2],
        scaleDotElemTypeToMLIRType(ctx, aElemType),
        scaleDotElemTypeToMLIRType(ctx, bElemType), elemTyD,
        /*withScale=*/true, allowXF32, features);
    if (failed(maybeMfmaIntrinsic))
      return op.emitError(
          "no matching matrix core intrinsic due to unsupported element type");

    if (failed(mlir::triton::HCU::validateMmacLayout(
            op, mfmaLayout.getMmacLayout(), *maybeMfmaIntrinsic)))
      return failure();

    StringRef intrinsicName = maybeMfmaIntrinsic->name;
    unsigned kBase = maybeMfmaIntrinsic->kBase;
    auto isHCUMmac = mfmaLayout.isHCUMmac();
    unsigned aKBase = isAElemB4Packed ? kBase / 2 : kBase;
    unsigned bKBase = isBElemB4Packed ? kBase / 2 : kBase;

    const auto kDimInstrSize =
        mfmaLayout.getInstrShapeForOperand(aLogicalKWidth, 0)[1];
    auto repA = isMlsA
        ? mfmaLayout.getRepForScaledOperand(aShapeE, aKWidth,
                                            aMlsElemBitTyKind, 0)
        : mfmaLayout.getRepForOperand(aTensorTy.getShape(), aKWidth, 0);
    auto repB = isMlsB
        ? mfmaLayout.getRepForScaledOperand(bShapeE, bKWidth,
                                            bMlsElemBitTyKind, 1)
        : mfmaLayout.getRepForOperand(bTensorTy.getShape(), bKWidth, 1);
    assert(repA[2] == repB[1]);

    Value loadedA = adaptor.getA();
    Value loadedB = adaptor.getB();
    Value loadedC = adaptor.getC();

    auto numRepM = repA[1];
    auto numRepN = repB[2];
    auto numRepK = repA[2];
    auto numRepB = repA[0];
    assert(repA[0] == repB[0]);

    auto operandA = getValuesFromDotOperandLayoutStruct(
        loadedA, numRepB, numRepM, numRepK, aKWidth, aKBase,
        aTensorTy.getElementType(), allowXF32, /*preserveBF16=*/false,
        /*isConstantScale=*/false, isHCUMmac);
    auto operandB = getValuesFromDotOperandLayoutStruct(
        loadedB, numRepB, numRepN, numRepK, bKWidth, bKBase,
        bTensorTy.getElementType(), allowXF32, /*preserveBF16=*/false,
        /*isConstantScale=*/false, isHCUMmac);

    prepareMmacOperands(operandA, intrinsicName, isAElemB4Packed);
    prepareMmacOperands(operandB, intrinsicName, isBElemB4Packed);

    prepareScales();

    auto dstElemTy = dTensorTy.getElementType();
    auto fc = unpackLLElements(loc, loadedC, rewriter);

    unsigned warpSize = triton::gpu::lookupThreadsPerWarp(rewriter);
    // compute number of output elements that each thread holds for one MFMA
    // instruction. subBlocks
    const int subBlocks = 1;
    auto elemsPerVec = mDim * nDim * subBlocks / warpSize;
    int numVecInKBase = numRepK * aKWidth / aKBase;

    Value firstMfma;
    auto vecTy = vec_ty(dstElemTy, elemsPerVec);

    // 2-step pingpong got local_loads + dot_scaled in the dot cluster
    // from the first step in the transform pingpong pass.
    // Here, in the second step, it splits operations into two clusters
    // The first cluster has local_load with mfma from the first half of K
    // and the second cluster with the other half K of mfma.
    // By splitting in K dim, we can retire registers used by the
    // first half of mfma, backend compiler is supposed to schedule it.
    int halfPoint = numVecInKBase * numRepB * numRepM * numRepN / 2;
    int currIter = 0;
    bool is2Step = op->hasAttr("pingpong_2step");
    int outerKBound = is2Step ? numVecInKBase : 1;
    int innerKBound = is2Step ? 1 : numVecInKBase;

    if (getSchedBarrierBeforeMmacAttr(op))
      ROCDL::SchedBarrier::create(rewriter, loc, /*mask=*/0);

    for (int outerK = 0; outerK < outerKBound; ++outerK) {
      for (int b = 0; b < numRepB; ++b) {
        for (int m = 0; m < numRepM; ++m) {
          for (int n = 0; n < numRepN; ++n) {
            // Insert pingpong cluster barrier when needed.
            if (is2Step && currIter++ == halfPoint) {
              ROCDL::SchedBarrier::create(rewriter, loc, 0);
              ROCDL::SBarrierOp::create(rewriter, loc);
              ROCDL::SchedBarrier::create(rewriter, loc, 0);
            }
            Value acc = tb.undef(vecTy);
            for (unsigned v = 0; v < elemsPerVec; ++v) {
              acc = tb.insert_element(
                  vecTy, acc,
                  fc[b * numRepM * numRepN * elemsPerVec +
                    m * numRepN * elemsPerVec + n * elemsPerVec + v],
                  tb.i32_val(v));
            }
            acc = zeroAuxiliarBlocks(subBlocks, acc);
            for (int innerK = 0; innerK < innerKBound; ++innerK) {
              int k = is2Step ? outerK : innerK;
              Value valA = operandA[{b, m, k}];
              Value valB = operandB[{b, n, k}];
              Type typeA = maybeMfmaIntrinsic->aElementType;
              Type typeB = maybeMfmaIntrinsic->bElementType;
              if (mfmaLayout.getIsTransposed()) {
                std::swap(valA, valB);
                std::swap(typeA, typeB);
              }
              auto scales = scaleAddresses(m, n, k);
              if (!scales.empty() && mfmaLayout.getIsTransposed())
                std::swap(scales[0], scales[1]);
              acc = generateScaledMFMAOp(intrinsicName, valA, valB, acc, typeA,
                                         typeB, scales);
              if (!firstMfma)
                firstMfma = acc;
            }
            acc = reduceSubBlocks(subBlocks, acc);
            adjustAccForSmallKDim(fc, acc, dstElemTy, b, m, n, numRepM, numRepN,
                                  kDimInstrSize, kDimOperandSize, elemsPerVec);
          }
        }
      }
    }

    if (getEmptyArriveAfterMmacAttr(op))
      ROCDL::SchedBarrier::create(rewriter, loc, /*mask=*/0);

    // Originally, setprio (high) is set to the high-level dot op. After dot is
    // being lowered to the series of mfma operations, it should be moved next
    // to the first mfma leaving the first mfma staying at the low priority. In
    // this way, incoming warp can be effectively waiting on the first mfma
    // instruction (low priority) while the other warp is executing mfma with
    // high priority. Otherwise, incoming warp can break the cluster.
    if (setPrioOp && firstMfma)
      setPrioOp->moveAfter(firstMfma.getDefiningOp());

    const size_t mmaCount =
        numRepB * numRepM * numRepN * numRepK * aKWidth / aKBase;
    packAndReplaceResult(op, fc, maybeMfmaIntrinsic, dstElemTy, elemTyA,
                         mmaCount);

    return success();
  }
};

} // namespace

namespace mlir::triton::HCU {
LogicalResult convertMFMA(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                          const LLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter) {
  auto rankedTType = [](Value tensor) {
    return cast<RankedTensorType>(tensor.getType());
  };

  assert(isa<DotOperandEncodingAttr>(rankedTType(op.getA()).getEncoding()) &&
         isa<DotOperandEncodingAttr>(rankedTType(op.getB()).getEncoding()) &&
         "Both A and B should be DotOperand layout.");

  auto cTensorTy = rankedTType(op.getC());
  auto dTensorTy = rankedTType(op.getD());
  assert(isa<AMDMfmaEncodingAttr>(cTensorTy.getEncoding()) &&
         "Currently, we only support C with a mfma layout.");

  assert(cTensorTy.getShape()[0] == dTensorTy.getShape()[0] &&
         cTensorTy.getShape()[1] == dTensorTy.getShape()[1] &&
         "DotOp's C operand should pass the same number of values as D.");

  auto loc = op.getLoc();
  auto mfmaLayout = cast<AMDMfmaEncodingAttr>(
      cast<RankedTensorType>(op.getResult().getType()).getEncoding());

  DotOpMFMAConversionHelper helper(mfmaLayout, rewriter, typeConverter, loc);

  return helper.convertDot(op, adaptor);
}

LogicalResult convertScaledMFMA(triton::DotScaledOp op,
                                triton::DotScaledOp::Adaptor adaptor,
                                const LLVMTypeConverter *typeConverter,
                                ConversionPatternRewriter &rewriter) {
  // Numeric scales have already been converted to amdg.dot_scaled.
  if (op.getAScale() || op.getBScale())
    return op.emitOpError(
        "numeric scales must be converted to amdg.dot_scaled before "
        "HCU scaled MMAC lowering");
  if (!isa<DotOperandEncodingAttr>(
          cast<RankedTensorType>(op.getA().getType()).getEncoding()) ||
      !isa<DotOperandEncodingAttr>(
          cast<RankedTensorType>(op.getB().getType()).getEncoding()))
    return op.emitError("HCU scaled MMAC requires A/B DotOperand layouts");

  auto cTensorTy = cast<RankedTensorType>(op.getC().getType());
  if (!isa<AMDMfmaEncodingAttr>(cTensorTy.getEncoding()))
    return op.emitError("HCU scaled MMAC requires an MFMA accumulator layout");

  auto loc = op.getLoc();
  auto mfmaLayout = cast<AMDMfmaEncodingAttr>(
      cast<RankedTensorType>(op.getResult().getType()).getEncoding());

  ScaledDotOpMFMAConversionHelper helper(mfmaLayout, rewriter, typeConverter,
                                         loc);

  return helper.convertScaledDot(op, adaptor);
}

LogicalResult convertScaledMFMA(triton::amdgpu::DotScaledOp op,
                                triton::amdgpu::DotScaledOp::Adaptor adaptor,
                                const LLVMTypeConverter *typeConverter,
                                ConversionPatternRewriter &rewriter) {
  // Scale operands reference allocated Scale Buffer storage.
  if (!isa<DotOperandEncodingAttr>(
          cast<RankedTensorType>(op.getA().getType()).getEncoding()) ||
      !isa<DotOperandEncodingAttr>(
          cast<RankedTensorType>(op.getB().getType()).getEncoding()))
    return op.emitError("HCU scaled MMAC requires A/B DotOperand layouts");

  auto cTensorTy = cast<RankedTensorType>(op.getC().getType());
  if (!isa<AMDMfmaEncodingAttr>(cTensorTy.getEncoding()))
    return op.emitError("HCU scaled MMAC requires an MFMA accumulator layout");

  auto loc = op.getLoc();
  auto mfmaLayout = cast<AMDMfmaEncodingAttr>(
      cast<RankedTensorType>(op.getResult().getType()).getEncoding());

  ScaledDotOpMFMAConversionHelper helper(mfmaLayout, rewriter, typeConverter,
                                         loc);

  return helper.convertScaledDot(op, adaptor);
}

// Map an instruction tile and lane to a byte offset in compact LDS. The
// geometry is fixed by the plan; only wave/lane coordinates are runtime values.
// fixedInst >= 0 identifies a round executed only by wave0.
static Value getScaleWriteSourceOffset(const ScaleMemoryPlan &plan,
                                       triton::gpu::MemDescType type,
                                       Value instTileId, int64_t fixedInst,
                                       Value phase, Value col, Location loc,
                                       ConversionPatternRewriter &rewriter) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  int64_t r = type.getShape()[0], s = type.getShape()[1];
  int64_t instTilesAlongS = plan.getInstsAlongS(s);
  Value mn = col;
  if (plan.g > 1) {
    Value group = plan.p == 1 ? phase : b.udiv(phase, b.i32_val(plan.p));
    mn = b.add(b.mul(group, b.i32_val(16)), mn);
  }
  if (r > 16 * plan.g) {
    Value tileR;
    if (fixedInst >= 0) {
      tileR = b.i32_val((fixedInst / instTilesAlongS) * 16 * plan.g);
    } else {
      tileR = instTilesAlongS == 1
                  ? instTileId
                  : b.udiv(instTileId, b.i32_val(instTilesAlongS));
      tileR = b.mul(tileR, b.i32_val(16 * plan.g));
    }
    mn = b.add(tileR, mn);
  }
  Value sk =
      plan.p == 1
          ? Value(b.i32_val(0))
          : Value(b.mul(b.urem(phase, b.i32_val(plan.p)), b.i32_val(plan.b)));
  if (instTilesAlongS > 1) {
    Value tileS;
    if (fixedInst >= 0) {
      tileS = b.i32_val((fixedInst % instTilesAlongS) * plan.instCols);
    } else {
      tileS = r <= 16 * plan.g ? instTileId
                               : b.urem(instTileId, b.i32_val(instTilesAlongS));
      tileS = b.mul(tileS, b.i32_val(plan.instCols));
    }
    sk = plan.p == 1 ? tileS : b.add(tileS, sk);
  }
  // Only partial instruction tiles have padded source coordinates.
  if (r % (16 * plan.g))
    mn = b.select(b.icmp_ult(mn, b.i32_val(r)), mn, b.i32_val(r - 1));
  if (s % plan.instCols)
    sk = b.select(b.icmp_ult(sk, b.i32_val(s - plan.b + 1)), sk,
                  b.i32_val(s - plan.b));
  auto shared =
      cast<triton::gpu::SwizzledSharedEncodingAttr>(type.getEncoding());
  return shared.getOrder()[0] == 1 ? b.add(b.mul(mn, b.i32_val(s)), sk)
                                   : b.add(b.mul(sk, b.i32_val(r)), mn);
}

// Distribute instruction tiles across waves and guard each ds2buf emission
// with both its ownership predicate and the software-pipeline predicate.
static void
emitCondWaveScaleWrite(triton::amdgpu::ScaleWriteOp op,
                       triton::amdgpu::ScaleWriteOp::Adaptor adaptor,
                       const ScaleMemoryPlan &plan, Value smem,
                       ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto sourceType = op.getSrc().getType();
  int64_t s = sourceType.getShape()[1];
  auto baseAttr = op.getRowBaseAttr();
  Value lane = getLaneId(rewriter, loc);
  Value waveId = getUniformWarpId(op, rewriter, loc);
  Value phase = b.udiv(lane, b.i32_val(16)), col = b.urem(lane, b.i32_val(16));
  // Swapping source phases swaps destination row bit2 for wide copies.
  if (plan.b > 1 && plan.bankXor)
    phase = b.xor_(phase, b.i32_val(4 / plan.b));
  int64_t wavesPerCTA = triton::gpu::lookupNumWarps(op);
  for (int64_t round = 0; round < plan.getMaxInstsPerWave(wavesPerCTA);
       ++round) {
    int64_t firstInst = round * wavesPerCTA;
    int64_t activeWaves = std::min(wavesPerCTA, plan.numInsts - firstInst);
    Value instTileId = activeWaves == 1 ? Value(b.i32_val(firstInst))
                       : firstInst == 0 ? waveId
                                        : b.add(waveId, b.i32_val(firstInst));
    Value active;
    if (activeWaves < wavesPerCTA)
      active = b.icmp_ult(waveId, b.i32_val(activeWaves));
    if (adaptor.getPred())
      active = active ? b.and_(active, adaptor.getPred()) : adaptor.getPred();
    Block *after = nullptr;
    if (active) {
      Block *current = rewriter.getInsertionBlock();
      after = rewriter.splitBlock(current, rewriter.getInsertionPoint());
      Block *body = rewriter.createBlock(after);
      rewriter.setInsertionPointToEnd(current);
      LLVM::CondBrOp::create(rewriter, loc, active, body, after);
      rewriter.setInsertionPointToStart(body);
    }
    Value offset = getScaleWriteSourceOffset(plan, sourceType, instTileId,
                                             activeWaves == 1 ? firstInst : -1,
                                             phase, col, loc, rewriter);
    Value addr = b.gep(smem.getType(), rewriter.getI8Type(), smem, offset);
    Value row;
    if (activeWaves == 1) {
      int64_t rowBase = baseAttr.getInt() + firstInst * plan.rowStride;
      if (plan.b == 1) {
        int64_t tileS = (firstInst % plan.getInstsAlongS(s)) * plan.instCols;
        rowBase += 4 * (((tileS / 4) & 1) ^ plan.bankXor);
      }
      row = b.i32_val(rowBase);
    } else {
      row = b.add(b.i32_val(baseAttr.getInt()),
                  b.mul(instTileId, b.i32_val(plan.rowStride)));
      if (plan.b == 1) {
        Value tileS =
            b.mul(b.urem(instTileId, b.i32_val(plan.getInstsAlongS(s))),
                  b.i32_val(plan.instCols));
        Value bank = b.xor_(b.and_(b.udiv(tileS, b.i32_val(4)), b.i32_val(1)),
                            b.i32_val(plan.bankXor));
        row = b.add(row, b.mul(bank, b.i32_val(4)));
      }
    }
    rewriter.create<ROCDL::ds_scale_copy_ds2buf>(
        loc, addr, row, rewriter.getI8IntegerAttr(op.getMode()),
        /*offset0=*/rewriter.getI8IntegerAttr(0),
        /*offset1=*/rewriter.getI8IntegerAttr(0));
    if (after) {
      LLVM::BrOp::create(rewriter, loc, after);
      rewriter.setInsertionPointToStart(after);
    }
  }
}

LogicalResult convertScaleWrite(triton::amdgpu::ScaleWriteOp op,
                                triton::amdgpu::ScaleWriteOp::Adaptor adaptor,
                                const LLVMTypeConverter *converter,
                                ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto type = op.getResult().getType();
  auto sourceType = op.getSrc().getType();
  auto shared = dyn_cast<triton::gpu::SwizzledSharedEncodingAttr>(
      sourceType.getEncoding());
  if (!shared || shared.getMaxPhase() != 1 ||
      (shared.getOrder() != ArrayRef<unsigned>({1, 0}) &&
       shared.getOrder() != ArrayRef<unsigned>({0, 1})))
    return op.emitError(
        "scale requires matching compact unswizzled rank-two i8 LDS");
  auto plan = ScaleMemoryPlan::fromMemDesc(type);
  // Logical [R,S]: R is M for A scales and N for B scales.
  int64_t r = type.getShape()[0], s = type.getShape()[1];
  auto baseAttr = op.getRowBaseAttr();
  // Type compatibility and row_base range/alignment are checked by the op
  // verifier. Lowering additionally requires late allocation to have run.
  if (!baseAttr)
    return op.emitError(
        "scale row_base must be allocated before LLVM lowering");
  if (s % plan.b != 0 || (shared.getOrder()[0] == 0 && plan.b != 1 && r > 1))
    return op.emitError(
        "scale copy width is incompatible with the LDS layout or shape");
  // The copy plan assumes the logical tile has a compact physical pitch.
  // A subslice may translate that instTileId, but may not change its row
  // stride.
  unsigned fast = shared.getOrder()[0];
  if (sourceType.getAllocShape().take_back(2)[fast] !=
      sourceType.getShape()[fast])
    return op.emitError("scale copy requires a compact LDS view; strided "
                        "subslices are unsupported");
  auto smemObject = LLVM::getSharedMemoryObjectFromStruct(
      loc, adaptor.getSrc(),
      converter->convertType(sourceType.getElementType()), rewriter);
  Value smem = smemObject.getShmemAffineBase(loc, rewriter, sourceType);
  emitCondWaveScaleWrite(op, adaptor, plan, smem, rewriter);
  // Carry the allocated storage reference to the consuming dot through SSA.
  rewriter.replaceOp(op, b.i32_val(baseAttr.getInt()));
  return success();
}

} // namespace mlir::triton::HCU
