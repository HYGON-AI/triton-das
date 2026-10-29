#include "TargetInfo.h"
#include "TritonAMDGPUToLLVM/TargetUtils.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {

// AMD's generic eight-element LUT upcast. The CDNA3 FP8 intermediate path
// depends on AMD-specific FP8 semantics and is not used for HCU.
SmallVector<Value> upcast8xFp4_SW(RewriterBase &rewriter, Operation *op,
                                  bool toFp16, Value packedVec) {
  Location loc = op->getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto permU32FnTy =
      LLVM::LLVMFunctionType::get(i32_ty, {i32_ty, i32_ty, i32_ty});
  LLVM::LLVMFuncOp funcOp =
      appendOrGetExternFuncOp(rewriter, op, "llvm.amdgcn.perm", permU32FnTy);
  Value input = b.bitcast(packedVec, i32_ty);
  // MXFP4 has 4 bits, S.EE.M, for Sign, Exponent, and Mantissa respectively.
  // For a specific S, we have a total of 8 bit patterns. We can encode all
  // these 8 resultant bf16/fp16 bit patterns in a lookup table (LUT). It
  // happens that llvm.amdgcn.perm supports selecting 4 bytes from 8 input bytes
  // using a 4-byte selector. So the overall idea is to use llvm.amdgcn.perm to
  // implement such a LUT; though we need to select the two bytes for the
  // resultant bf16/fp16 bit patterns separately. For the byte containing S, we
  // also need to handle the S and E bits separately.

  // FP4 has 4 bits: S.EE.M. Bf16/fp16 bit patterns for positive values:
  //
  // FP4    | BF16   | FP16   | Value
  // ------ | ------ | ------ | -----
  // 0.00.0 | 0x0000 | 0x0000 | + 0.0
  // 0.00.1 | 0x3f00 | 0x3800 | + 0.5
  // 0.01.0 | 0x3f80 | 0x3c00 | + 1.0
  // 0.01.1 | 0x3fc0 | 0x3e00 | + 1.5
  // 0.10.0 | 0x4000 | 0x4000 | + 2.0
  // 0.10.1 | 0x4040 | 0x4200 | + 3.0
  // 0.11.0 | 0x4080 | 0x4400 | + 4.0
  // 0.11.1 | 0x40c0 | 0x4600 | + 6.0
  //
  // Encode Byte #0 (M) for BF16/FP16 in a LUT.
  Value resB0LutLo = toFp16 ? b.i32_val(0) : b.i32_val(0xc0800000);
  Value resB0LutHi = toFp16 ? b.i32_val(0) : b.i32_val(0xc0804000);
  // Encode Byte #1 (EM, non-S part) for BF16/FP16 in a LUT.
  Value resB1LutLoNoS = toFp16 ? b.i32_val(0x3e3c3800) : b.i32_val(0x3f3f3f00);
  Value resB1LutHiNoS = toFp16 ? b.i32_val(0x46444240) : b.i32_val(0x40404040);

  // Step 1: extract EM bits for elements 0,2,4,6 and 1,3,5,7 respectively.
  // e2m1_6420_idx = | 0[0e6EM] | 0[0e4EM] | 0[0e2EM] | 0[0e0EM] |
  Value e2m1_6420_idx = b.and_(input, b.i32_val(0x07070707));
  // e2m1_7531_idx = | [0e7EM]0 | [0e5EM]0 | [0e3EM]0 | [0e1EM]0 |
  Value e2m1_7531_idx = b.and_(input, b.i32_val(0x70707070));
  // e2m1_7531_idx = | 0[0e7EM] | 0[0e5EM] | 0[0e3EM] | 0[0e1EM] |
  e2m1_7531_idx = b.lshr(e2m1_7531_idx, b.i32_val(4));

  // Step 2: extract S bit for elements 0,2,4,6 and 1,3,5,7
  // s_6420 = | 0[e6S000] | 0[e4S000] | 0[e2S000] | 0[e0S000] |
  Value s_6420 = b.and_(input, b.i32_val(0x08080808));
  // s_6420 = | [e6S000]0 | [e4S000]0 | [e2S000]0 | [e0S000]0 |
  s_6420 = b.shl(s_6420, b.i32_val(4));
  // s_7531 = | [e7S000]0 | [e5S000]0 | [e3S000]0 | [e1S000]0 |
  Value s_7531 = b.and_(input, b.i32_val(0x80808080));

  // Step 3: Upcast elements 0,2,4,6 to 4 16-bit elements
  // Select Byte #0. It's always 0 if upcasting to fp16.
  // resB0_6420 = | e6B0 | e4B0 | e2B0 | e0B0 |
  Value resB0_6420 = b.i32_val(0);
  if (!toFp16) {
    resB0_6420 = LLVM::createLLVMCallOp(rewriter, loc, funcOp,
                                        {resB0LutHi, resB0LutLo, e2m1_6420_idx})
                     .getResult();
  }
  // Select Byte #1
  Value resB1NoS_6420 =
      LLVM::createLLVMCallOp(rewriter, loc, funcOp,
                             {resB1LutHiNoS, resB1LutLoNoS, e2m1_6420_idx})
          .getResult();
  // resB1_6420 = | e6B1 | e4B1 | e2B1 | e0B1 |
  Value resB1_6420 = b.or_(resB1NoS_6420, s_6420);
  // Construct 16-bit values of e0 and e2
  // res_20 = | e2B1 | e2B0 | e0B1 | e0B0 | = | e2_f16 | e0_f16 |
  Value res_20 =
      LLVM::createLLVMCallOp(rewriter, loc, funcOp,
                             {resB1_6420, resB0_6420, b.i32_val(0x05010400)})
          .getResult();
  // Construct 16-bit values of e4 and e6
  // res_64 = | e6B1 | e6B0 | e4B1 | e4B0 | = | e6_f16 | e4_f16 |
  Value res_64 =
      LLVM::createLLVMCallOp(rewriter, loc, funcOp,
                             {resB1_6420, resB0_6420, b.i32_val(0x07030602)})
          .getResult();

  // Step 4: Upcast elements 1,3,5,7 to 4 16-bit elements
  // This is a copy of step 3 on different group of elements
  // Select Byte #0. It's always 0 if upcasting to fp16.
  // resB0_7531 = | e7B0 | e5B0 | e3B0 | e1B0 |
  Value resB0_7531 = b.i32_val(0);
  if (!toFp16) {
    resB0_7531 = LLVM::createLLVMCallOp(rewriter, loc, funcOp,
                                        {resB0LutHi, resB0LutLo, e2m1_7531_idx})
                     .getResult();
  }
  // Select Byte #1
  Value resB1NoS_7531 =
      LLVM::createLLVMCallOp(rewriter, loc, funcOp,
                             {resB1LutHiNoS, resB1LutLoNoS, e2m1_7531_idx})
          .getResult();
  // resB1_7531 = | e7B1 | e5B1 | e3B1 | e1B1 |
  Value resB1_7531 = b.or_(resB1NoS_7531, s_7531);
  // Construct 16-bit values of e1 and e3
  // res_31 = | e3B1 | e3B0 | e1B1 | e1B0 | = | e3_f16 | e1_f16 |
  Value res_31 =
      LLVM::createLLVMCallOp(rewriter, loc, funcOp,
                             {resB1_7531, resB0_7531, b.i32_val(0x05010400)})
          .getResult();
  // Construct 16-bit values of e5 and e7
  // res_75 = | e7B1 | e7B0 | e5B1 | e5B0 | = | e7_f16 | e5_f16 |
  Value res_75 =
      LLVM::createLLVMCallOp(rewriter, loc, funcOp,
                             {resB1_7531, resB0_7531, b.i32_val(0x07030602)})
          .getResult();

  // Step 5: Reorder 16-bit elements to be 0,1,2,3,4,5,6,7
  // res_10 = | e1_f16 | e0_f16 |
  Value res_10 = LLVM::createLLVMCallOp(rewriter, loc, funcOp,
                                        {res_31, res_20, b.i32_val(0x05040100)})
                     .getResult();
  // res_32 = | e3_f16 | e2_f16 |
  Value res_32 = LLVM::createLLVMCallOp(rewriter, loc, funcOp,
                                        {res_31, res_20, b.i32_val(0x07060302)})
                     .getResult();
  // res_54 = | e5_f16 | e4_f16 |
  Value res_54 = LLVM::createLLVMCallOp(rewriter, loc, funcOp,
                                        {res_75, res_64, b.i32_val(0x05040100)})
                     .getResult();
  // res_76 = | e7_f16 | e6_f16 |
  Value res_76 = LLVM::createLLVMCallOp(rewriter, loc, funcOp,
                                        {res_75, res_64, b.i32_val(0x07060302)})
                     .getResult();

  SmallVector<Value, 4> pkVals{res_10, res_32, res_54, res_76};
  SmallVector<Value> results;
  Type elmTy = toFp16 ? f16_ty : bf16_ty;
  for (int j = 0; j < 4; j++) {
    Value elements = b.bitcast(pkVals[j], vec_ty(elmTy, 2));
    results.push_back(b.extract_element(elements, b.i32_val(0)));
    results.push_back(b.extract_element(elements, b.i32_val(1)));
  }
  return results;
}

Value upcast2xFp4_HW(Location loc, ConversionPatternRewriter &rewriter,
                     Value input, Type elemType) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value scale = b.i32_val(127);
  // E8M0 127 is scale=1; src_sel=4 selects the low compact FP4 byte.
  if (elemType.isF16())
    return ROCDL::HCUCvtScalePkF16Fp4Op::create(
        rewriter, loc, vec_ty(elemType, 2), input, scale,
        /*srcSel=*/4, /*clamp=*/false, /*scaleSel=*/0);
  Value pair = ROCDL::HCUCvtScalePkBf16Fp4Op::create(
      rewriter, loc, vec_ty(i16_ty, 2), input, scale,
      /*srcSel=*/4, /*clamp=*/false, /*scaleSel=*/0);
  return b.bitcast(pair, vec_ty(elemType, 2));
}

class Fp4ToFpOpPattern : public ConvertOpToLLVMPattern<Fp4ToFpOp> {
private:
  const AMD::TargetInfo &targetInfo;

public:
  Fp4ToFpOpPattern(LLVMTypeConverter &typeConverter,
                   const AMD::TargetInfo &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<Fp4ToFpOp>(typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(Fp4ToFpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type elemType = op.getType().getElementType();
    assert((elemType.isF16() || elemType.isBF16()) && "unsupported FP4 upcast");
    bool useHardware = AMD::supportsHCUISAFeature(
        targetInfo.getArch(), AMD::HCUISAFeature::CVT_SCALE_PK);
    auto inputs = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    SmallVector<Value> results;
    results.reserve(inputs.size() * 2);
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    if (useHardware) {
      for (Value value : inputs) {
        Value pair =
            upcast2xFp4_HW(loc, rewriter, b.zext(i32_ty, value), elemType);
        results.push_back(b.extract_element(elemType, pair, b.i32_val(0)));
        results.push_back(b.extract_element(elemType, pair, b.i32_val(1)));
      }
    } else {
      for (unsigned i = 0; i < inputs.size(); i += 4) {
        unsigned count = std::min<unsigned>(4, inputs.size() - i);
        Value packed = b.undef(vec_ty(i8_ty, 4));
        for (unsigned j = 0; j < count; ++j)
          packed = b.insert_element(packed, inputs[i + j], b.i32_val(j));
        auto values = upcast8xFp4_SW(rewriter, op, elemType.isF16(), packed);
        results.append(values.begin(), values.begin() + count * 2);
      }
    }
    rewriter.replaceOp(op, packLLElements(loc, getTypeConverter(), results,
                                          rewriter, op.getType()));
    return success();
  }
};
} // namespace

namespace mlir::triton::HCU {
void populateFp4ToFpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                   RewritePatternSet &patterns,
                                   const AMD::TargetInfo &targetInfo,
                                   PatternBenefit benefit) {
  patterns.add<Fp4ToFpOpPattern>(typeConverter, targetInfo, benefit);
}
} // namespace mlir::triton::HCU
