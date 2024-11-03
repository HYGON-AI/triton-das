#include "TritonAMDGPUTransforms/MfmaGroup.h"

namespace mlir {

static MfmaTypeId chooseAppropriateMfmaId(mlir::Type dataTypeA,
                                          mlir::Type dataTypeB) {
  if (dataTypeA.isF32() && dataTypeB.isF32()) {
    return MfmaTypeId::Fp32TyId;
  }
  if (dataTypeA.isF16() && dataTypeB.isF16()) {
    return MfmaTypeId::Fp16TyId;
  }
  if (dataTypeA.isBF16() && dataTypeB.isBF16()) {
    return MfmaTypeId::Bf16TyId;
  }
  if (dataTypeA.isInteger(8) && dataTypeB.isInteger(8)) {
    return MfmaTypeId::I8TyId;
  }
  if (dataTypeA.isFloat8E4M3FNUZ() && dataTypeB.isFloat8E4M3FNUZ()) {
    return MfmaTypeId::Fp8Fp8TyId;
  }
  if (dataTypeA.isFloat8E4M3FNUZ() && dataTypeB.isFloat8E5M2FNUZ()) {
    return MfmaTypeId::Fp8Bf8TyId;
  }
  if (dataTypeA.isFloat8E5M2FNUZ() && dataTypeB.isFloat8E4M3FNUZ()) {
    return MfmaTypeId::Bf8Fp8TyId;
  }
  if (dataTypeA.isFloat8E5M2FNUZ() && dataTypeB.isFloat8E5M2FNUZ()) {
    return MfmaTypeId::Bf8Bf8TyId;
  }
  if (dataTypeA.isFloat8E5M2() && dataTypeB.isFloat8E5M2()) {
    return MfmaTypeId::Fp16TyId;
  }
  llvm_unreachable("Unsupported input argument type.");
}

using MfmaInsnGroupMap = llvm::DenseMap<MfmaInsnGroupSelectKey, MfmaInsnAttr,
                                        MfmaInsnGroupSelectKeyInfo>;

auto getMfmaInsnGroupAttrMap = []() -> const MfmaInsnGroupMap & {
  static MfmaInsnGroupMap MfmaInsnMap{
      // f32
      // mmac_f32_16x16x8f32
      {{16, 16, MfmaTypeId::Fp32TyId, 3},
       {16, 16, 8, 2, ROCDL::mmac_f32_16x16x8f32::getOperationName()}},
      // mmac_f32_16x16x4f32
      {{16, 16, MfmaTypeId::Fp32TyId, 3},
       {16, 16, 4, 1, ROCDL::mmac_f32_16x16x4f32::getOperationName()}},
      // f16
      // mmac_f32_16x16x16xf16
      {{16, 16, MfmaTypeId::Fp16TyId, 3},
       {16, 16, 16, 4, ROCDL::mmac_f32_16x16x16f16::getOperationName()}},
      // bf16
      // mmac_f32_16x16x16xbf16
      {{16, 16, MfmaTypeId::Bf16TyId, 3},
       {16, 16, 16, 4, ROCDL::mmac_f32_16x16x16bf16::getOperationName()}},
      // int8
      // mmac_i32_16x16x32i8
      {{16, 16, MfmaTypeId::I8TyId, 3},
       {16, 16, 32, 8, ROCDL::mmac_i32_16x16x32i8::getOperationName()}},
      // fp8 * pf8
      // mmac_f32_16x16x32_fp8_fp8
      {{16, 16, MfmaTypeId::Fp8Fp8TyId, 3},
       {16, 16, 32, 8, ROCDL::mmac_f32_16x16x32_fp8_fp8::getOperationName()}},
      // mmac_f32_16x16x32_fp8_bf8
      {{16, 16, MfmaTypeId::Fp8Bf8TyId, 3},
       {16, 16, 32, 8, ROCDL::mmac_f32_16x16x32_fp8_bf8::getOperationName()}},
      // mmac_f32_16x16x32_bf8_fp8
      {{16, 16, MfmaTypeId::Bf8Fp8TyId, 3},
       {16, 16, 32, 8, ROCDL::mmac_f32_16x16x32_bf8_fp8::getOperationName()}},
      // mmac_f32_16x16x32_bf8_bf8
      {{16, 16, MfmaTypeId::Bf8Bf8TyId, 3},
       {16, 16, 32, 8, ROCDL::mmac_f32_16x16x32_bf8_bf8::getOperationName()}}};
  return MfmaInsnMap;
};

std::pair<mlir::Type, mlir::Type> TypesFromMfmaId(mlir::MLIRContext *ctx,
                                                  MfmaTypeId id) {
  auto f8e5m2 = Float8E5M2Type::get(ctx);
  auto f8e4m3fnuz = Float8E4M3FNUZType::get(ctx);
  auto f8e5m2fnuz = Float8E5M2FNUZType::get(ctx);
  auto f16 = Float16Type::get(ctx);
  auto bf16 = BFloat16Type::get(ctx);
  auto f32 = Float32Type::get(ctx);
  auto i8 = IntegerType::get(ctx, 8, IntegerType::Signed);
  switch (id) {
  case MfmaTypeId::Fp32TyId:
    return {f32, f32};
  case MfmaTypeId::Fp16TyId:
    return {f16, f16};
  case MfmaTypeId::Bf16TyId:
    return {bf16, bf16};
  case MfmaTypeId::I8TyId:
    return {i8, i8};
  case MfmaTypeId::Fp8Fp8TyId:
    return {f8e4m3fnuz, f8e4m3fnuz};
  case MfmaTypeId::Fp8Bf8TyId:
    return {f8e4m3fnuz, f8e5m2fnuz};
  case MfmaTypeId::Bf8Fp8TyId:
    return {f8e5m2fnuz, f8e4m3fnuz};
  case MfmaTypeId::Bf8Bf8TyId:
    return {f8e5m2fnuz, f8e5m2fnuz};
  }
  assert(false && "unsupported MfmaTypeId");
}

FailureOr<MfmaInsn> MfmaInsn::selectMfma(unsigned mDim, unsigned nDim,
                                         Type elementTypeA, Type elementTypeB,
                                         int mfmaVersion) {
  auto mfmaInsnAttrMap = getMfmaInsnGroupAttrMap();
  MfmaTypeId mfmaId = chooseAppropriateMfmaId(elementTypeA, elementTypeB);
  MfmaInsnGroupSelectKey key = {mDim, nDim, mfmaId, mfmaVersion};
  auto it = mfmaInsnAttrMap.find(key);
  if (it == mfmaInsnAttrMap.end())
    return failure();
  auto [instrElementTypeA, instrElementTypeB] =
      TypesFromMfmaId(elementTypeA.getContext(), mfmaId);
  return MfmaInsn(instrElementTypeA, instrElementTypeB, it->second);
}

MfmaInsn::MfmaInsn(Type elementTypeA, Type elementTypeB,
                   const MfmaInsnAttr &attr)
    : elementTypeA(elementTypeA), elementTypeB(elementTypeB), attr(attr) {}

unsigned MfmaInsn::getKDim() { return attr.k; }
unsigned MfmaInsn::getMDim() { return attr.m; }
unsigned MfmaInsn::getNDim() { return attr.n; }
StringRef MfmaInsn::getInsnName() { return attr.insn; }
unsigned MfmaInsn::getKBase() { return attr.kBase; }
Type MfmaInsn::getElementTypeA() { return elementTypeA; }
Type MfmaInsn::getElementTypeB() { return elementTypeB; }
} // namespace mlir
