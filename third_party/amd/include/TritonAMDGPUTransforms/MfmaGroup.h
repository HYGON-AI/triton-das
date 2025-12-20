#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTRANSFORMS_MFMAGROUP_H_
#define TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTRANSFORMS_MFMAGROUP_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/StringRef.h"
#include "TritonAMDGPUToLLVM/TargetUtils.h"

namespace mlir {
using mlir::triton::AMD::HCUISAFeature;

#define MFMA_INTERLEAVE_GET_KIND(flags) ((flags) & 0x3)
#define MFMA_INTERLEAVE_GET_OPERAND(flags) (((flags) >> 2) & 0x3)
#define MFMA_INTERLEAVE_PACK(kind, operand) \
  (((kind) & 0x3) | \
   (((operand) & 0x3) << 2))

// Returns true if the given type is an OCP FP8/FP6/FP6 type.
inline bool isF8F6F4(mlir::Type type) {
  return llvm::isa<Float8E4M3FNType, Float8E5M2Type, Float6E3M2FNType,
                   Float6E2M3FNType, Float4E2M1FNType>(type);
}

struct MfmaIntrinsic {
  // Chooses a suitable mfma instrinsic for the given input case.
  static FailureOr<MfmaIntrinsic> selectFor(int version,unsigned mDim,
                                            unsigned nDim, unsigned inputKDim,
                                            Type aElemType, Type bElemType,
                                            bool withScale, bool useTF32,
                                            HCUISAFeature features,
                                            unsigned interleaveInfo = 0);

  MfmaIntrinsic(StringRef symbol, unsigned m, unsigned n, unsigned k,
                unsigned kB, Type aET, Type bET, unsigned interleaveInfo = 0)
      : name(symbol), mDim(m), nDim(n), kDim(k), kBase(kB), aElementType(aET),
        bElementType(bET), interleaveInfo(interleaveInfo) {}

  MfmaIntrinsic(const MfmaIntrinsic &other) = default;
  MfmaIntrinsic(MfmaIntrinsic &&other) = default;
  MfmaIntrinsic() = default;
  MfmaIntrinsic &operator=(MfmaIntrinsic &&other) = default;

  SmallVector<unsigned, 2> getMfmaTile() { return {mDim, nDim}; }
  SmallVector<unsigned, 2> getInstrShape() { return {instMDim, instNDim}; }
  SmallVector<unsigned, 2> getInstrsPerWarp() { return {instMDim / 16, instNDim / 16}; }
  unsigned getInterleaveKind() { return MFMA_INTERLEAVE_GET_KIND(interleaveInfo); }

  llvm::StringRef name;

  // m, n, and k refer to the shapes of the two operands of an mfma intrinsic:
  // Operand A has shape [m]x[k]; operand B has shape [k]x[n].
  // For mfma32 and mfma16 intrinsics, they are encoded in the instruction
  // name, i.e. mfma_DType_[m]x[n]x[k]xABType.
  unsigned mDim;
  unsigned nDim;
  unsigned kDim;

  // kBase is the number of elements each thread holds.
  unsigned kBase;

  Type aElementType;
  Type bElementType;

  unsigned interleaveInfo;
  unsigned instMDim = 16;
  unsigned instNDim = 16;
};
} // namespace mlir

#endif // TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTRANSFORMS_MFMAGROUP_H_
