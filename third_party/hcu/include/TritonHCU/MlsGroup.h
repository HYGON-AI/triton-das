// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef TRITON_THIRD_PARTY_HCU_INCLUDE_TRITONHCU_MLSGROUP_H_
#define TRITON_THIRD_PARTY_HCU_INCLUDE_TRITONHCU_MLSGROUP_H_

#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "TritonAMDGPUToLLVM/TargetUtils.h"
#include "TritonHCU/TargetUtils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"



namespace mlir::triton::HCU {

//===----------------------------------------------------------------------===//
// HCU MLS instruction selection utilities
//===----------------------------------------------------------------------===//

// flags: lsb: elem3 | row3 << 8 | col3 << 16 | alt2 << 24  msb
// elem3: 0: 4bits,  1: 8bits,   2: 16bit,   3: 32bit,                              Note: for padbyte insn, 0: 4bits, 1: 6bits
// row3 : 0: 8elems, 1: 16elems, 2: 32elems, 3: 64elems, 4: 128elems   5: 256elems, Note: for padbyte insn, fixed 0 ie 32elems
// col3 : 0: 8elems, 1: 16elems, 2: 32elems, 3: 64elems, 4: 128elems   5: 256elems, Note: for padbyte insn, fixed 0 ie 32elems
// alt2 : 0: interleave none, 1: interleave 2, 2: interleave 4, 3: interleave 8
#define MLS_DS_FLAGS_GET_ELEM3(flags) ((flags) & 0xFF)
#define MLS_DS_FLAGS_GET_ROW3(flags) (((flags) >> 8) & 0xFF)
#define MLS_DS_FLAGS_GET_COL3(flags) (((flags) >> 16) & 0xFF)
#define MLS_DS_FLAGS_GET_ALT2(flags) (((flags) >> 24) & 0xFF)

#define MLS_DS_FLAGS_PACK(elem3, row3, col3, alt2) \
  (((elem3) & 0xFF) | \
   (((row3) & 0xFF) << 8) | \
   (((col3) & 0xFF) << 16) | \
   (((alt2) & 0xFF) << 24))

#define MLS_INST_DS_CNT_MAX  (16)

enum class MlsInterleaveKind : unsigned {
  InterleaveNone = 0,
  Interleave2    = 1,
  Interleave4    = 2,
  Interleave8    = 3,
};

using MlsElemBitTyKind = triton::gpu::MlsElemBitTyKind;
using MlsTileKind = triton::gpu::MlsTileKind;

#define MLS_GEN_SCALED_EXT(mlsElemBitTyKind, isNonKPack) \
  ((((unsigned)(mlsElemBitTyKind) & 0xF) << 0) | \
   (((unsigned)(isNonKPack) & 0x1) << 4))

#define MLS_GET_SCALED_EXT_ELEMBITTYKIND(scaledExt) ((unsigned)(scaledExt) & 0xF)
#define MLS_GET_SCALED_EXT_ISNONKPACK(scaledExt) ((((unsigned)(scaledExt) >> 4) & 0x1) != 0)

inline MlsElemBitTyKind getMlsElemBitTyKind(unsigned mlsScaledExt) {
  return static_cast<MlsElemBitTyKind>(
      MLS_GET_SCALED_EXT_ELEMBITTYKIND(mlsScaledExt));
}

inline bool getMlsIsNonKPack(unsigned mlsScaledExt) {
  return MLS_GET_SCALED_EXT_ISNONKPACK(mlsScaledExt);
}

inline bool isMlsB4PackedElemTyKind(MlsElemBitTyKind elemBitTyKind) {
  return elemBitTyKind == MlsElemBitTyKind::B4;
}

inline bool isMlsB4MixElemTyKind(MlsElemBitTyKind elemBitTyKind) {
  return elemBitTyKind == MlsElemBitTyKind::B4Mix;
}

inline bool isMlsPaddedBit4ElemTyKind(MlsElemBitTyKind elemBitTyKind) {
  return elemBitTyKind == MlsElemBitTyKind::I4 ||
         elemBitTyKind == MlsElemBitTyKind::U4 ||
         elemBitTyKind == MlsElemBitTyKind::F4;
}

inline bool isMlsBit4ElemTyKind(MlsElemBitTyKind elemBitTyKind) {
  return isMlsB4PackedElemTyKind(elemBitTyKind) ||
         isMlsB4MixElemTyKind(elemBitTyKind) ||
         isMlsPaddedBit4ElemTyKind(elemBitTyKind);
}

// 4-bit MLS operands go through different physical states across the pipeline:
// Global -> LDS -> DotOperand -> MMAC.
// Keep that flow logic centralized here so shape expansion, address math and
// kWidth selection all follow the same rules.
enum class MlsDataFlowView : unsigned {
  Global = 0,
  Shared = 1,
  DotOperand = 2,
};

struct MlsDataFlowInfo {
  MlsElemBitTyKind elemBitTyKind;

  bool isBit4() const { return isMlsBit4ElemTyKind(elemBitTyKind); }

  bool usesPackedShape(MlsDataFlowView view) const {
    if (!isBit4())
      return false;

    switch (view) {
    case MlsDataFlowView::Global:
      return true;
    case MlsDataFlowView::Shared:
      return !isMlsPaddedBit4ElemTyKind(elemBitTyKind);
    case MlsDataFlowView::DotOperand:
      return isMlsB4PackedElemTyKind(elemBitTyKind);
    }
    llvm_unreachable("unexpected MLS data-flow view");
  }

  bool expandsShape(MlsDataFlowView view) const {
    return isBit4() && !usesPackedShape(view);
  }

  unsigned getMfmaKWidth(unsigned dotOperandKWidth) const {
    return usesPackedShape(MlsDataFlowView::DotOperand) ? dotOperandKWidth * 2
                                                        : dotOperandKWidth;
  }
};

inline MlsDataFlowInfo getMlsDataFlowInfo(MlsElemBitTyKind elemBitTyKind) {
  return {elemBitTyKind};
}

inline MlsDataFlowInfo getMlsDataFlowInfo(unsigned mlsScaledExt) {
  return getMlsDataFlowInfo(getMlsElemBitTyKind(mlsScaledExt));
}

// Map ScaleDotElemType to MlsElemBitTyKind for MLS instructions.
inline MlsElemBitTyKind scaleDotElemTypeToMlsElemBitTyKind(
    mlir::triton::ScaleDotElemType elemType, bool isF4Padding) {
  switch (elemType) {
  case mlir::triton::ScaleDotElemType::E2M1:
    return isF4Padding ? MlsElemBitTyKind::F4 : MlsElemBitTyKind::B4;
  case mlir::triton::ScaleDotElemType::E4M3:
  case mlir::triton::ScaleDotElemType::E5M2:
    return MlsElemBitTyKind::None;
  case mlir::triton::ScaleDotElemType::E2M3:
  case mlir::triton::ScaleDotElemType::E3M2:
    return MlsElemBitTyKind::F6;
  default:
    return MlsElemBitTyKind::None;
  }
}

// Determine MLS version based on HCU ISA features
// Returns: 1 for NMZ, 2 for YY, 3 for SB
inline unsigned getMlsVersionFromFeatures(HCUISAFeature features) {
  if (features & HCUISAFeature::MLS_B4)     // SB supports MLS_B4 (fp4 non-padded)
    return 3;

  if (features & HCUISAFeature::MLS_FP6FP4) // YY supports MLS_FP6FP4 (fp6 and fp4 padded)
    return 2;

  if (features & HCUISAFeature::MLS) // NMZ supports basic MLS
    return 1;

  return 0;
}


struct MlsInsnAttr {
  unsigned opIdx;                  // or nonKTileIdx;
  std::array<unsigned, 2> mlsTile; // A: [m, k] or B: [k, n]
  bool kMajor;
  unsigned elemBitWidth;           // 8, 16, 32
  MlsElemBitTyKind elemBitTyKind;
  MlsInterleaveKind interleaveKind;
  unsigned mlsVersion;             // 1: NMZ, 2: YY, 3: SB

  struct Mlnsn{
    std::array<unsigned, 2> instrShape;
    std::array<unsigned, 2> instrsPerWarp;
    std::array<unsigned, 2> instrOrder;
    llvm::StringRef insn;
    std::array<unsigned, MLS_INST_DS_CNT_MAX> dsByteOffsets;
  } mlInsn;

  struct DsInsn {
    std::array<unsigned, 2> instrShape;
    std::array<unsigned, 2> instrsPerWarp;
    std::array<unsigned, 2> instrOrder;
    llvm::StringRef insn;
    unsigned flags;
    std::array<unsigned, MLS_INST_DS_CNT_MAX> dsByteOffsets;
  } dsInsn;

  struct MsInsn {
    std::array<unsigned, 2> instrShape;
    std::array<unsigned, 2> instrsPerWarp;
    std::array<unsigned, 2> instrOrder;
    llvm::StringRef insn;
  } msInsn;
};

using MatrixLoadInsnAttr   = MlsInsnAttr::Mlnsn;
using DsReadMatrixInsnAttr = MlsInsnAttr::DsInsn;
using MatrixStoreInsnAttr  = MlsInsnAttr::MsInsn;

class MlsInsn {
private:
  MlsInsnAttr attr;

public:
  // get the candidates of elems tile for the given parameters, the result is sorted by the nonKTile,
  // the first element of the pair is the nonKTile, the second element of the pair is the kTile
  static SmallVector<std::pair<unsigned, unsigned>> getMlsElemsTileCandidates(
                                              unsigned opIdx,
                                              bool kMajor,
                                              unsigned elemBitWidth,
                                              MlsElemBitTyKind elemBitTyKind,
                                              unsigned mlsVersion,
                                              MlsInterleaveKind altKind = MlsInterleaveKind::InterleaveNone);

  static FailureOr<MlsInsn> selectOrGetMlsInsn(unsigned nonKTile,       /* elems of non-k-tile */
                                               unsigned kTile,          /* elems of k-tile */
                                               unsigned elemBitWidth,
                                               MlsElemBitTyKind elemBitTyKind,
                                               unsigned opIdx,
                                               bool kMajor,
                                               unsigned mlsVersion,
                                               MlsInterleaveKind altKind = MlsInterleaveKind::InterleaveNone);
  static FailureOr<MlsInsn>
  selectOrGetMatrixStoreInsn(unsigned mTile, unsigned nTile,
                             unsigned elemBitWidth, unsigned mlsVersion,
                             MlsInterleaveKind interleaveKind,
                             bool transpose);
  MlsInsn(const MlsInsnAttr &attr) : attr(attr) {}

  unsigned getOpIdx() const { return attr.opIdx; }

  bool isKMajor() const { return attr.kMajor; }

  bool isTranspose() const { return attr.kMajor; }

  bool isRowMajor() const { return attr.opIdx == 0 ? attr.kMajor : !attr.kMajor; }

  unsigned getMajorDimIndex() const { return attr.opIdx == 0 ? attr.kMajor : !attr.kMajor; }

  unsigned getDotLayoutKWidth() const {
    return attr.elemBitWidth == 16 ? 4
             : (attr.elemBitWidth == 8 ? 8 : 8); // F8F6F4 due to padding: 8, B4 due to 2 pack: 8
  }

  // Returns the instruction tile in elem/global/shared view. Tensor shapes are
  // kept in logic space; only instruction-level tiles switch view here.
  SmallVector<unsigned> getMlsTile(MlsTileKind tileKind = MlsTileKind::Elems) const {
    SmallVector<unsigned> tile{attr.mlsTile[0], attr.mlsTile[1]};
    if (tileKind == MlsTileKind::Elems || attr.elemBitTyKind == MlsElemBitTyKind::None)
      return tile;

    auto flow = getMlsDataFlowInfo(attr.elemBitTyKind);
    bool shrinkPackedDim =
        flow.usesPackedShape(tileKind == MlsTileKind::Global
                                 ? MlsDataFlowView::Global
                                 : MlsDataFlowView::Shared);
    if (!shrinkPackedDim)
      return tile;

    unsigned packDimIdx = isKMajor() ? !attr.opIdx : attr.opIdx;
    tile[packDimIdx] /= 2;
    return tile;
  }

  unsigned getElemBitWidth() const { return attr.elemBitWidth; }

  unsigned getElemBitTyKind() const { return static_cast<unsigned>(attr.elemBitTyKind); }

  MlsInterleaveKind getInterleaveKind() const { return attr.interleaveKind; }

  unsigned getAlt2Kind() const { return static_cast<unsigned>(getInterleaveKind()); }

  unsigned getMlsVersion() const { return attr.mlsVersion; }

  const MatrixLoadInsnAttr &getMatrixLoadInsnAttr() const { return attr.mlInsn; }

  const DsReadMatrixInsnAttr &getDsReadMatrixInsnAttr() const { return attr.dsInsn; }

  const MatrixStoreInsnAttr &getMatrixStoreInsnAttr() const { return attr.msInsn; }
};

} // namespace mlir::triton::HCU

#endif // TRITON_THIRD_PARTY_HCU_INCLUDE_TRITONHCU_MLSGROUP_H_
