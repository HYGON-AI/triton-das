#include "TritonHCU/MlsGroup.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::triton::HCU {

struct MlsInsnGroupSelectKey {
  unsigned nonKTile;
  unsigned kTile;
  bool transpose;
  unsigned elemBitWidth;
  MlsElemBitTyKind elemBitTyKind;
  MlsInterleaveKind alt2Kind;
  unsigned version;
};

struct iMlsInsnAttr {
  struct {
    unsigned instNonKDim;
    unsigned instKDim;
    llvm::StringRef insn;
    std::array<unsigned, MLS_INST_DS_CNT_MAX> dsByteOffsets;
  } mlInfo;
  struct {
    unsigned instNonKDim;
    unsigned instKDim;
    llvm::StringRef insn;
    unsigned flags; // lsb: elem3 | row3 << 8 | col3 << 16 | alt2 << 24  msb
    std::array<unsigned, MLS_INST_DS_CNT_MAX> dsByteOffsets;
  } dsInfo;
};


template <typename T>
constexpr typename std::underlying_type<T>::type cast_as_underlying2(T t) {
  return static_cast<typename std::underlying_type<T>::type>(t);
}
struct MlsInsnGroupSelectKeyInfo
    : public llvm::DenseMapInfo<MlsInsnGroupSelectKey> {
  static inline MlsInsnGroupSelectKey getEmptyKey() {
    return {0, 0, false, 16, MlsElemBitTyKind::None,
            MlsInterleaveKind::InterleaveNone, 0};
  }

  static inline MlsInsnGroupSelectKey getTombstoneKey() {
    return {~0U, ~0U, true, ~0U, MlsElemBitTyKind::None,
            MlsInterleaveKind::InterleaveNone, ~0U};
  }

  static inline bool isEqual(const MlsInsnGroupSelectKey &lhs,
                             const MlsInsnGroupSelectKey &rhs) {
    return lhs.nonKTile == rhs.nonKTile && lhs.kTile == rhs.kTile &&
           lhs.transpose == rhs.transpose && lhs.elemBitWidth == rhs.elemBitWidth &&
           lhs.elemBitTyKind == rhs.elemBitTyKind &&
           lhs.version == rhs.version && lhs.alt2Kind == rhs.alt2Kind;
  }

  static unsigned getHashValue(const MlsInsnGroupSelectKey &key) {
    auto nonKTileHash = llvm::detail::combineHashValue(key.nonKTile, key.kTile);
    auto transposeHash = llvm::detail::combineHashValue(nonKTileHash, key.transpose);
    auto elemBitWidthHash = llvm::detail::combineHashValue(transposeHash, key.elemBitWidth);
    auto elemBitTyKindHash = llvm::detail::combineHashValue(
        elemBitWidthHash, cast_as_underlying2(key.elemBitTyKind));
    auto alt2KindHash =
        llvm::detail::combineHashValue(elemBitTyKindHash,
                                       cast_as_underlying2(key.alt2Kind));
    return llvm::detail::combineHashValue(alt2KindHash, key.version);
  }
};


using MlsInsnGroupMap = llvm::DenseMap<MlsInsnGroupSelectKey, iMlsInsnAttr,
                                        MlsInsnGroupSelectKeyInfo>;

template <typename Fn>
static bool forEachCompatibleMlsVersion(unsigned version, Fn &&fn) {
  if (fn(version))
    return true;
  if (version == 3)
    return fn(2);
  return false;
}

auto getMlsInsnGroupAttrMap = []() -> const MlsInsnGroupMap & {
  static MlsInsnGroupMap MlsInsnMap {
    // -------------------------------------- NMZ --------------------------------------------------
    // b16:
    // matrix_load_32x16_b16: AT/BN(transpose = 1), interleave 0,  mnk required elems tile: [16, 16, 32] -> load tile: [16, 16, 32] -> lds tile: [16, 16, 32]
    {{16, 32, true, 16, MlsElemBitTyKind::None, MlsInterleaveKind::InterleaveNone, 1},
      {{16, 32, ROCDL::hcu_matrix_load_32X16_b16::getOperationName(), {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {16, 32, ROCDL::hcu_ds_read_matrix_trans_format::getOperationName(),
                MLS_DS_FLAGS_PACK(2, 2, 1, 0), {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}
      }
    },

    // matrix_load_32x16_b16: AN/BT(transpose = 0), interleave 0,  mnk required elems tile: [32, 32, 16] -> load tile: [32, 32, 16] -> lds tile: [32, 32, 16]
    {{32, 16, false, 16, MlsElemBitTyKind::None, MlsInterleaveKind::InterleaveNone, 1},
      {{32, 16, ROCDL::hcu_matrix_load_32X16_b16::getOperationName(), {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {32, 16, ROCDL::hcu_ds_read_matrix_format::getOperationName(),
                MLS_DS_FLAGS_PACK(2, 2, 1, 0), {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}
      }
    },

    // matrix_load_64x16_b16: AT/BN(transpose = 1), interleave 0,  mnk required elems tile: [16, 16, 64] -> load tile: [16, 16, 64] -> lds tile: [16, 16, 64]
    {{16, 64, true, 16, MlsElemBitTyKind::None, MlsInterleaveKind::InterleaveNone, 1},
      {{16, 64, ROCDL::hcu_matrix_load_64X16_b16::getOperationName(), {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {16, 32, ROCDL::hcu_ds_read_matrix_trans_format::getOperationName(),
                MLS_DS_FLAGS_PACK(2, 2, 1, 0), {0, 1024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}
      }
    },

    // matrix_load_64x16_b16: AN/BT(transpose = 0), interleave 0,  mnk required elems tile: [64, 64, 16] -> load tile: [64, 64, 16] -> lds tile: [64, 64, 16]
    {{64, 16, false, 16, MlsElemBitTyKind::None, MlsInterleaveKind::InterleaveNone, 1},
      {{64, 16, ROCDL::hcu_matrix_load_64X16_b16::getOperationName(), {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {32, 16, ROCDL::hcu_ds_read_matrix_format::getOperationName(),
                MLS_DS_FLAGS_PACK(2, 2, 1, 0), {0, 1024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}
      }
    },

    // b8:
    // matrix_load_64x16_b8: AT/BN(transpose = 1), interleave 0,  mnk required elems tile: [16, 16, 64] -> load tile: [16, 16, 64] -> lds tile: [16, 16, 64]
    {{16, 64, true, 8, MlsElemBitTyKind::None, MlsInterleaveKind::InterleaveNone, 1},
      {{16, 64, ROCDL::hcu_matrix_load_64X16_b8::getOperationName(), {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {16, 64, ROCDL::hcu_ds_read_matrix_trans_format::getOperationName(),
                MLS_DS_FLAGS_PACK(1, 3, 1, 0), {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}
      }
    },
    // matrix_load_64x16_b8: AN/BT(transpose = 0), interleave 0,  mnk required elems tile: [64, 64, 16] -> load tile: [64, 64, 16] -> lds tile: [64, 64, 16]
    {{64, 32, false, 8, MlsElemBitTyKind::None, MlsInterleaveKind::InterleaveNone, 1},
      {{64, 16, ROCDL::hcu_matrix_load_64X16_b8::getOperationName(), {0, 1024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {32, 32, ROCDL::hcu_ds_read_matrix_format::getOperationName(),
                MLS_DS_FLAGS_PACK(1, 2, 2, 0), {0, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}
      }
    },
    // matrix_load_128x16_b8: AT/BN(transpose = 1), interleave 0,  mnk required elems tile: [16, 16, 128] -> load tile: [16, 16, 128] -> lds tile: [16, 16, 128]
    {{16, 128, true, 8, MlsElemBitTyKind::None, MlsInterleaveKind::InterleaveNone, 1},
      {{16, 128, ROCDL::hcu_matrix_load_128X16_b8::getOperationName(), {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {16, 64, ROCDL::hcu_ds_read_matrix_trans_format::getOperationName(),
                MLS_DS_FLAGS_PACK(1, 3, 1, 0), {0, 1024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}
      }
    },
    // matrix_load_128x16_b8: AN/BT(transpose = 0), interleave 0,  mnk required elems tile: [128, 128, 32] -> load tile: [128, 128, 32] -> lds tile: [128, 128, 32]
    {{128, 32, false, 8, MlsElemBitTyKind::None, MlsInterleaveKind::InterleaveNone, 1},
      {{128, 16, ROCDL::hcu_matrix_load_128X16_b8::getOperationName(), {0, 1024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {32, 32, ROCDL::hcu_ds_read_matrix_format::getOperationName(),
                MLS_DS_FLAGS_PACK(1, 2, 2, 0), {0, 32, 2048, 2080, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}
      }
    },

    // -------------------------------------- YY and SB --------------------------------------------
    // b16:
    // matrix_load_32x16_b16: AT/BN(transpose = 1), interleave 0,  mnk required elems tile: [16, 16, 32] -> load tile: [16, 16, 32] -> lds tile [16, 16, 32]
    {{16, 32, true, 16, MlsElemBitTyKind::None, MlsInterleaveKind::InterleaveNone, 2},
      {{16, 32, ROCDL::hcu_matrix_load_32X16_b16::getOperationName(), {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {16, 32, ROCDL::hcu_ds_read_matrix_trans_format::getOperationName(),
                MLS_DS_FLAGS_PACK(2, 2, 1, 0), {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}
      }
    },

    // matrix_load_32x16_b16: AN/BT(transpose = 0), interleave 0,  mnk required elems tile: [32, 32, 16] -> load tile: [32, 32, 16] -> lds tile [32, 32, 16]
    {{32, 16, false, 16, MlsElemBitTyKind::None, MlsInterleaveKind::InterleaveNone, 2},
      {{32, 16, ROCDL::hcu_matrix_load_32X16_b16::getOperationName(), {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {32, 16, ROCDL::hcu_ds_read_matrix_format::getOperationName(),
                MLS_DS_FLAGS_PACK(2, 2, 1, 0), {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}
      }
    },

    // matrix_load_64x16_b16: AT/BN(transpose = 1), interleave 0,  mnk required elems tile: [64, 64, 64] -> load tile: [64, 64, 64] -> lds tile [64, 64, 64]
    {{64, 64, true, 16, MlsElemBitTyKind::None, MlsInterleaveKind::InterleaveNone, 2},
      {{16, 64, ROCDL::hcu_matrix_load_64X16_b16::getOperationName(), {0, 1024, 2048, 3072, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {16, 32, ROCDL::hcu_ds_read_matrix_trans_format::getOperationName(),
                MLS_DS_FLAGS_PACK(2, 2, 1, 0), {0, 4096, 1024, 5120, 2048, 6144, 3072, 7168, 0, 0, 0, 0, 0, 0, 0, 0}}
      }
    },

    // matrix_load_64x16_b16: AN/BT(transpose = 0), interleave 0,  mnk required elems tile: [64, 64, 64] -> load tile: [64, 64, 64] -> lds tile [64, 64, 64]
    {{64, 64, false, 16, MlsElemBitTyKind::None, MlsInterleaveKind::InterleaveNone, 2},
      {{64, 16, ROCDL::hcu_matrix_load_64X16_b16::getOperationName(), {0, 1024, 2048, 3072, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {32, 16, ROCDL::hcu_ds_read_matrix_format::getOperationName(),
                MLS_DS_FLAGS_PACK(2, 2, 1, 0), {0, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 0, 0, 0, 0, 0, 0, 0, 0}}
      }
    },

    // matrix_load_64x16_b16: AN/BT, interleave 2 for fp16 VGPR matrix-store
    {{64, 64, false, 16, MlsElemBitTyKind::None, MlsInterleaveKind::Interleave2, 2},
      {{64, 16, ROCDL::hcu_matrix_load_64X16_b16::getOperationName(), {0, 1024, 2048, 3072, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {32, 16, ROCDL::hcu_ds_read_matrix_format::getOperationName(),
                MLS_DS_FLAGS_PACK(2, 2, 1, 1), {0, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 0, 0, 0, 0, 0, 0, 0, 0}}
      }
    },

    // b8:
    // matrix_load_64x16_b8: AT/BN(transpose = 1), interleave 0,  mnk required elems tile: [16, 16, 64] -> load tile: [16, 16, 64] -> lds tile [16, 16, 64]
    {{16, 64, true, 8, MlsElemBitTyKind::None, MlsInterleaveKind::InterleaveNone, 2},
      {{16, 64, ROCDL::hcu_matrix_load_64X16_b8::getOperationName(), {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {16, 64, ROCDL::hcu_ds_read_matrix_trans_format::getOperationName(),
                MLS_DS_FLAGS_PACK(1, 3, 1, 0), {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}
      }
    },
    // matrix_load_64x16_b8: AN/BT(transpose = 0), interleave 0,  mnk required elems tile: [64, 64, 16] -> load tile: [64, 64, 16] -> lds tile [64, 64, 16]
    {{64, 32, false, 8, MlsElemBitTyKind::None, MlsInterleaveKind::InterleaveNone, 2},
      {{64, 16, ROCDL::hcu_matrix_load_64X16_b8::getOperationName(), {0, 1024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {32, 32, ROCDL::hcu_ds_read_matrix_format::getOperationName(),
                MLS_DS_FLAGS_PACK(1, 2, 2, 0), {0, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}
      }
    },
    // matrix_load_128x16_b8: AT/BN(transpose = 1), interleave 0,  mnk required elems tile: [64, 64, 128] -> load tile: [64, 64, 128] -> lds tile [64, 64, 128]
    {{64, 128, true, 8, MlsElemBitTyKind::None, MlsInterleaveKind::InterleaveNone, 2},
      {{16, 128, ROCDL::hcu_matrix_load_128X16_b8::getOperationName(), {0, 1024, 2048, 3072, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {16, 64, ROCDL::hcu_ds_read_matrix_trans_format::getOperationName(),
                MLS_DS_FLAGS_PACK(1, 3, 1, 0), {0, 4096, 1024, 5120, 2048, 6144, 3072, 7168, 0, 0, 0, 0, 0, 0, 0, 0}}
      }
    },
    // matrix_load_128x16_b8: AN/BT(transpose = 0), interleave 0,  mnk required elems tile: [128, 128, 64] -> load tile: [128, 128, 64] -> lds tile [128, 128, 64]
    {{128, 64, false, 8, MlsElemBitTyKind::None, MlsInterleaveKind::InterleaveNone, 2},
      {{128, 16, ROCDL::hcu_matrix_load_128X16_b8::getOperationName(), {0, 1024, 2048, 3072, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {32, 32, ROCDL::hcu_ds_read_matrix_format::getOperationName(),
                MLS_DS_FLAGS_PACK(1, 2, 2, 0), {0, 2048, 32, 2080, 4096, 6144, 4128, 6176, 0, 0, 0, 0, 0, 0, 0, 0}}
      }
    },

    // situation1: U4I4F4(Global 4bit -> Shared 8bit -> DotOperand 8bit)
    // b4 u4 padding:    YY don't have ds_pad_b4 insts, but have ds_pad_b6 insts
    // matrix_load_64x16_u4: AT/BN(transpose = 1), interleave 0,  mnk required elems tile: [16, 16, 64] -> load tile: [16, 16, 32] -> lds tile [16, 16, 64]
    {{16, 64, true, 8, MlsElemBitTyKind::U4, MlsInterleaveKind::InterleaveNone, 2},
      {{16, 64, ROCDL::hcu_matrix_load_64X16_u4::getOperationName(), {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {16, 64, ROCDL::hcu_ds_read_matrix_trans_format::getOperationName(),
                MLS_DS_FLAGS_PACK(1, 3, 1, 0), {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}
      }
    },
    // matrix_load_64x16_u4: AN/BT(transpose = 0), interleave 0,  mnk required elems tile: [64, 64, 16] -> load tile: [32, 32, 16] -> lds tile: [64, 64, 16]
    {{64, 32, false, 8, MlsElemBitTyKind::U4, MlsInterleaveKind::InterleaveNone, 2},
      {{64, 16, ROCDL::hcu_matrix_load_64X16_u4::getOperationName(), {0, 1024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {32, 32, ROCDL::hcu_ds_read_matrix_format::getOperationName(),
                MLS_DS_FLAGS_PACK(1, 2, 2, 0), {0, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}
      }
    },
    // b4 i4 padding:
    // matrix_load_64x16_i4: AT/BN(transpose = 1), interleave 0,  mnk required elems tile: [16, 16, 64] -> load tile: [16, 16, 32] -> lds tile [16, 16, 64]
    {{16, 64, true, 8, MlsElemBitTyKind::I4, MlsInterleaveKind::InterleaveNone, 2},
      {{16, 64, ROCDL::hcu_matrix_load_64X16_i4::getOperationName(), {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {16, 64, ROCDL::hcu_ds_read_matrix_trans_format::getOperationName(),
                MLS_DS_FLAGS_PACK(1, 3, 1, 0), {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}
      }
    },
    // matrix_load_64x16_i4: AN/BT(transpose = 0), interleave 0,  mnk required elems tile: [64, 64, 16] -> load tile: [32, 32, 16] -> lds tile: [64, 64, 16]
    {{64, 32, false, 8, MlsElemBitTyKind::I4, MlsInterleaveKind::InterleaveNone, 2},
      {{64, 16, ROCDL::hcu_matrix_load_64X16_i4::getOperationName(), {0, 1024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {32, 32, ROCDL::hcu_ds_read_matrix_format::getOperationName(),
                MLS_DS_FLAGS_PACK(1, 2, 2, 0), {0, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}
      }
    },
    // b4 f4 padding:
    // matrix_load_64x16_fp4: AT/BN(transpose = 1), interleave 0,  mnk required elems tile: [16, 16, 64] -> load tile: [16, 16, 32] -> lds tile [16, 16, 64]
    {{16, 64, true, 8, MlsElemBitTyKind::F4, MlsInterleaveKind::InterleaveNone, 2},
      {{16, 64, ROCDL::hcu_matrix_load_64X16_fp4::getOperationName(), {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {16, 64, ROCDL::hcu_ds_read_matrix_trans_format::getOperationName(),
                MLS_DS_FLAGS_PACK(1, 3, 1, 0), {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}
      }
    },
    // matrix_load_64x16_fp4: AN/BT(transpose = 0), interleave 0,  mnk required elems tile: [64, 64, 16] -> load tile: [32, 32, 16] -> lds tile: [64, 64, 16]
    {{64, 32, false, 8, MlsElemBitTyKind::F4, MlsInterleaveKind::InterleaveNone, 2},
      {{64, 16, ROCDL::hcu_matrix_load_64X16_fp4::getOperationName(), {0, 1024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {32, 32, ROCDL::hcu_ds_read_matrix_format::getOperationName(),
                MLS_DS_FLAGS_PACK(1, 2, 2, 0), {0, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}
      }
    },
    // b6 f6 padding:
    // matrix_load_64x16_b6: AT/BN(transpose = 1), interleave 0,  mnk required elems tile: [16, 16, 64] -> load tile: [16, 16, 64?] -> lds tile [16, 16, 64]
    {{32, 64, true, 8, MlsElemBitTyKind::F6, MlsInterleaveKind::InterleaveNone, 2},
      {{16, 64, ROCDL::hcu_matrix_load_64X16_b6::getOperationName(), {0, 1024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {32, 32, ROCDL::hcu_ds_read_matrix_trans_padbyte::getOperationName(),
                MLS_DS_FLAGS_PACK(1, 0, 0, 0), {0, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}
      }
    },
    // matrix_load_64x16_b6: AN/BT(transpose = 0), interleave 0,  mnk required elems tile: [64, 64, 32] -> load tile: [64, 64, 32?] -> lds tile: [64, 64, 32]
    {{64, 32, false, 8, MlsElemBitTyKind::F6, MlsInterleaveKind::InterleaveNone, 2},
      {{64, 16, ROCDL::hcu_matrix_load_64X16_b6::getOperationName(), {0, 1024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {32, 32, ROCDL::hcu_ds_read_matrix_padbyte::getOperationName(),
                MLS_DS_FLAGS_PACK(1, 0, 0, 0), {0, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}
      }
    },

    // -------------------------------------- SB ---------------------------------------------------
    // situation2: B4Mix(Global 4bit -> Shared 4bit -> DotOperand 8bit)
    // b4 f4 ds_read padding for scaled_f8f6f4xf8f6f4:
    // matrix_load_128x16_b4: AT/BN(transpose = 1), interleave 0,  mnk required elems tile: [32, 32, 128] -> load tile: [32, 32, 64] -> lds tile: [32, 32, 64]
    {{32, 128, true, 8, MlsElemBitTyKind::B4Mix, MlsInterleaveKind::InterleaveNone, 3},
      {{16, 128, ROCDL::hcu_matrix_load_128X16_b4::getOperationName(), {0, 1024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {32, 32, ROCDL::hcu_ds_read_matrix_trans_padbyte::getOperationName(),
                MLS_DS_FLAGS_PACK(0, 0, 0, 0), {0, 16, 32, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}
      }
    },
    // matrix_load_128x16_b4: AN/BT(transpose = 0), interleave 0,  mnk required elems tile: [128, 128, 32] -> load tile: [64, 64, 32] -> lds tile: [64, 64, 32]
    {{128, 32, false, 8, MlsElemBitTyKind::B4Mix, MlsInterleaveKind::InterleaveNone, 3},
      {{128, 16, ROCDL::hcu_matrix_load_128X16_b4::getOperationName(), {0, 1024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {32, 32, ROCDL::hcu_ds_read_matrix_padbyte::getOperationName(),
                MLS_DS_FLAGS_PACK(0, 0, 0, 0), {0, 16, 32, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}
      }
    },
    // matrix_load_256x16_b4: AT/BN(transpose = 1), interleave 0,  mnk required elems tile: [64, 64, 256] -> load tile: [64, 64, 128] -> lds tile: [64, 64, 128]
    {{64, 256, true, 8, MlsElemBitTyKind::B4Mix, MlsInterleaveKind::InterleaveNone, 3},
      {{16, 256, ROCDL::hcu_matrix_load_256X16_b4::getOperationName(), {0, 1024, 2048, 3072, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {32, 32, ROCDL::hcu_ds_read_matrix_trans_padbyte::getOperationName(),
                MLS_DS_FLAGS_PACK(0, 0, 0, 0), {0, 16, 32, 48, 4096, 4112, 4128, 4144, 2048, 2064, 2080, 2096, 6144, 6160, 6176, 6192}}
      }
    },
    // matrix_load_256x16_b4: AN/BT(transpose = 0), interleave 0,  mnk required elems tile: [256, 256, 64] -> load tile: [128, 128, 64] -> lds tile: [128, 128, 64]
    {{256, 64, false, 8, MlsElemBitTyKind::B4Mix, MlsInterleaveKind::InterleaveNone, 3},
      {{256, 16, ROCDL::hcu_matrix_load_256X16_b4::getOperationName(), {0, 1024, 2048, 3072, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {32, 32, ROCDL::hcu_ds_read_matrix_padbyte::getOperationName(),
                MLS_DS_FLAGS_PACK(0, 0, 0, 0), {0, 16, 32, 48, 4096, 4112, 4128, 4144, 2048, 2064, 2080, 2096, 6144, 6160, 6176, 6192}}
      }
    },
    // TODO: add i4, u4 if need

    // situation3: B4(Global 4bit -> Shared 4bit -> DotOperand 4bit)
    // b4 ds_read packed for scaled_f4xf4:
    // matrix_load_128x16_b4: AT/BN(transpose = 1), interleave 0,  mnk required elems tile: [32, 32, 128] -> load tile: [32, 32, 64] -> lds tile: [32, 32, 64]
    {{32, 128, true, 8, MlsElemBitTyKind::B4, MlsInterleaveKind::InterleaveNone, 3},
      {{16, 128, ROCDL::hcu_matrix_load_128X16_b4::getOperationName(), {0, 1024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {32, 64, ROCDL::hcu_ds_read_matrix_trans_format::getOperationName(),
                MLS_DS_FLAGS_PACK(0, 3, 2, 0), {0, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}
      }
    },
    // matrix_load_128x16_b4: AN/BT(transpose = 0), interleave 0,  mnk required elems tile: [128, 128, 32] -> load tile: [64, 64, 32] -> lds tile: [64, 64, 32]
    {{128, 32, false, 8, MlsElemBitTyKind::B4, MlsInterleaveKind::InterleaveNone, 3},
      {{128, 16, ROCDL::hcu_matrix_load_128X16_b4::getOperationName(), {0, 1024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {64, 32, ROCDL::hcu_ds_read_matrix_format::getOperationName(),
                MLS_DS_FLAGS_PACK(0, 3, 2, 0), {0, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}
      }
    },
    // matrix_load_256x16_b4: AT/BN(transpose = 1), interleave 0,  mnk required elems tile: [64, 64, 256] -> load tile: [64, 64, 128] -> lds tile: [64, 64, 128]
    {{64, 256, true, 8, MlsElemBitTyKind::B4, MlsInterleaveKind::InterleaveNone, 3},
      {{16, 256, ROCDL::hcu_matrix_load_256X16_b4::getOperationName(), {0, 1024, 2048, 3072, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {32, 64, ROCDL::hcu_ds_read_matrix_trans_format::getOperationName(),
                MLS_DS_FLAGS_PACK(0, 3, 2, 0), {0, 32, 4096, 4128, 2048, 2080, 6144, 6176, 0, 0, 0, 0, 0, 0, 0, 0}}
      }
    },
    // matrix_load_256x16_b4: AN/BT(transpose = 0), interleave 0,  mnk required elems tile: [256, 256, 64] -> load tile: [128, 128, 64] -> lds tile: [128, 128, 64]
    {{256, 64, false, 8, MlsElemBitTyKind::B4, MlsInterleaveKind::InterleaveNone, 3},
      {{256, 16, ROCDL::hcu_matrix_load_256X16_b4::getOperationName(), {0, 1024, 2048, 3072, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {64, 32, ROCDL::hcu_ds_read_matrix_format::getOperationName(),
                MLS_DS_FLAGS_PACK(0, 3, 2, 0), {0, 32, 4096, 4128, 2048, 2080, 6144, 6176, 0, 0, 0, 0, 0, 0, 0, 0}}
      }
    },
  };

  return MlsInsnMap;
};


SmallVector<std::pair<unsigned, unsigned>> MlsInsn::getMlsElemsTileCandidates(
                                                       unsigned opIdx,
                                                       bool kMajor,
                                                       unsigned elemBitWidth,
                                                       MlsElemBitTyKind elemBitTyKind,
                                                       unsigned mlsVersion,
                                                       MlsInterleaveKind altKind) {
  bool transpose = kMajor;
  const auto &MlsInsnMap = getMlsInsnGroupAttrMap();

  llvm::SmallSetVector<std::pair<unsigned, unsigned>, 8> candidatesSet;
  for (const auto &entry : MlsInsnMap) {
    const MlsInsnGroupSelectKey &key = entry.first;
    bool versionMatched = forEachCompatibleMlsVersion(
        mlsVersion, [&](unsigned compatibleVersion) {
          return key.version == compatibleVersion;
        });

    if (key.transpose == transpose &&
        key.elemBitWidth == elemBitWidth &&
        key.elemBitTyKind == elemBitTyKind &&
        versionMatched &&
        key.alt2Kind == altKind) {
      candidatesSet.insert({key.nonKTile, key.kTile});
    }
  }

  // Convert to SmallVector and sort by nonKTile first, then by kTile
  SmallVector<std::pair<unsigned, unsigned>> result(candidatesSet.begin(), candidatesSet.end());
  llvm::sort(result, [](const std::pair<unsigned, unsigned> &a,
                        const std::pair<unsigned, unsigned> &b) {
    if (a.first != b.first)
      return a.first < b.first;  // Sort by nonKTile ascending
    return a.second < b.second;  // Then by kTile ascending
  });

  return result;
}

FailureOr<MlsInsn> MlsInsn::selectOrGetMlsInsn(unsigned nonKTile,
                                               unsigned kTile,
                                               unsigned elemBitWidth,
                                               MlsElemBitTyKind elemBitTyKind,
                                               unsigned opIdx, bool kMajor,
                                               unsigned mlsVersion,
                                               MlsInterleaveKind altKind) {
  bool transpose = kMajor;  // HCU if not AN/BT, set transpose to true.
  const auto &MlsInsnAttrMap = getMlsInsnGroupAttrMap();
  auto it = MlsInsnAttrMap.end();
  bool found = forEachCompatibleMlsVersion(
      mlsVersion, [&](unsigned compatibleVersion) {
        MlsInsnGroupSelectKey key = {
            nonKTile, kTile, transpose, elemBitWidth,
            elemBitTyKind, altKind,
            compatibleVersion};
        it = MlsInsnAttrMap.find(key);
        return it != MlsInsnAttrMap.end();
      });
  if (!found)
    return failure();

  // fill the MlsInsnAttr
  std::array<unsigned, 2> order = opIdx == 0 ? (kMajor ? std::array<unsigned, 2>{1, 0}
                                                       : std::array<unsigned, 2>{0, 1})
                                             : (kMajor ? std::array<unsigned, 2>{0, 1}
                                                       : std::array<unsigned, 2>{1, 0});
  iMlsInsnAttr iAttr = it->second;

  MlsInsnAttr oAttr;

  oAttr.opIdx = opIdx;
  oAttr.mlsTile = opIdx == 0 ? std::array<unsigned, 2>{nonKTile, kTile}
                             : std::array<unsigned, 2>{kTile, nonKTile};
  oAttr.kMajor = kMajor;
  oAttr.elemBitWidth = elemBitWidth;
  oAttr.elemBitTyKind = elemBitTyKind;
  oAttr.interleaveKind = altKind;
  oAttr.mlsVersion = mlsVersion;
  assert(oAttr.interleaveKind == static_cast<MlsInterleaveKind>(MLS_DS_FLAGS_GET_ALT2(iAttr.dsInfo.flags)));

  if (opIdx == 0) {
    oAttr.mlInsn.instrShape   = {iAttr.mlInfo.instNonKDim, iAttr.mlInfo.instKDim};
    oAttr.mlInsn.instrsPerWarp= {nonKTile/iAttr.mlInfo.instNonKDim, kTile/iAttr.mlInfo.instKDim};
    oAttr.mlInsn.instrOrder   = order;
    oAttr.mlInsn.insn         = iAttr.mlInfo.insn;
    oAttr.mlInsn.dsByteOffsets= iAttr.mlInfo.dsByteOffsets;

    oAttr.dsInsn.instrShape   = {iAttr.dsInfo.instNonKDim, iAttr.dsInfo.instKDim};
    oAttr.dsInsn.instrsPerWarp= {nonKTile/iAttr.dsInfo.instNonKDim, kTile/iAttr.dsInfo.instKDim};
    oAttr.dsInsn.instrOrder   = order;  // ds_read_matrix instrOrder is always for tensor order.
    oAttr.dsInsn.insn         = iAttr.dsInfo.insn;
    oAttr.dsInsn.flags        = iAttr.dsInfo.flags;
    oAttr.dsInsn.dsByteOffsets= iAttr.dsInfo.dsByteOffsets;
  } else {
    oAttr.mlInsn.instrShape   = {iAttr.mlInfo.instKDim, iAttr.mlInfo.instNonKDim};
    oAttr.mlInsn.instrsPerWarp= {kTile/iAttr.mlInfo.instKDim, nonKTile/iAttr.mlInfo.instNonKDim};
    oAttr.mlInsn.instrOrder   = order;
    oAttr.mlInsn.insn         = iAttr.mlInfo.insn;
    oAttr.mlInsn.dsByteOffsets= iAttr.mlInfo.dsByteOffsets;

    oAttr.dsInsn.instrShape   = {iAttr.dsInfo.instKDim, iAttr.dsInfo.instNonKDim};
    oAttr.dsInsn.instrsPerWarp= {kTile/iAttr.dsInfo.instKDim, nonKTile/iAttr.dsInfo.instNonKDim};
    oAttr.dsInsn.instrOrder   = order;
    oAttr.dsInsn.insn         = iAttr.dsInfo.insn;
    oAttr.dsInsn.flags        = iAttr.dsInfo.flags;
    oAttr.dsInsn.dsByteOffsets= iAttr.dsInfo.dsByteOffsets;
  }

  oAttr.msInsn = {};

  // Note: current matrix_load tile seems only exist [1, 1], [xxx, 1], [1, xxx] and no [xxx, xxx] case
  // so cur instrOrder set to tensor order now.
  assert(oAttr.mlInsn.instrsPerWarp[0] == 1 || oAttr.mlInsn.instrsPerWarp[1] == 1);

  unsigned dsInstsPerWarp = oAttr.dsInsn.instrsPerWarp[0] * oAttr.dsInsn.instrsPerWarp[1];
  unsigned mlInstsPerWarp = oAttr.mlInsn.instrsPerWarp[0] * oAttr.mlInsn.instrsPerWarp[1];
  assert(dsInstsPerWarp <= MLS_INST_DS_CNT_MAX && mlInstsPerWarp <= MLS_INST_DS_CNT_MAX);

  return MlsInsn(oAttr);
}

FailureOr<MlsInsn> MlsInsn::selectOrGetMatrixStoreInsn(
    unsigned mTile, unsigned nTile, unsigned elemBitWidth,
    unsigned mlsVersion, MlsInterleaveKind interleaveKind) {
  // VGPR matrix-store support is currently available only for the verified
  // fp16 32x16 form used by MMAC C layout 3 with interleave 2.
  if (interleaveKind != MlsInterleaveKind::Interleave2 ||
      elemBitWidth != 16 || mTile != 32 || nTile != 16 ||
      (mlsVersion != 2 && mlsVersion != 3))
    return failure();

  MlsInsnAttr attr{};
  attr.opIdx = 0;
  attr.mlsTile = {mTile, nTile};
  attr.kMajor = false;
  attr.elemBitWidth = elemBitWidth;
  attr.elemBitTyKind = MlsElemBitTyKind::None;
  attr.interleaveKind = interleaveKind;
  attr.mlsVersion = mlsVersion;
  attr.msInsn.instrShape = {32, 16};
  attr.msInsn.instrsPerWarp = {mTile / 32, nTile / 16};
  attr.msInsn.instrOrder = {0, 1};
  attr.msInsn.insn = "rocdl.matrix.store.32x16.b16";
  return MlsInsn(attr);
}


} // namespace mlir::triton::HCU
