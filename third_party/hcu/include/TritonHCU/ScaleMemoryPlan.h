// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#ifndef TRITON_HCU_SCALE_MEMORY_PLAN_H
#define TRITON_HCU_SCALE_MEMORY_PLAN_H

#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "llvm/Support/MathExtras.h"

#include <optional>

namespace mlir::triton::HCU {

// ScaleMemoryPlan is shared by scale-write selection, Scale Buffer allocation
// and MMAC lowering. All three must agree on where each logical scale lands.
// It describes LDS -> Scale Buffer; Global load policy and LDS layout are
// selected by ScalePipeline before selecting this plan.
//
// Distinguish the three tile levels. A and B have separate storage plans:
//
//   CTA/BK tile [BM, BN, BK], S=BK/32
//       |-- A scales [BM,S] -> Plan A -> A ds2buf instTiles --\
//       `-- B scales [BN,S] -> Plan B -> B ds2buf instTiles ---+-->
//                                                    MMAC [16,16,instK]
//
//   One ds2buf instTile = [16*g, b*p] from one operand, written by ONE
//   instruction execution on Wave64. It can supply several MMAC instructions.
//   Each MMAC consumes instK/32 adjacent scales for an M16 group from A and
//   an N16 group from B. Copy-wave ownership and MMAC-wave ownership differ.
//
// instTile in THIS plan means a ds2buf instruction tile. It is neither the
// complete CTA scale tensor nor an MMAC instTile. numInsts counts copies, not
// loop iterations or MMAC instructions. instScaleGroups=instK/32 is the
// consumer requirement; the ds2buf instTile width b*p must contain complete
// instScaleGroups.
//
// Copy-wave distribution is separate from instTile geometry. For W waves
// per CTA, instTileId=waveId+round*W, guarded by instTileId<numInsts:
//
//              wave0        wave1        ...  wave(W-1)
//   round 0    instTile0    instTile1     ...  instTile(W-1)
//   round 1    instTileW    instTile(W+1) ...  instTile(2W-1)
//
// getMaxInstsPerWave(W)=ceil(numInsts/W) is the round count, not uniform
// ownership. A partial final round disables excess waves. W comes from the
// current CTA or WASP consumer partition; it is not duplicated in this plan.
//
// Definitions (all scale elements are one-byte E8M0 values):
//   S[R, S]: R is M or N; S is the number of K/32 scale groups.
//   instTile: source block copied by one execution of ds2buf on a single
//             wave (all 64 lanes), not the full CTA tensor or pipeline stage.
//   b:       bytes read per lane, 1/2/4 for op_ctrl0 mode 0/1/2.
//   g:       M16/N16 groups covered by one instTile, 1/2/4.
//   p = 4/g: 16-lane phases assigned to each M16/N16 group.
//
// Only b selects the instruction mode; g/p describe compiler source mapping.
// Wave64 has four phases. For lane l, phaseId=l/16 and c=l%16. Each phase
// writes rows phaseId*b through phaseId*b+b-1, with column c, relative to
// this instruction's destination start. Together the four phases write one
// contiguous span of 4*b rows; bankXor changes source phases or the start row,
// not the spacing between destination rows:
//
//   lanes      0..15       16..31       32..47       48..63
//          +-----------+-----------+-----------+-----------+
//          | phaseId=0 | phaseId=1 | phaseId=2 | phaseId=3 |
//          +-----------+-----------+-----------+-----------+
//
// Within an instTile, phase phaseId reads 16 rows x b columns, starting at
// (r, s) = (16*(phaseId/p)+c, b*(phaseId%p)).
// This is the bankXor=0 source mapping. For b=2/4, bankXor=1 swaps
// source phaseId with phaseId XOR (4/b), reversing destination row bit2.
// Thus an instTile covers [16*g, b*p]:
//
//                          s -> (p groups of b bytes)
//                      +-------------+-------------+-----+
//   M16/N16 group 0     |   phase 0   |   phase 1   | ... |
//                      +-------------+-------------+-----+
//   M16/N16 group 1     |   phase p   | phase p+1   | ... |
//                      +-------------+-------------+-----+
//             ...      |             |             |     |
//
// This is a LOGICAL-coordinate diagram, independent of LDS order. Source byte
// offsets are r*S+s for order=[1,0], or s*R+r for order=[0,1]. The source is
// compact and unswizzled. Nondegenerate R-contiguous sources require b=1.
// Incomplete edge instTiles clamp source coordinates; MMAC reads only valid
// scales. No LDS padding or out-of-bounds source read is introduced.
//
// In Scale Buffer, each instTile writes 4*b rows with 16 byte columns.
// Before applying the bank permutation, using coordinates LOCAL to this tile:
//
//   logical source [16*g, b*p]       destination [4*b rows, 16 columns]
//        (rLocal, sLocal)  ------>  rawRow = (rLocal/16)*(b*p) + sLocal
//                                  col = rLocal%16
//
// Scale Buffer has two SRAM banks: bank(row)=(row/4)%2. Selected plans require
// bank=((s/4)%2)^bankXor for every valid scale, using the full tensor's s
// coordinate, independently of its M/N group. A/B use opposite bankXor values.
// For b=2/4, relative row=rawRow XOR (4*bankXor);
// copy lowering permutes source phases to match. For b=1, relative row is
// rawRow+4*bank: select either half of the reserved eight-row block.
//
//   reserved rows     0..3       4..7
//   SRAM bank           0          1
//   b=1, s=0..3       A data     B data   (in separate A/B reservations)
//
// Reserved instTile starts are separated by rowStride=alignTo(4*b, 8), in Scale
// Buffer ROWS. Alignment gaps belong to Scale Buffer, not LDS. instTiles are
// numbered along s first, then r. getRow() takes full-tensor (r,s) coordinates
// and returns a row RELATIVE to the allocated region. Late allocation assigns
// the descriptor producer's 8-row-aligned row_base. MMAC lowering splits the
// address into an m0 window base and a 5-bit row offset; it changes windows
// when needed. A whole scale tensor need not fit in a single 32-row window.
//
// Selection enumerates b/g within the fixed LDS layout and rejects plans
// incompatible with copy/MMAC addressing. Within each b, phase read-service
// bounds eliminate provably inferior candidates. Among surviving candidates,
// prefer fewer copies, fewer reserved rows, wider b, then smaller g.
// Bounds are NOT cycles: unknown DS2BUF same-word combining and uncalibrated
// mode throughput prevent an exact cross-mode cycle comparison. Selection
// and its helpers are implemented in ScaleMemoryPlan.cpp.
struct ScaleMemoryPlan {
  int64_t b;         // Bytes per lane: 1/2/4.
  int64_t g;         // M16/N16 groups per instTile: 1/2/4.
  int64_t p;         // Phases per M16/N16 group: 4/g.
  int64_t instCols;  // b*p scale columns.
  int64_t rowStride; // Scale Buffer rows between instTile starts.
  int64_t numInsts;  // Wave64 ds2buf instructions for the full tensor.
  int64_t rows;      // Includes Scale Buffer alignment gaps.
  int64_t bankXor;   // 0/1 SRAM-bank polarity; A/B use opposite values.

  static bool isSupportedType(RankedTensorType type) {
    return type && type.getRank() == 2 && type.hasStaticShape() &&
           type.getElementType().isInteger(8) && type.getShape()[0] > 0 &&
           type.getShape()[1] > 0;
  }

  // Construct candidate geometry for a positive [R,S] shape and b/g in 1/2/4.
  // Consumer compatibility is checked separately by select/getRejectionReason.
  static ScaleMemoryPlan create(ArrayRef<int64_t> shape, int64_t b, int64_t g,
                                int64_t bankXor = 0) {
    int64_t r = shape[0], s = shape[1];
    int64_t p = 4 / g;
    int64_t cols = b * p;
    int64_t insts = llvm::divideCeil(r, 16 * g) * llvm::divideCeil(s, cols);
    int64_t stride = llvm::alignTo(4 * b, 8);
    return {b, g, p, cols, stride, insts, insts * stride, bankXor};
  }

  amdgpu::ScaleEncodingAttr getEncoding(MLIRContext *ctx) const {
    return amdgpu::ScaleEncodingAttr::get(ctx, {16 * g, instCols}, rowStride,
                                          bankXor);
  }

  // Restore fixed geometry from a verified encoding, without selecting a plan.
  static ScaleMemoryPlan fromMemDesc(triton::gpu::MemDescType type) {
    auto layout = cast<amdgpu::ScaleEncodingAttr>(type.getEncoding());
    int64_t g = layout.getBlockShape()[0] / 16;
    int64_t b = layout.getBlockShape()[1] * g / 4;
    return create(type.getShape(), b, g, layout.getBankXor());
  }

  // Select copy geometry within the fixed LDS layout for this consumer.
  // instTilesPerWarp/warps describe MFMA ownership, not copy-wave ownership.
  static std::optional<ScaleMemoryPlan>
  select(RankedTensorType type, int64_t instScaleGroups,
         int64_t instTilesPerWarp, int64_t warps, int64_t sourceOrder = 1);

  // nullptr means the fixed plan is compatible with this source and consumer.
  // The type and consumer dimensions must already be valid and positive.
  const char *getRejectionReason(RankedTensorType type, int64_t instScaleGroups,
                                 int64_t instTilesPerWarp, int64_t warps,
                                 int64_t sourceOrder) const;

  int64_t getCopyMode() const { return llvm::Log2_64(b); }

  int64_t getInstsAlongS(int64_t scaleColumns) const {
    return llvm::divideCeil(scaleColumns, instCols);
  }

  // Some waves are inactive in the last round; this is a maximum, not a
  // claim that all waves execute the same number of copies.
  int64_t getMaxInstsPerWave(int64_t wavesPerCTA) const {
    return llvm::divideCeil(numInsts, wavesPerCTA);
  }

  int64_t getRow(int64_t r, int64_t s, int64_t scaleColumns) const {
    int64_t instTileId =
        (r / (16 * g)) * getInstsAlongS(scaleColumns) + s / instCols;
    int64_t innerRow = ((r / 16) % g) * instCols + s % instCols;
    // Four consecutive rows share one SRAM bank. A byte copy occupies only
    // four rows of its eight-row reservation; wider copies permute phases.
    if (b == 1)
      innerRow += 4 * (((s / 4) & 1) ^ bankXor);
    else
      innerRow ^= 4 * bankXor;
    return instTileId * rowStride + innerRow;
  }
};

} // namespace mlir::triton::HCU
#endif
