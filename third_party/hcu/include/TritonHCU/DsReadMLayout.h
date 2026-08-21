// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef TRITON_THIRD_PARTY_HCU_INCLUDE_TRITONHCU_DSREADMLAYOUT_H_
#define TRITON_THIRD_PARTY_HCU_INCLUDE_TRITONHCU_DSREADMLAYOUT_H_

#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mlir::triton::HCU {

inline bool isDsReadMEnabled() {
  // Keep ds_read_m enabled by default.  The environment variable is only a
  // debugging escape hatch: 0/false/off disables the optimization.
  std::string value = tools::getStrEnv("TRITON_HCU_ENABLE_DS_READ_M");
  if (value.empty())
    return true;
  if (auto enabled = tools::isEnvValueBool(value))
    return *enabled;
  return true;
}

struct DsReadMLayoutSpec {
  unsigned elementBitWidth;
  unsigned kPerTile;
  unsigned kWidth;
  unsigned vec;
  unsigned perPhase;
  unsigned panelMaxPhase;
};

inline std::optional<DsReadMLayoutSpec>
getDsReadMLayoutSpec(unsigned elementBitWidth) {
  if (elementBitWidth == 16)
    return DsReadMLayoutSpec{/*elementBitWidth=*/16,
                             /*kPerTile=*/16,
                             /*kWidth=*/4,
                             /*vec=*/8,
                             /*perPhase=*/2,
                             /*panelMaxPhase=*/2};
  if (elementBitWidth == 8)
    return DsReadMLayoutSpec{/*elementBitWidth=*/8,
                             /*kPerTile=*/32,
                             /*kWidth=*/8,
                             /*vec=*/16,
                             /*perPhase=*/1,
                             /*panelMaxPhase=*/2};
  return std::nullopt;
}

inline int64_t getDsReadMFlatMaxPhase(const DsReadMLayoutSpec &spec,
                                     int64_t blockN) {
  return spec.elementBitWidth == 8 ? std::min<int64_t>(4, blockN / 16) : 2;
}

inline bool isDsReadMTileShape(int64_t blockK, int64_t blockN,
                              const DsReadMLayoutSpec &spec) {
  return blockK >= spec.kPerTile && blockK % spec.kPerTile == 0 &&
         blockN >= 32 && blockN % 32 == 0;
}

inline bool isDsReadMPanelShape(int64_t blockK, int64_t blockN,
                               const DsReadMLayoutSpec &spec) {
  return isDsReadMTileShape(blockK, blockN, spec) &&
         llvm::isPowerOf2_64(blockK) && llvm::isPowerOf2_64(blockN);
}

// Logical B remains [BK, BN].  Physically, LDS is ordered as
// [BN/32 panels][BK rows][32 columns].  Each N32 panel therefore has the row
// stride expected by ds_read_m.  The K basis carries the XOR used by the
// matching matrix-read address formula:
//   b16: vec=8,  perPhase=2, maxPhase=2
//   b8:  vec=16, perPhase=1, maxPhase=2
// Limiting b8 to two phases is intentional: its N XOR must stay inside one
// physical N32 panel.  Its panel bases additionally XOR panel-id bits [1:0]
// into K bits [1:0], rotating N-contiguous vector stores across all 32 LDS
// banks without padding; the matching read address reverses this skew.
inline gpu::SharedLinearEncodingAttr
getDsReadMPanelEncoding(MLIRContext *context, int64_t blockK, int64_t blockN,
                        unsigned elementBitWidth) {
  auto spec = getDsReadMLayoutSpec(elementBitWidth);
  assert(spec && isDsReadMPanelShape(blockK, blockN, *spec) &&
         "invalid ds_read_m panel shape or element type");

  StringAttr offset = StringAttr::get(context, "offset");
  StringAttr block = StringAttr::get(context, "block");
  SmallVector<StringAttr> outDims = standardOutDimNames(context, 2);
  std::vector<std::vector<int32_t>> bases;
  for (int32_t n = 1; n < 32; n *= 2)
    bases.push_back({0, n});
  for (int32_t k = 1; k < blockK; k *= 2)
    bases.push_back(
        {k, static_cast<int32_t>(spec->vec *
                                 ((k / spec->perPhase) % spec->panelMaxPhase))});
  for (int32_t panelN = 32; panelN < blockN; panelN *= 2) {
    // Consecutive N16 vectors are written by consecutive lanes, so the first
    // four N32 panels would otherwise start on the same eight LDS banks.  For
    // b8, XOR the two low panel-id bits into the physical K row.  This rotates
    // those panel bases by 0/32/64/96 bytes without padding or changing the
    // logical [BK, BN] tensor.  Include the matching N swizzle component so
    // the existing logical-K phase remains unchanged after inversion.
    int32_t panelK = 0;
    if (elementBitWidth == 8 && panelN < 128)
      panelK = panelN / 32;
    int32_t panelCoord =
        panelN + static_cast<int32_t>(
                     spec->vec * ((panelK / spec->perPhase) %
                                  spec->panelMaxPhase));
    bases.push_back({panelK, panelCoord});
  }

  LinearLayout layout({{offset, std::move(bases)}, {block, {}}}, outDims);
  return gpu::SharedLinearEncodingAttr::get(context, layout,
                                             /*layoutAlignment=*/16);
}

inline bool isDsReadMPanelEncoding(gpu::SharedLinearEncodingAttr encoding,
                                   int64_t blockK, int64_t blockN,
                                   unsigned elementBitWidth) {
  auto spec = getDsReadMLayoutSpec(elementBitWidth);
  return encoding && spec && isDsReadMPanelShape(blockK, blockN, *spec) &&
         encoding == getDsReadMPanelEncoding(encoding.getContext(), blockK,
                                             blockN, elementBitWidth);
}

// Recognize the exact rank-2 physical LDS contract rather than the source-level
// GEMM spelling. A multi-buffer pipeline adds a leading allocation dimension;
// its memdesc_index view restores this exact [BK, BN] tile and encoding.
inline bool isDsReadMPanelMemDesc(gpu::MemDescType memDesc) {
  if (!memDesc || !memDesc.getElementType().isIntOrFloat())
    return false;
  auto encoding =
      dyn_cast<gpu::SharedLinearEncodingAttr>(memDesc.getEncoding());
  ArrayRef<int64_t> tileShape = memDesc.getShape();
  if (!encoding || tileShape.size() != 2 ||
      memDesc.getAllocShape().take_back(2) != tileShape)
    return false;
  return isDsReadMPanelEncoding(
      encoding, tileShape[0], tileShape[1],
      memDesc.getElementType().getIntOrFloatBitWidth());
}

} // namespace mlir::triton::HCU

#endif
