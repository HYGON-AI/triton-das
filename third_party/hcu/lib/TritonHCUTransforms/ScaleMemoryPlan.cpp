// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include "TritonHCU/ScaleMemoryPlan.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include <array>
#include <tuple>

namespace mlir::triton::HCU {

// Check a fixed candidate against both the LDS source and its MMAC consumer.
// sourceOrder=1 means s-contiguous [R,S]; sourceOrder=0 means R-contiguous.
// instScaleGroups=instK/32 counts adjacent scales read by one MMAC: K64 needs
// two rows, so a copy instTile cannot split that pair. instTilesPerWarp/warps
// describe MMAC ownership along R, not which waves execute ds2buf.
//
// Check every R16 group and every K step. Each compute wave's row-base delta
// must preserve 8-row m0 alignment; bank polarity must depend only on K, not
// M/N pairing; wave base + static window + immediate must reproduce getRow().
// For example, relative row 34 uses window +32 and immediate 2. A two-row
// MMAC can read offsets 2/3, but cannot read a pair across offsets 31/32.
// Return the first failed constraint, or nullptr when the candidate is legal.
const char *ScaleMemoryPlan::getRejectionReason(RankedTensorType type,
                                                int64_t instScaleGroups,
                                                int64_t instTilesPerWarp,
                                                int64_t warps,
                                                int64_t sourceOrder) const {
  int64_t r = type.getShape()[0], s = type.getShape()[1];
  if (sourceOrder == 0 && b != 1 && r > 1)
    return "R-contiguous LDS does not place K scales in one per-lane copy span";
  if (s % b != 0)
    return "compact LDS does not align each source to the per-lane copy width";
  if (s % instScaleGroups != 0 || instCols < instScaleGroups ||
      instCols % instScaleGroups != 0)
    return "MMAC K groups are not adjacent within a copy instTile";
  if (rows > 512)
    return "one operand alone exceeds the 512-row scale buffer";

  int64_t activeWaves =
      std::min<int64_t>(warps, llvm::divideCeil(r, 16 * instTilesPerWarp));
  for (int64_t group = 0; group < llvm::divideCeil(r, 16); ++group) {
    int64_t wave = (group / instTilesPerWarp) % activeWaves;
    int64_t staticR = 16 * (group - wave * instTilesPerWarp);
    int64_t firstRow =
        getRow(wave * 16 * instTilesPerWarp, 0, s) - getRow(0, 0, s);
    // A wave-dependent residue cannot be encoded as a static immediate.
    if (firstRow % 8 != 0)
      return "compute-wave first row needs a varying m0 immediate";
    for (int64_t k = 0; k < s; k += instScaleGroups) {
      int64_t relative = getRow(staticR, k, s);
      int64_t address = getRow(16 * group, k, s);
      // Bank choice must depend only on K, never on the M/N group. Opposite
      // descriptor polarities then separate A/B for every MMAC pairing.
      if (((address >> 2) & 1) != (((k / 4) & 1) ^ bankXor))
        return "MMAC scale bank depends on the M/N group";
      int64_t window = firstRow + (relative / 32) * 32;
      int64_t offset = relative % 32;
      bool valid = offset + instScaleGroups <= 32 && window + offset == address;
      for (int64_t j = 1; j < instScaleGroups; ++j)
        valid &= getRow(16 * group, k + j, s) == address + j;
      if (!valid)
        return "actual MMAC rows do not fit a common static window";
    }
  }
  return nullptr;
}

namespace {

struct Candidate {
  ScaleMemoryPlan plan;
  // Summed 16-lane phase service bounds; these are not cycle estimates.
  int64_t ldsReadLowerBound = 0;
  int64_t ldsReadUpperBound = 0;
};

// Estimate LDS bank service using the source mapping used by scale-write
// lowering. Each copy has four 16-lane phases; each lane reads b bytes.
// Clamp edge coordinates as lowering does: padded lanes can repeat addresses.
//
// Bank=(byteOffset/4)%64. For each phase, count distinct dwords and lane
// requests per bank, take the busiest bank, then sum over all phases/copies.
// For example, 16 lanes requesting one dword give bounds [1,16]: the lower
// bound assumes combining; the upper bound counts every request separately.
// Sixteen different dwords in one bank give [16,16]. These are service counts,
// not cycles; opcode throughput and inter-phase overlap are not modeled.
void computeLdsReadBounds(Candidate &candidate, RankedTensorType type,
                        int64_t sourceOrder) {
  const auto &plan = candidate.plan;
  int64_t r = type.getShape()[0], s = type.getShape()[1];
  int64_t instTilesAlongS = plan.getInstsAlongS(s);
  for (int64_t instTile = 0; instTile < plan.numInsts; ++instTile) {
    for (int64_t phase = 0; phase < 4; ++phase) {
      // There are at most 16 source dwords in a phase. One small set suffices
      // to count distinct words across all banks; repeated words share a bank.
      llvm::SmallDenseSet<int64_t, 16> words;
      std::array<int64_t, 64> distinctWords{}, requests{};
      for (int64_t col = 0; col < 16; ++col) {
        int64_t mn = (instTile / instTilesAlongS) * (16 * plan.g) +
                     (phase / plan.p) * 16 + col;
        int64_t sk = (instTile % instTilesAlongS) * plan.instCols +
                     (phase % plan.p) * plan.b;
        mn = std::min(mn, r - 1);
        sk = std::min(sk, s - plan.b);
        int64_t byte = sourceOrder == 1 ? mn * s + sk : sk * r + mn;
        int64_t word = byte / 4;
        if (words.insert(word).second)
          distinctWords[word % 64]++;
        requests[word % 64]++;
      }
      // Distinct words must be served. Same-word DS2BUF combining, including
      // clamped b=4 addresses, is unknown, so retain a per-lane upper bound.
      candidate.ldsReadLowerBound +=
          *std::max_element(distinctWords.begin(), distinctWords.end());
      candidate.ldsReadUpperBound +=
          *std::max_element(requests.begin(), requests.end());
    }
  }
}

} // namespace

// Select ds2buf geometry after the producer's compact LDS layout is fixed.
// Enumerate at most nine (b,g) pairs, reject incompatible copy/MMAC mappings,
// then estimate LDS service for the survivors. All calculations are compile
// time; this search generates no runtime instructions.
//
// Within the SAME mode b, discard a candidate if another one's upper bound
// is below its lower bound, or equal with a better tieCost. For example,
// [4,8] proves better than [12,16]; overlapping [4,12] and [8,16] do not.
// Different modes have no calibrated common cycle cost, so their service
// bounds are not compared directly. Among survivors prefer fewer copies,
// fewer reserved rows, wider b, then smaller g.
//
// Example candidate: [R,S]=[64,8], b=4/g=2 gives p=2, instTile=[32,8],
// two copies and 32 reserved rows. Whether it wins still depends on the LDS
// order, MMAC ownership and other legal candidates. Return nullopt for invalid
// inputs or if no candidate passes the checks.
std::optional<ScaleMemoryPlan> ScaleMemoryPlan::select(RankedTensorType type,
                                                       int64_t instScaleGroups,
                                                       int64_t instTilesPerWarp,
                                                       int64_t warps,
                                                       int64_t sourceOrder) {
  SmallVector<Candidate> candidates;
  if (!ScaleMemoryPlan::isSupportedType(type) || instScaleGroups <= 0 ||
      instTilesPerWarp <= 0 || warps <= 0 ||
      (sourceOrder != 0 && sourceOrder != 1))
    return std::nullopt;

  for (int64_t b : {1, 2, 4}) {
    for (int64_t g : {1, 2, 4}) {
      Candidate candidate{create(type.getShape(), b, g)};
      if (candidate.plan.getRejectionReason(
              type, instScaleGroups, instTilesPerWarp, warps, sourceOrder))
        continue;
      computeLdsReadBounds(candidate, type, sourceOrder);
      candidates.push_back(std::move(candidate));
    }
  }

  // Lexicographic preference, separate from LDS service bounds. Wider b wins
  // only after copy count and reserved row count are equal.
  auto tieCost = [&](unsigned index) {
    const auto &plan = candidates[index].plan;
    return std::tuple(plan.numInsts, plan.rows, -plan.b, plan.g);
  };
  unsigned selected = candidates.size();
  for (unsigned i = 0; i < candidates.size(); ++i) {
    const auto &candidate = candidates[i];
    bool dominated = false;
    for (unsigned j = 0; j < candidates.size(); ++j) {
      const auto &other = candidates[j];
      // Different modes have no common cycle unit without calibrated opcode
      // throughput. Only eliminate proven worse candidates in the same mode.
      if (i == j || other.plan.b != candidate.plan.b)
        continue;
      if (other.ldsReadUpperBound < candidate.ldsReadLowerBound ||
          (other.ldsReadUpperBound == candidate.ldsReadLowerBound &&
           tieCost(j) < tieCost(i))) {
        dominated = true;
        break;
      }
    }
    if (!dominated &&
        (selected == candidates.size() || tieCost(i) < tieCost(selected)))
      selected = i;
  }
  if (selected == candidates.size())
    return std::nullopt;
  return candidates[selected].plan;
}

} // namespace mlir::triton::HCU
