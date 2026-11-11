// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#ifndef TRITON_HCU_LDS_READ_ACCESS_PATTERNS_H
#define TRITON_HCU_LDS_READ_ACCESS_PATTERNS_H

#include "TritonHCU/LdsReadModel.h"
#include <cassert>
#include <vector>

namespace mlir::triton::HCU {

struct LdsReadAccess {
  unsigned bytesPerLane;
  // One logical starting byte per lane. Matrix reads describe source address
  // contributors, NOT the lanes receiving the permuted register results.
  std::vector<unsigned> logicalBytes;
  std::vector<std::vector<unsigned>> phases;
  // Compatibility for the existing wave64 profiles. New target profiles must
  // supply their broadcast rule explicitly rather than infer it from dtype.
  LdsBroadcast broadcast = LdsBroadcast::SameBankWord;
};

inline std::vector<std::vector<unsigned>>
consecutiveLdsPhases(unsigned waveSize, unsigned lanesPerPhase) {
  std::vector<std::vector<unsigned>> result;
  if (!lanesPerPhase || waveSize % lanesPerPhase)
    return result;
  for (unsigned start = 0; start < waveSize; start += lanesPerPhase) {
    result.emplace_back();
    for (unsigned lane = start; lane < start + lanesPerPhase; ++lane)
      result.back().push_back(lane);
  }
  return result;
}

// Wave64 DS_READ_B128 phases: the 32-bank profile follows CK; the 64-bank
// HCU profile uses four independently verified 16-lane groups.
// Not a default for arbitrary architectures or matrix-read instructions.
inline std::vector<std::vector<unsigned>>
readB128PhasesWave64(unsigned banks = 32) {
  if (banks == 64)
    return {{0, 1, 2, 3, 8, 9, 10, 11, 20, 21, 22, 23, 28, 29, 30, 31},
            {32, 33, 34, 35, 40, 41, 42, 43, 52, 53, 54, 55, 60, 61, 62, 63},
            {4, 5, 6, 7, 12, 13, 14, 15, 16, 17, 18, 19, 24, 25, 26, 27},
            {36, 37, 38, 39, 44, 45, 46, 47, 48, 49, 50, 51, 56, 57, 58, 59}};
  return {{0, 1, 2, 3, 20, 21, 22, 23},     {4, 5, 6, 7, 16, 17, 18, 19},
          {8, 9, 10, 11, 28, 29, 30, 31},   {12, 13, 14, 15, 24, 25, 26, 27},
          {32, 33, 34, 35, 52, 53, 54, 55}, {36, 37, 38, 39, 48, 49, 50, 51},
          {40, 41, 42, 43, 60, 61, 62, 63}, {44, 45, 46, 47, 56, 57, 58, 59}};
}

// Ordinary MMAC operand: kWidth consecutive elements in each lane; nonK
// occupies the low lane bits (mfmaDotToLinearLayout). Repetitions enumerate
// every non-K tile, independent of its distribution over consumer waves.
inline std::vector<LdsReadAccess>
contiguousMmacReads(unsigned nonK, unsigned k, unsigned elementBytes,
                    unsigned instructionNonK, unsigned kWidth,
                    unsigned waveSize,
                    const std::vector<std::vector<unsigned>> &phases,
                    unsigned maxReadBytes = 0) {
  std::vector<LdsReadAccess> reads;
  unsigned fragmentBytes = kWidth * elementBytes;
  unsigned readBytes = maxReadBytes ? std::min(fragmentBytes, maxReadBytes)
                                    : fragmentBytes;
  if (!instructionNonK || !kWidth || waveSize % instructionNonK ||
      !readBytes || readBytes < elementBytes || fragmentBytes % readBytes ||
      nonK % instructionNonK || k % ((waveSize / instructionNonK) * kWidth))
    return reads;
  for (unsigned mn = 0; mn < nonK; mn += instructionNonK)
    for (unsigned kt = 0; kt < k; kt += waveSize / instructionNonK * kWidth)
     for (unsigned part = 0; part < fragmentBytes; part += readBytes) {
      LdsReadAccess read{readBytes, {}, phases};
      for (unsigned lane = 0; lane < waveSize; ++lane)
        read.logicalBytes.push_back(((mn + lane % instructionNonK) * k + kt +
                                     lane / instructionNonK * kWidth) *
                                    elementBytes + part);
      reads.push_back(std::move(read));
    }
  return reads;
}

// HCU b16 matrix read source-address phases. Keep this target-specific table
// separate from the ordinary B128 table: the destination register permutation
// is not the memory issue grouping.
inline std::vector<std::vector<unsigned>>
matrixB16PhasesWave64(unsigned banks = 32) {
  if (banks == 64)
    return {{0, 1, 4, 5, 8, 9, 12, 13, 18, 19, 22, 23, 26, 27, 30, 31},
            {32, 33, 36, 37, 40, 41, 44, 45, 50, 51, 54, 55, 58, 59, 62, 63},
            {2, 3, 6, 7, 10, 11, 14, 15, 16, 17, 20, 21, 24, 25, 28, 29},
            {34, 35, 38, 39, 42, 43, 46, 47, 48, 49, 52, 53, 56, 57, 60, 61}};
  std::vector<std::vector<unsigned>> phases;
  for (unsigned shift : {0u, 2u, 8u, 10u, 32u, 34u, 40u, 42u}) {
    phases.emplace_back();
    for (unsigned lane : {0u, 4u, 1u, 5u, 18u, 22u, 19u, 23u})
      phases.back().push_back(lane ^ shift);
  }
  return phases;
}

inline std::vector<LdsReadAccess>
matrixB16Reads(unsigned k, unsigned nonK,
               const std::vector<std::vector<unsigned>> &phases) {
  std::vector<LdsReadAccess> reads;
  const auto &inst = matrixReadInstruction(MatrixReadOpcode::M32x16B16);
  if (!k || !nonK || k % inst.columns || nonK % inst.rows)
    return reads;
  for (unsigned kt = 0; kt < k; kt += inst.columns)
    for (unsigned panel = 0; panel < nonK; panel += inst.rows) {
      LdsReadAccess read{16, {}, phases};
      for (unsigned lane = 0; lane < 64; ++lane)
        read.logicalBytes.push_back(
            ((kt + lane / 4) * nonK + panel + lane % 4 * 8) * 2);
      reads.push_back(std::move(read));
    }
  return reads;
}

inline std::vector<LdsReadAccess> matrixB16Reads(unsigned k, unsigned nonK) {
  return matrixB16Reads(k, nonK, matrixB16PhasesWave64());
}

inline std::vector<std::vector<unsigned>>
matrixB8PhasesWave64(unsigned banks = 32) {
  if (banks == 64)
    return {{0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30},
            {32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62},
            {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31},
            {33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63}};
  std::vector<std::vector<unsigned>> phases;
  for (unsigned shift : {0u, 1u, 8u, 9u, 32u, 33u, 40u, 41u}) {
    phases.emplace_back();
    for (unsigned lane : {0u, 2u, 4u, 6u, 17u, 19u, 21u, 23u})
      phases.back().push_back(lane ^ shift);
  }
  return phases;
}

// Routing and bank arbitration are separate target properties. Pair-collision
// probes on gfx936 and gfx938 validate these lane groups for ordinary b16/b8;
// gfx92a retains that profile; gfx946/gfx948 combine groups for 64 banks.
// Other variants remain gfx936-only or unmodelled. No phase issue ORDER is
// implied. b4 is absent because its conflict-cycle increment differs; lane
// partitions alone are not a sufficient cost model.
inline std::optional<std::vector<std::vector<unsigned>>>
matrixReadBankPhases(std::string_view arch, MatrixReadOpcode opcode) {
  if (!hasLegacyMatrixReadProfile(arch, opcode))
    return std::nullopt;
  unsigned banks = arch == "gfx946" || arch == "gfx948" ? 64 : 32;
  switch (opcode) {
  case MatrixReadOpcode::M32x16B16:
    return matrixB16PhasesWave64(banks);
  case MatrixReadOpcode::M32x32B8:
  case MatrixReadOpcode::M32x16B16Alt:
    return matrixB8PhasesWave64(banks);
  case MatrixReadOpcode::M32x8B32:
    return readB128PhasesWave64(banks);
  default:
    return std::nullopt;
  }
}

inline std::optional<LdsReadTrace>
matrixReadTrace(std::string_view arch, MatrixReadOpcode opcode,
                const std::vector<uint64_t> &sourceBitAddresses) {
  auto phases = matrixReadBankPhases(arch, opcode);
  auto results = matrixReadResults(arch, opcode, sourceBitAddresses);
  if (!phases || !results)
    return std::nullopt;
  LdsReadTrace trace;
  trace.broadcast = LdsBroadcast::SameBankWord;
  trace.results = std::move(*results);
  for (const auto &phase : *phases) {
    trace.phases.emplace_back();
    for (unsigned lane : phase)
      trace.phases.back().push_back({lane, {sourceBitAddresses[lane], 128}});
  }
  return trace;
}

inline std::vector<LdsReadAccess>
matrixB8Reads(unsigned k, unsigned nonK,
              const std::vector<std::vector<unsigned>> &phases) {
  std::vector<LdsReadAccess> reads;
  const auto &inst = matrixReadInstruction(MatrixReadOpcode::M32x32B8);
  if (!k || !nonK || k % inst.columns || nonK % inst.rows)
    return reads;
  for (unsigned kt = 0; kt < k; kt += inst.columns)
    for (unsigned panel = 0; panel < nonK; panel += inst.rows) {
      LdsReadAccess read{16, {}, phases};
      for (unsigned lane = 0; lane < 64; ++lane)
        read.logicalBytes.push_back((kt + lane / 2) * nonK + panel +
                                    lane % 2 * 16);
      reads.push_back(std::move(read));
    }
  return reads;
}

inline std::vector<LdsReadAccess> matrixB8Reads(unsigned k, unsigned nonK) {
  return matrixB8Reads(k, nonK, matrixB8PhasesWave64());
}

// The phase table is an instruction/target property, not a tile-size policy.
// Sub-dword reads use a whole-wave phase. Distinct bytes/halfwords of the
// same bank word are handled as a broadcast by the conflict scorer.
inline std::vector<std::vector<unsigned>>
scalarReadPhases(unsigned bytes, unsigned banks = 32) {
  if (banks != 32 && banks != 64)
    return {};
  switch (bytes) {
  case 1:
  case 2:
    return consecutiveLdsPhases(64, 64);
  case 4:
    return consecutiveLdsPhases(64, banks);
  case 8:
    return consecutiveLdsPhases(64, banks / 2);
  case 16:
    return readB128PhasesWave64(banks);
  default:
    return {};
  }
}

} // namespace mlir::triton::HCU
#endif
