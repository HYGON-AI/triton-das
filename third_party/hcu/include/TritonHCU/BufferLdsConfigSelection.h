// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#ifndef TRITON_HCU_BUFFER_LDS_CONFIG_SELECTION_H
#define TRITON_HCU_BUFFER_LDS_CONFIG_SELECTION_H

// Select a copy configuration from the consumer's logical LDS read addresses.
#include "TritonHCU/BufferLdsConfig.h"
#include "TritonHCU/LdsReadAccessPatterns.h"
#include <algorithm>
#include <limits>
#include <tuple>
#include <vector>

namespace mlir::triton::HCU {

inline auto bufferLdsConfigCost(const BufferLdsConfig &copyConfig,
                                const LdsGeometry &hw) {
  unsigned preferredChunk =
      std::min(copyConfig.rows / copyConfig.waves,
               std::max(1u, hw.banks * hw.bankBytes / copyConfig.rowBytes));
  unsigned preferredGroup = std::min(
      copyConfig.waves, std::max(2u, copyConfig.rows / copyConfig.waves));
  unsigned preferredWrap =
      std::min(copyConfig.waves, std::max(2u, preferredChunk));
  auto distance = [](unsigned a, unsigned b) { return a > b ? a - b : b - a; };
  // Bank conflicts take precedence over global coalescing. Keep instruction
  // count ahead of deterministic locality tie-breakers when comparing widths.
  return std::make_tuple(
      copyConfig.worstReadWay, copyConfig.readPhaseCost,
      copyConfig.globalSectorCost,
      copyConfig.tileBytes() / copyConfig.bytesPerWaveTransfer(),
      distance(copyConfig.rowsPerChunk, preferredChunk),
      distance(copyConfig.rowGroupWaves(), preferredGroup),
      distance(copyConfig.wrapCount, preferredWrap), copyConfig.wrapCount,
      copyConfig.wrapStepBytes,
      std::numeric_limits<unsigned>::max() - copyConfig.rowsPerChunk);
}

// Score actual bank words; two lanes reading the same dword are a broadcast.
// Also prove that every read remains contiguous after the producer rotation.
inline bool
evaluateBufferLdsBankConflicts(BufferLdsConfig &copyConfig,
                               const LdsGeometry &hw,
                               const std::vector<LdsReadAccess> &reads) {
  if (!hw.banks || !hw.bankBytes || !hw.waveSize || !copyConfig.rowsPerChunk ||
      !copyConfig.rowBytes || !copyConfig.waves || !copyConfig.wrapCount ||
      !copyConfig.copyBytesPerLane || copyConfig.waveSize != hw.waveSize ||
      !isBufferLdsPowerOfTwo(copyConfig.rowGroupWaves()) ||
      copyConfig.rowGroupWaves() > copyConfig.waves ||
      !copyConfig.totalBytesPerWave())
    return false;
  copyConfig.worstReadWay = 0;
  copyConfig.readPhaseCost = 0;
  std::vector<unsigned> starts(hw.waveSize);
  for (const auto &read : reads) {
    if (!read.bytesPerLane || read.logicalBytes.size() != hw.waveSize ||
        read.phases.empty())
      return false;
    std::vector<bool> seen(hw.waveSize);
    for (unsigned lane = 0; lane < hw.waveSize; ++lane) {
      unsigned logical = read.logicalBytes[lane];
      if (logical >= copyConfig.tileBytes() ||
          read.bytesPerLane > copyConfig.tileBytes() - logical)
        return false;
      starts[lane] = copyConfig.physicalByte(logical);
      if (!copyConfig.isPhysicalRangeContiguous(logical, read.bytesPerLane))
        return false;
    }
    LdsReadTrace trace;
    trace.broadcast = read.broadcast;
    for (const auto &phase : read.phases) {
      if (phase.empty() ||
          uint64_t(phase.size()) * read.bytesPerLane > hw.phaseBytes)
        return false;
      trace.phases.emplace_back();
      for (unsigned lane : phase) {
        if (lane >= hw.waveSize || seen[lane])
          return false;
        seen[lane] = true;
        trace.phases.back().push_back(
            {lane, {uint64_t(starts[lane]) * 8, read.bytesPerLane * 8}});
      }
    }
    if (std::find(seen.begin(), seen.end(), false) != seen.end())
      return false;
    auto cost =
        scoreLdsReadTrace(trace, {hw.banks, hw.bankBytes * 8, hw.waveSize});
    if (!cost)
      return false;
    copyConfig.worstReadWay = std::max(copyConfig.worstReadWay, cost->worstWay);
    copyConfig.readPhaseCost += cost->phaseService;
  }
  return !reads.empty();
}

// Bounded compile-time search over row-group permutations and legal M0 shifts.
// Logical rows are contiguous here; the MLIR adapter must prove global order,
// alignment and mask granularity before converting a producer. A single
// configuration describes both sides of the copy. No kernel shape or A/B identity
// is encoded.
inline std::optional<BufferLdsConfig>
selectBufferLdsConfig(unsigned rows, unsigned rowBytes, unsigned waves,
                      unsigned copyBytesPerLane, const LdsGeometry &hw,
                      const std::vector<LdsReadAccess> &reads,
                      uint64_t globalRowStrideBytes = 0,
                      const BufferLdsWrapCapabilities &wrap = {}) {
  if (!isBufferLdsPowerOfTwo(rows) || !isBufferLdsPowerOfTwo(rowBytes) ||
      !isBufferLdsPowerOfTwo(waves) || !isBufferLdsPowerOfTwo(hw.waveSize) ||
      !isBufferLdsPowerOfTwo(hw.banks) ||
      !isBufferLdsPowerOfTwo(hw.bankBytes) ||
      !isBufferLdsPowerOfTwo(hw.phaseBytes) ||
      !isBufferLdsPowerOfTwo(hw.capacityBytes) ||
      !isBufferLdsPowerOfTwo(wrap.granuleBytes) ||
      (copyBytesPerLane != 4 && copyBytesPerLane != 8 &&
       copyBytesPerLane != 16) ||
      rowBytes % copyBytesPerLane || rows % waves ||
      uint64_t(rows) * rowBytes > hw.capacityBytes ||
      (uint64_t(rows) * rowBytes) %
          (uint64_t(waves) * hw.waveSize * copyBytesPerLane))
    return std::nullopt;
  if (!globalRowStrideBytes)
    globalRowStrideBytes = rowBytes;
  if (globalRowStrideBytes < rowBytes ||
      globalRowStrideBytes > std::numeric_limits<uint64_t>::max() / rows ||
      globalRowStrideBytes % copyBytesPerLane)
    return std::nullopt;
  std::optional<BufferLdsConfig> selectedConfig;
  auto cost = [&hw](const BufferLdsConfig &candidateConfig) {
    return bufferLdsConfigCost(candidateConfig, hw);
  };
  for (unsigned group = 1; group <= waves; group *= 2) {
    for (unsigned chunk = 1; chunk <= rows / waves; chunk *= 2) {
      BufferLdsConfig candidateConfig{
          rows, rowBytes, waves, copyBytesPerLane, hw.waveSize, chunk, 1, 0};
      candidateConfig.wavesPerRowGroup = group;
      // Count 32B global sectors for each wave instruction. Each lane's vector
      // stays in one logical row, hence no hidden uncoalesced gather within it.
      for (unsigned base = 0; base < candidateConfig.tileBytes();
           base += candidateConfig.bytesPerWaveTransfer()) {
        std::vector<uint64_t> sectors;
        for (unsigned byte = 0; byte < candidateConfig.bytesPerWaveTransfer();
             byte += copyBytesPerLane) {
          unsigned logical = candidateConfig.logicalByte(base + byte);
          uint64_t global =
              (logical / rowBytes) * globalRowStrideBytes + logical % rowBytes;
          sectors.push_back(global / 32);
          sectors.push_back((global + copyBytesPerLane - 1) / 32);
        }
        std::sort(sectors.begin(), sectors.end());
        candidateConfig.globalSectorCost +=
            std::unique(sectors.begin(), sectors.end()) - sectors.begin();
      }
      for (unsigned count = 1; count <= waves; count *= 2) {
        candidateConfig.wrapCount = count;
        unsigned limit =
            count == 1 ? 0
                       : std::min(wrap.maxBytes / (count - 1),
                                  candidateConfig.bytesPerWaveTransfer() - 1);
        for (unsigned step = count == 1 ? 0 : wrap.granuleBytes; step <= limit;
             step += wrap.granuleBytes) {
          candidateConfig.wrapStepBytes = step;
          // Generic async copies use wave-major placement. A schedule that
          // requires independently addressable transfer slices may explicitly
          // convert the selected configuration to transfer-major placement
          // afterwards.
          candidateConfig.transferMajor = false;
          if (evaluateBufferLdsBankConflicts(candidateConfig, hw, reads) &&
              (!selectedConfig ||
               cost(candidateConfig) < cost(*selectedConfig)))
            selectedConfig = candidateConfig;
        }
      }
    }
  }
  return selectedConfig;
}

// Derive reads from the selected legacy MMAC operand, rather than a named A/B
// workload. Rejection means there is no proven instruction profile, not that a
// caller may assume the fixed four-generation layout.
inline std::optional<BufferLdsConfig> selectMmacBufferLdsConfig(
    unsigned rows, unsigned columns, unsigned elementBytes, unsigned waves,
    unsigned copyBytes, bool kContiguous, unsigned instructionNonK,
    unsigned kWidth, const LdsGeometry &hw = {}, bool matrixRead = true,
    unsigned maxCopyBytes = 16, const BufferLdsWrapCapabilities &wrap = {},
    std::string_view arch = "gfx936") {
  if (!copyBytes) {
    std::optional<BufferLdsConfig> selectedConfig;
    for (unsigned width : {16u, 8u, 4u}) {
      if (width > maxCopyBytes)
        continue;
      if (auto candidateConfig = selectMmacBufferLdsConfig(
              rows, columns, elementBytes, waves, width, kContiguous,
              instructionNonK, kWidth, hw, matrixRead, maxCopyBytes, wrap,
              arch))
        if (!selectedConfig || bufferLdsConfigCost(*candidateConfig, hw) <
                                   bufferLdsConfigCost(*selectedConfig, hw))
          selectedConfig = candidateConfig;
    }
    return selectedConfig;
  }
  if (hw.waveSize != 64 || !rows || !columns ||
      (elementBytes != 1 && elementBytes != 2 && elementBytes != 4 &&
       elementBytes != 8) ||
      uint64_t(columns) * elementBytes > hw.capacityBytes)
    return std::nullopt;
  std::vector<LdsReadAccess> reads;
  auto matrix =
      selectLegacyMatrixRead(elementBytes * 8, instructionNonK, kWidth);
  if (kContiguous) {
    unsigned readBytes =
        legacyContiguousReadBytes(elementBytes, kWidth, copyBytes);
    auto phases = scalarReadPhases(readBytes);
    if (phases.empty())
      return std::nullopt;
    reads = contiguousMmacReads(rows, columns, elementBytes, instructionNonK,
                                kWidth, hw.waveSize, phases, readBytes);
  } else if (matrixRead && columns % 32 == 0 && matrix) {
    auto phases = matrixReadBankPhases(arch, *matrix);
    if (!phases)
      return std::nullopt;
    reads = *matrix == MatrixReadOpcode::M32x16B16
                ? matrixB16Reads(rows, columns, *phases)
                : matrixB8Reads(rows, columns, *phases);
  } else {
    auto phases = scalarReadPhases(elementBytes);
    unsigned instructionK =
        instructionNonK ? hw.waveSize / instructionNonK * kWidth : 0;
    if (phases.empty() || !instructionK || rows % instructionK ||
        columns % instructionNonK)
      return std::nullopt;
    // Non-K contiguous but no matrix-read instruction: each lane's K-fragment
    // elements occupy different rows, so model individual scalar LDS reads.
    for (unsigned kt = 0; kt < rows; kt += instructionK)
      for (unsigned nt = 0; nt < columns; nt += instructionNonK)
        for (unsigned item = 0; item < kWidth; ++item) {
          LdsReadAccess read{elementBytes, {}, phases};
          for (unsigned lane = 0; lane < hw.waveSize; ++lane)
            read.logicalBytes.push_back(
                ((kt + lane / instructionNonK * kWidth + item) * columns + nt +
                 lane % instructionNonK) *
                elementBytes);
          reads.push_back(std::move(read));
        }
  }
  return selectBufferLdsConfig(rows, columns * elementBytes, waves, copyBytes,
                               hw, reads, 0, wrap);
}

} // namespace mlir::triton::HCU
#endif
