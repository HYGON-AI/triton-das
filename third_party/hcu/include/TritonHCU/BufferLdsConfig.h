// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#ifndef TRITON_HCU_BUFFER_LDS_CONFIG_H
#define TRITON_HCU_BUFFER_LDS_CONFIG_H

// Target capabilities and byte mappings, independent of configuration search,
// MLIR and pipeline scheduling.
#include <cstdint>
#include <optional>
#include <string_view>

namespace mlir::triton::HCU {

struct LdsGeometry {
  unsigned banks = 32;
  unsigned bankBytes = 4;
  unsigned waveSize = 64;
  unsigned phaseBytes = 128;
  unsigned capacityBytes = 64 * 1024;
};

// The generic placement search only needs to know which byte rotations the
// target can encode. How those rotations are represented in M0 remains in the
// target codec below.
struct BufferLdsWrapCapabilities {
  unsigned granuleBytes = 16;
  unsigned maxBytes = 31 * 16;
};

// buffer_load_* ... lds shares M0 between the LDS byte address and a
// target-specific wrap control field.  Keep this encoding detail out of the
// placement search: the search operates on byte rotations, while lowering
// encodes the selected rotation for the target.
enum class BufferLdsWrapEncoding {
  BmzCoarse4Dword,
  SplitCoarse4DwordFineDword,
  LinearDword,
};

struct BufferLdsTargetProfile {
  LdsGeometry lds;
  BufferLdsWrapCapabilities wrap;
  unsigned ldsOffsetBits;
  unsigned wrapFieldOffset;
  unsigned wrapFieldBits;
  BufferLdsWrapEncoding wrapEncoding;

  constexpr unsigned ldsOffsetMask() const { return (1u << ldsOffsetBits) - 1; }

  constexpr std::optional<unsigned> encodeWrapBytes(unsigned wrapBytes) const {
    if (wrapBytes > wrap.maxBytes || wrapBytes % wrap.granuleBytes)
      return std::nullopt;
    unsigned encoded = 0;
    switch (wrapEncoding) {
    case BufferLdsWrapEncoding::BmzCoarse4Dword:
      encoded = wrapBytes / 16;
      break;
    case BufferLdsWrapEncoding::SplitCoarse4DwordFineDword: {
      // NMZ/YY expose five control bits in an unusual order: the low three
      // field bits select a 4-dword group and the high two select a dword
      // inside that group. The physical rotation is still
      // coarse * 16B + fine * 4B; only its M0 representation is split.
      unsigned coarse = wrapBytes / 16;
      unsigned fine = wrapBytes % 16 / 4;
      encoded = coarse | (fine << 3);
      break;
    }
    case BufferLdsWrapEncoding::LinearDword:
      encoded = wrapBytes / 4;
      break;
    }
    if (encoded >= (1u << wrapFieldBits))
      return std::nullopt;
    return encoded;
  }

  constexpr std::optional<unsigned> decodeWrapField(unsigned encoded) const {
    if (encoded >= (1u << wrapFieldBits))
      return std::nullopt;
    unsigned wrapBytes = 0;
    switch (wrapEncoding) {
    case BufferLdsWrapEncoding::BmzCoarse4Dword:
      wrapBytes = encoded * 16;
      break;
    case BufferLdsWrapEncoding::SplitCoarse4DwordFineDword:
      wrapBytes = (encoded & 7) * 16 + (encoded >> 3) * 4;
      break;
    case BufferLdsWrapEncoding::LinearDword:
      wrapBytes = encoded * 4;
      break;
    }
    if (wrapBytes > wrap.maxBytes)
      return std::nullopt;
    return wrapBytes;
  }
};

inline std::optional<BufferLdsTargetProfile>
getBufferLdsTargetProfile(std::string_view arch) {
  // BMZ / gfx936: M0[15:0] LDS bytes, M0[20:16] in 4-dword units.
  if (arch == "gfx936")
    return BufferLdsTargetProfile{
        {/*banks=*/32, /*bankBytes=*/4, /*waveSize=*/64,
         /*phaseBytes=*/128, /*capacityBytes=*/64 * 1024},
        {/*granuleBytes=*/16, /*maxBytes=*/31 * 16},
        /*ldsOffsetBits=*/16,
        /*wrapFieldOffset=*/16,
        /*wrapFieldBits=*/5,
        BufferLdsWrapEncoding::BmzCoarse4Dword};
  // NMZ / gfx938: M0[15:0] LDS bytes. M0[18:16] selects a 4-dword
  // group and M0[20:19] selects a dword within that group.
  if (arch == "gfx938")
    return BufferLdsTargetProfile{
        {/*banks=*/32, /*bankBytes=*/4, /*waveSize=*/64,
         /*phaseBytes=*/128, /*capacityBytes=*/64 * 1024},
        {/*granuleBytes=*/4, /*maxBytes=*/31 * 4},
        /*ldsOffsetBits=*/16,
        /*wrapFieldOffset=*/16,
        /*wrapFieldBits=*/5,
        BufferLdsWrapEncoding::SplitCoarse4DwordFineDword};
  // YY / gfx92a uses the NMZ split encoding in M0[28:24].
  if (arch == "gfx92a")
    return BufferLdsTargetProfile{
        {/*banks=*/32, /*bankBytes=*/4, /*waveSize=*/64,
         /*phaseBytes=*/128, /*capacityBytes=*/64 * 1024},
        {/*granuleBytes=*/4, /*maxBytes=*/31 * 4},
        /*ldsOffsetBits=*/16,
        /*wrapFieldOffset=*/24,
        /*wrapFieldBits=*/5,
        BufferLdsWrapEncoding::SplitCoarse4DwordFineDword};
  // SB / gfx946 has 128 KiB LDS and 64 physical 4-byte banks.  A hardware
  // phase remains 128 bytes (32 bank accesses), rather than widening to all
  // 64 banks. M0[16:0] is the LDS byte address and M0[29:24] is a linear
  // 0..63 dword wrap offset.
  if (arch == "gfx946")
    return BufferLdsTargetProfile{
        {/*banks=*/64, /*bankBytes=*/4, /*waveSize=*/64,
         /*phaseBytes=*/128, /*capacityBytes=*/128 * 1024},
        {/*granuleBytes=*/4, /*maxBytes=*/63 * 4},
        /*ldsOffsetBits=*/17,
        /*wrapFieldOffset=*/24,
        /*wrapFieldBits=*/6,
        BufferLdsWrapEncoding::LinearDword};
  return std::nullopt;
}

struct BufferLdsConfig {
  unsigned rows, rowBytes, waves, copyBytesPerLane, waveSize;
  unsigned rowsPerChunk, wrapCount, wrapStepBytes;
  unsigned worstReadWay = 0;
  uint64_t readPhaseCost = 0;
  uint64_t globalSectorCost = 0;
  // Zero preserves the original all-wave interleave. Smaller groups keep
  // adjacent waves on neighbouring rows instead of spanning the whole tile.
  unsigned wavesPerRowGroup = 0;
  // Place each transfer from all waves together, rather than all transfers
  // from one wave together. The logical global row interleave is unchanged.
  // This is a physical-layout contract for schedules that consume transfer
  // slices independently; it is not a bank-conflict planning variable.
  bool transferMajor = false;
  unsigned rowGroupWaves() const {
    return wavesPerRowGroup ? wavesPerRowGroup : waves;
  }

  // One issued transfer, versus all transfers assigned to the wave in a tile.
  unsigned bytesPerWaveTransfer() const { return waveSize * copyBytesPerLane; }
  // M0 wrap is a circular rotation inside one issued wave transfer. It does
  // not wrap at an LDS hardware phase (128B), nor across all transfers owned
  // by a wave. Keep this name at the address-model boundary so lowering and
  // compile-time scoring use the same hardware domain.
  unsigned wrapPeriodBytes() const { return bytesPerWaveTransfer(); }
  unsigned tileBytes() const { return rows * rowBytes; }
  unsigned totalBytesPerWave() const { return tileBytes() / waves; }

  unsigned wrapRotationBytes(unsigned producerWave) const {
    return producerWave % wrapCount * wrapStepBytes;
  }

  unsigned wrapPhysicalByte(unsigned unwrapped, unsigned producerWave) const {
    unsigned period = wrapPeriodBytes();
    return unwrapped / period * period +
           (unwrapped % period + wrapRotationBytes(producerWave)) % period;
  }

  unsigned unwrappedByte(unsigned logicalByte) const {
    unsigned row = logicalByte / rowBytes;
    unsigned col = logicalByte % rowBytes;
    unsigned groupWaves = rowGroupWaves();
    unsigned groupRows = rows / waves * groupWaves;
    unsigned group = row / groupRows;
    row %= groupRows;
    unsigned wave = group * groupWaves + (row / rowsPerChunk) % groupWaves;
    unsigned localRow =
        (row / (rowsPerChunk * groupWaves)) * rowsPerChunk + row % rowsPerChunk;
    unsigned local = localRow * rowBytes + col;
    if (transferMajor)
      return (local / bytesPerWaveTransfer() * waves + wave) *
                 bytesPerWaveTransfer() +
             local % bytesPerWaveTransfer();
    return wave * totalBytesPerWave() + local;
  }
  unsigned logicalByte(unsigned physicalByte) const {
    unsigned wave = transferMajor
                        ? physicalByte / bytesPerWaveTransfer() % waves
                        : physicalByte / totalBytesPerWave();
    unsigned local = transferMajor
                         ? physicalByte / (waves * bytesPerWaveTransfer()) *
                                   bytesPerWaveTransfer() +
                               physicalByte % bytesPerWaveTransfer()
                         : physicalByte % totalBytesPerWave();
    unsigned row = local / rowBytes;
    unsigned groupWaves = rowGroupWaves();
    unsigned logicalRow =
        wave / groupWaves * (rows / waves * groupWaves) +
        (row / rowsPerChunk * groupWaves + wave % groupWaves) * rowsPerChunk +
        row % rowsPerChunk;
    return logicalRow * rowBytes + local % rowBytes;
  }
  unsigned physicalByte(unsigned logical) const {
    unsigned offset = unwrappedByte(logical);
    unsigned wave = transferMajor ? offset / bytesPerWaveTransfer() % waves
                                  : offset / totalBytesPerWave();
    return wrapPhysicalByte(offset, wave);
  }

  bool isPhysicalRangeContiguous(unsigned logical, unsigned bytes) const {
    if (!bytes)
      return false;
    unsigned start = physicalByte(logical);
    for (unsigned byte = 1; byte < bytes; ++byte)
      if (physicalByte(logical + byte) != start + byte)
        return false;
    return true;
  }
};

inline bool isBufferLdsConfigEncodable(const BufferLdsConfig &copyConfig,
                                       const BufferLdsTargetProfile &profile) {
  if (copyConfig.waveSize != profile.lds.waveSize || !copyConfig.wrapCount ||
      uint64_t(copyConfig.rows) * copyConfig.rowBytes >
          profile.lds.capacityBytes)
    return false;
  unsigned maxRotation = (copyConfig.wrapCount - 1) * copyConfig.wrapStepBytes;
  return profile.encodeWrapBytes(copyConfig.wrapStepBytes).has_value() &&
         profile.encodeWrapBytes(maxRotation).has_value();
}

inline bool isBufferLdsPowerOfTwo(unsigned n) { return n && !(n & (n - 1)); }

} // namespace mlir::triton::HCU

#endif // TRITON_HCU_BUFFER_LDS_CONFIG_H
