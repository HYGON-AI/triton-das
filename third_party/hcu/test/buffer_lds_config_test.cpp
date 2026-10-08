// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include "TritonHCU/BufferLdsConfigSelection.h"
#include <cassert>
#include <iostream>
#include <set>

using namespace mlir::triton::HCU;

static std::vector<LdsReadAccess>
mmacReads(unsigned rows, unsigned columns, unsigned elementBytes,
          bool kContiguous, unsigned instructionNonK, unsigned kWidth,
          bool matrixRead, const LdsGeometry &hw, unsigned copyBytes = 16,
          std::string_view arch = "gfx936") {
  if (kContiguous)
    return contiguousMmacReads(
        rows, columns, elementBytes, instructionNonK, kWidth, hw.waveSize,
        scalarReadPhases(std::min(kWidth * elementBytes, copyBytes)),
        copyBytes);
  if (matrixRead && columns % 32 == 0 && instructionNonK == 16 &&
      elementBytes == 2 && kWidth == 4) {
    auto phases = matrixReadBankPhases(arch, MatrixReadOpcode::M32x16B16);
    return phases ? matrixB16Reads(rows, columns, *phases)
                  : std::vector<LdsReadAccess>{};
  }
  if (matrixRead && columns % 32 == 0 && instructionNonK == 16 &&
      elementBytes == 1 && kWidth == 8) {
    auto phases = matrixReadBankPhases(arch, MatrixReadOpcode::M32x32B8);
    return phases ? matrixB8Reads(rows, columns, *phases)
                  : std::vector<LdsReadAccess>{};
  }
  std::vector<LdsReadAccess> reads;
  unsigned instructionK =
      instructionNonK ? hw.waveSize / instructionNonK * kWidth : 0;
  auto phases = scalarReadPhases(elementBytes);
  if (!instructionK || rows % instructionK || columns % instructionNonK)
    return reads;
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
  return reads;
}

// Exhaust the production candidate space and prove that selection minimizes
// the bank-conflict pair before applying any transport/locality tie-breakers.
// lds_read_model_test.cpp independently checks the bit-level bank scoring.
static void proveMinimumBankConflict(unsigned rows, unsigned columns,
                                     unsigned elementBytes, unsigned waves,
                                     unsigned copyBytes, bool kContiguous,
                                     unsigned instructionNonK, unsigned kWidth,
                                     bool matrixRead, const LdsGeometry &hw,
                                     const BufferLdsWrapCapabilities &wrap = {},
                                     std::string_view arch = "gfx936",
                                     unsigned maxCopyBytes = 16) {
  auto selected = selectMmacBufferLdsConfig(
      rows, columns, elementBytes, waves, copyBytes, kContiguous,
      instructionNonK, kWidth, hw, matrixRead, maxCopyBytes, wrap, arch);
  assert(selected);
  auto best = std::make_pair(std::numeric_limits<unsigned>::max(),
                             std::numeric_limits<uint64_t>::max());
  for (unsigned width : {16u, 8u, 4u}) {
    if ((copyBytes && width != copyBytes) || width > maxCopyBytes ||
        columns * elementBytes % width || rows % waves ||
        uint64_t(rows) * columns * elementBytes %
            (uint64_t(waves) * hw.waveSize * width))
      continue;
    auto reads =
        mmacReads(rows, columns, elementBytes, kContiguous, instructionNonK,
                  kWidth, matrixRead, hw, width, arch);
    assert(!reads.empty());
    for (unsigned group = 1; group <= waves; group *= 2) {
      for (unsigned chunk = 1; chunk <= rows / waves; chunk *= 2)
        for (unsigned count = 1; count <= waves; count *= 2) {
          unsigned limit = count == 1 ? 0
                                      : std::min(wrap.maxBytes / (count - 1),
                                                 hw.waveSize * width - 1);
          for (unsigned step = count == 1 ? 0 : wrap.granuleBytes;
               step <= limit; step += wrap.granuleBytes) {
            BufferLdsConfig candidate{rows,        columns * elementBytes,
                                      waves,       width,
                                      hw.waveSize, chunk,
                                      count,       step};
            candidate.wavesPerRowGroup = group;
            if (!evaluateBufferLdsBankConflicts(candidate, hw, reads))
              continue;
            best = std::min(best, std::make_pair(candidate.worstReadWay,
                                                 candidate.readPhaseCost));
          }
        }
    }
  }
  assert(std::make_pair(selected->worstReadWay, selected->readPhaseCost) ==
         best);
  // transferMajor is a downstream scheduling contract, not a configuration
  // search candidate. Reordering complete transfers must not change the bank
  // score.
  assert(!selected->transferMajor);
  BufferLdsConfig transferMajor = *selected;
  transferMajor.transferMajor = true;
  auto selectedReads =
      mmacReads(rows, columns, elementBytes, kContiguous, instructionNonK,
                kWidth, matrixRead, hw, selected->copyBytesPerLane, arch);
  assert(evaluateBufferLdsBankConflicts(transferMajor, hw, selectedReads));
  assert(transferMajor.readPhaseCost == selected->readPhaseCost &&
         transferMajor.worstReadWay == selected->worstReadWay);
}

int main() {
  // Target profiles keep the byte-level placement policy independent from
  // each generation's M0 bit layout.
  {
    auto bmz = getBufferLdsTargetProfile("gfx936");
    auto nmz = getBufferLdsTargetProfile("gfx938");
    auto yy = getBufferLdsTargetProfile("gfx92a");
    auto sb = getBufferLdsTargetProfile("gfx946");
    assert(bmz && nmz && yy && sb);
    assert(!getBufferLdsTargetProfile("gfx948"));

    assert(bmz->ldsOffsetBits == 16 && bmz->wrapFieldOffset == 16);
    assert(bmz->encodeWrapBytes(0) == 0);
    assert(bmz->encodeWrapBytes(16) == 1);
    assert(bmz->encodeWrapBytes(31 * 16) == 31);
    assert(!bmz->encodeWrapBytes(4));

    for (const auto &profile : {*nmz, *yy}) {
      assert(profile.encodeWrapBytes(0) == 0);
      assert(profile.encodeWrapBytes(4) == 8);
      assert(profile.encodeWrapBytes(12) == 24);
      assert(profile.encodeWrapBytes(16) == 1);
      assert(profile.encodeWrapBytes(31 * 4) == 31);
      assert(!profile.encodeWrapBytes(128));
      for (unsigned bytes = 0; bytes <= 31 * 4; bytes += 4) {
        auto encoded = profile.encodeWrapBytes(bytes);
        assert(encoded && profile.decodeWrapField(*encoded) == bytes);
      }
    }
    assert(nmz->wrapFieldOffset == 16 && yy->wrapFieldOffset == 24);

    assert(sb->ldsOffsetBits == 17 && sb->ldsOffsetMask() == 0x1ffff);
    assert(sb->wrapFieldOffset == 24 && sb->lds.capacityBytes == 128 * 1024);
    assert(sb->lds.phaseBytes == 128 && sb->lds.banks == 64);
    assert(sb->encodeWrapBytes(4) == 1);
    assert(sb->encodeWrapBytes(63 * 4) == 63);
    assert(!sb->encodeWrapBytes(64 * 4));
    for (unsigned bytes = 0; bytes <= 63 * 4; bytes += 4) {
      auto encoded = sb->encodeWrapBytes(bytes);
      assert(encoded && sb->decodeWrapField(*encoded) == bytes);
    }
    for (unsigned bytes = 0; bytes <= 31 * 16; bytes += 16) {
      auto encoded = bmz->encodeWrapBytes(bytes);
      assert(encoded && bmz->decodeWrapField(*encoded) == bytes);
    }

    BufferLdsConfig bmzConfig{256, 32, 8, 16, 64, 8, 2, 16};
    assert(isBufferLdsConfigEncodable(bmzConfig, *bmz));
    assert(isBufferLdsConfigEncodable(bmzConfig, *nmz));
    assert(isBufferLdsConfigEncodable(bmzConfig, *yy));
    assert(isBufferLdsConfigEncodable(bmzConfig, *sb));
    bmzConfig.wrapStepBytes = 4;
    assert(!isBufferLdsConfigEncodable(bmzConfig, *bmz));
    assert(isBufferLdsConfigEncodable(bmzConfig, *nmz));
    assert(isBufferLdsConfigEncodable(bmzConfig, *yy));
    assert(isBufferLdsConfigEncodable(bmzConfig, *sb));

    BufferLdsConfig largeConfig{2, 65536, 1, 16, 64, 2, 1, 0};
    assert(!isBufferLdsConfigEncodable(largeConfig, *bmz));
    assert(isBufferLdsConfigEncodable(largeConfig, *sb));

    LdsReadAccess broadcast{4, std::vector<unsigned>(64, 0),
                            consecutiveLdsPhases(64, 32)};
    assert(!selectBufferLdsConfig(2, 65536, 1, 16, bmz->lds, {broadcast}, 0,
                                  bmz->wrap));
    assert(selectBufferLdsConfig(2, 65536, 1, 16, sb->lds, {broadcast}, 0,
                                 sb->wrap));
  }
  // NMZ fine wrap is a 4-byte circular rotation inside each issued wave
  // transfer. The period scales with the copy width and is deliberately not
  // the 128-byte LDS phase or the wave's complete tile allocation.
  {
    BufferLdsConfig fine4{2, 1024, 2, 4, 64, 1, 2, 4};
    assert(fine4.wrapPeriodBytes() == 256);
    assert(fine4.totalBytesPerWave() == 1024);
    assert(fine4.wrapPhysicalByte(1024, 1) == 1028);
    assert(fine4.wrapPhysicalByte(1276, 1) == 1024);
    assert(fine4.wrapPhysicalByte(1280, 1) == 1284);
    assert(fine4.wrapPhysicalByte(1532, 1) == 1280);
    // A normal contiguous LDS vector cannot cross the circular boundary.
    assert(fine4.isPhysicalRangeContiguous(1024, 4));
    assert(!fine4.isPhysicalRangeContiguous(1278, 4));

    BufferLdsConfig fine16 = fine4;
    fine16.copyBytesPerLane = 16;
    assert(fine16.wrapPeriodBytes() == 1024);
    assert(fine16.wrapPhysicalByte(1024, 1) == 1028);
    assert(fine16.wrapPhysicalByte(2044, 1) == 1024);
    assert(!fine16.isPhysicalRangeContiguous(2040, 8));
  }
  LdsGeometry hw;
  for (unsigned width : {2u, 4u, 8u, 16u}) {
    auto phases = width == 16 ? readB128PhasesWave64()
                              : consecutiveLdsPhases(64, width == 8   ? 16
                                                         : width == 4 ? 32
                                                                      : 64);
    LdsReadAccess read{width, std::vector<unsigned>(64, 0), phases};
    BufferLdsConfig copyConfig{256, 32, 8, 16, 64, 32, 1, 0};
    assert(evaluateBufferLdsBankConflicts(copyConfig, hw, {read}));
    assert(copyConfig.worstReadWay == 1); // same-word broadcast, not 64-way
    read.phases[0].push_back(0);
    assert(!evaluateBufferLdsBankConflicts(copyConfig, hw, {read}));
  }
  for (bool kContiguous : {true, false}) {
    auto selectedConfig = selectMmacBufferLdsConfig(kContiguous ? 256 : 16,
                                                    kContiguous ? 16 : 256, 2,
                                                    8, 16, kContiguous, 16, 4);
    assert(selectedConfig);
    const auto &copyConfig = *selectedConfig;
    assert(copyConfig.rowsPerChunk == (kContiguous ? 4u : 1u));
    assert(copyConfig.rowGroupWaves() == (kContiguous ? 8u : 2u));
    assert(copyConfig.wrapCount == (kContiguous ? 4u : 2u));
    assert(copyConfig.wrapStepBytes == (kContiguous ? 16u : 64u));
    // Independently specify the reference per-lane global assignment; a
    // producer/consumer round-trip alone would miss a shared mapping mistake.
    for (unsigned w = 0; w < 8; ++w)
      for (unsigned lane = 0; lane < 64; ++lane) {
        unsigned row = kContiguous ? 32 * (lane / 8) + 4 * w + (lane / 2) % 4
                                   : 4 * (w / 2) + w % 2 + 2 * (lane / 32);
        unsigned col = kContiguous ? 8 * (lane % 2) : 8 * (lane % 32);
        assert(copyConfig.logicalByte(w * 1024 + lane * 16) ==
               row * copyConfig.rowBytes + col * 2);
      }
    assert(copyConfig.bytesPerWaveTransfer() ==
           copyConfig.waveSize * copyConfig.copyBytesPerLane);
    assert(copyConfig.totalBytesPerWave() * copyConfig.waves ==
           copyConfig.tileBytes());
    assert(copyConfig.totalBytesPerWave() % copyConfig.bytesPerWaveTransfer() ==
           0);
    std::set<unsigned> addresses;
    for (unsigned byte = 0; byte < copyConfig.tileBytes(); ++byte) {
      assert(copyConfig.logicalByte(copyConfig.unwrappedByte(byte)) == byte);
      assert(copyConfig.physicalByte(byte) < copyConfig.tileBytes());
      addresses.insert(copyConfig.physicalByte(byte));
    }
    assert(addresses.size() == copyConfig.tileBytes());
    std::cout << (kContiguous ? "K-contiguous" : "nonK-contiguous")
              << " chunk=" << copyConfig.rowsPerChunk
              << " wrap=" << copyConfig.wrapCount
              << " step=" << copyConfig.wrapStepBytes
              << " way=" << copyConfig.worstReadWay
              << " phases=" << copyConfig.readPhaseCost << '\n';
  }
  // The four legacy HCU targets share matrix-read lane routing/phases while
  // retaining target-specific LDS geometry and M0 wrap encodings.
  for (std::string_view arch : {"gfx92a", "gfx936", "gfx938", "gfx946"}) {
    auto profile = getBufferLdsTargetProfile(arch);
    assert(profile);
    auto copyConfig = selectMmacBufferLdsConfig(
        16, 128, 2, 8, 0, /*kContiguous=*/false, 16, 4, profile->lds,
        /*matrixRead=*/true, 16, profile->wrap, arch);
    assert(copyConfig && copyConfig->worstReadWay > 0);
    proveMinimumBankConflict(16, 128, 2, 8, 0, /*kContiguous=*/false,
                             16, 4, /*matrixRead=*/true, profile->lds,
                             profile->wrap, arch);
    proveMinimumBankConflict(32, 128, 1, 8, 0, /*kContiguous=*/false,
                             16, 8, /*matrixRead=*/true, profile->lds,
                             profile->wrap, arch);
    std::set<unsigned> physical;
    for (unsigned byte = 0; byte < copyConfig->tileBytes(); ++byte)
      physical.insert(copyConfig->physicalByte(byte));
    assert(physical.size() == copyConfig->tileBytes());
  }
  // WASP producer slicing changes the backing A/B tiles and producer-wave
  // count after the initial full-CTA selection. Prove that rescoring those
  // final Shaobo tiles is bank-optimal across every legal copy width.
  {
    auto sb = getBufferLdsTargetProfile("gfx946");
    assert(sb);
    for (unsigned waves : {1u, 2u, 4u}) {
      proveMinimumBankConflict(32, 16, 2, waves, 0,
                               /*kContiguous=*/true, 16, 4,
                               /*matrixRead=*/true, sb->lds, sb->wrap,
                               "gfx946");
      proveMinimumBankConflict(16, 64, 2, waves, 0,
                               /*kContiguous=*/false, 16, 4,
                               /*matrixRead=*/true, sb->lds, sb->wrap,
                               "gfx946");
    }
  }
  {
    auto selected = selectMmacBufferLdsConfig(256, 32, 2, 8, 16, true, 16, 4);
    assert(selected && selected->rowsPerChunk == 2);
    BufferLdsConfig transferMajor = *selected;
    transferMajor.transferMajor = true;
    transferMajor.wrapCount = 2;
    transferMajor.wrapStepBytes = 32;
    std::set<unsigned> addresses;
    for (unsigned byte = 0; byte < transferMajor.tileBytes(); ++byte) {
      assert(transferMajor.logicalByte(transferMajor.unwrappedByte(byte)) ==
             byte);
      addresses.insert(transferMajor.physicalByte(byte));
    }
    assert(addresses.size() == transferMajor.tileBytes());
    for (unsigned transfer = 0; transfer < 2; ++transfer)
      for (unsigned wave = 0; wave < 8; ++wave)
        for (unsigned lane = 0; lane < 64; ++lane) {
          unsigned oldOffset = wave * 2048 + transfer * 1024 + lane * 16;
          unsigned newOffset = transfer * 8192 + wave * 1024 + lane * 16;
          unsigned logical = selected->logicalByte(oldOffset);
          assert(transferMajor.logicalByte(newOffset) == logical);
          assert(transferMajor.unwrappedByte(logical) == newOffset);
          assert(transferMajor.physicalByte(logical) ==
                 newOffset / 1024 * 1024 +
                     (newOffset % 1024 + (wave & 1) * 32) % 1024);
        }
  }
  // Different element widths, tile shapes, consumer instructions and wave
  // counts exercise configuration selection, not only the production adapter's
  // two tiles.
  for (unsigned elem : {1u, 2u, 4u})
    for (unsigned waves : {2u, 4u, 8u})
      for (unsigned k : {16u, 32u, 64u}) {
        unsigned rows = 256;
        auto phases = elem == 4 ? readB128PhasesWave64()
                                : consecutiveLdsPhases(64, elem == 2 ? 16 : 32);
        auto reads = contiguousMmacReads(rows, k, elem, 16, 4, 64, phases);
        auto copyConfig =
            selectBufferLdsConfig(rows, k * elem, waves, 16, hw, reads);
        if (rows * k * elem < waves * 64 * 16) {
          assert(!copyConfig);
          continue;
        }
        assert(copyConfig);
        auto base = *copyConfig;
        base.rowsPerChunk = rows / waves;
        base.wrapCount = 1;
        base.wrapStepBytes = 0;
        assert(evaluateBufferLdsBankConflicts(base, hw, reads));
        assert(copyConfig->readPhaseCost <= base.readPhaseCost);
      }
  // Production FP16 and gfx936 FP8-emulation profiles. Prove optimality over
  // the complete bounded row-interleave/wrap search, for both MMAC operands.
  for (unsigned elem : {1u, 2u})
    for (unsigned waves : {2u, 4u, 8u})
      for (unsigned nonK : {64u, 128u, 256u}) {
        unsigned kWidth = elem == 1 ? 4 : 4;
        unsigned k = elem == 1 ? 32 : 16;
        proveMinimumBankConflict(nonK, k, elem, waves, 0, true, 16, kWidth,
                                 true, hw);
        proveMinimumBankConflict(k, nonK, elem, waves, 0, false, 16, kWidth,
                                 true, hw);
      }
  // Historical GPU transport fixtures remain legal placements, but are not
  // assumed to be winners after changing the candidate set or objective.
  struct ExpectedConfig {
    unsigned elem, rows, columns, waves, chunk, group, wrap, step;
    bool kContiguous;
  };
  const ExpectedConfig expectedConfigs[] = {
      {1, 64, 32, 2, 4, 2, 2, 16, true},   {1, 128, 32, 4, 4, 4, 4, 16, true},
      {1, 256, 32, 8, 4, 8, 4, 16, true},  {1, 32, 64, 2, 4, 2, 2, 16, false},
      {1, 32, 128, 4, 4, 4, 4, 16, false}, {1, 32, 256, 8, 2, 8, 8, 16, false},
      {2, 64, 16, 2, 4, 2, 2, 16, true},   {2, 128, 16, 4, 4, 4, 4, 16, true},
      {2, 256, 16, 8, 4, 8, 4, 16, true},  {2, 16, 64, 2, 1, 2, 2, 64, false},
      {2, 16, 128, 4, 1, 4, 2, 64, false}, {2, 16, 256, 8, 1, 2, 2, 64, false},
  };
  for (const auto &e : expectedConfigs) {
    auto copyConfig = selectMmacBufferLdsConfig(
        e.rows, e.columns, e.elem, e.waves, 0, e.kContiguous, 16, 4);
    assert(copyConfig);
    BufferLdsConfig fixture{
        e.rows, e.columns * e.elem, e.waves, 16, 64, e.chunk, e.wrap, e.step};
    fixture.wavesPerRowGroup = e.group;
    auto reads =
        mmacReads(e.rows, e.columns, e.elem, e.kContiguous, 16, 4, true, hw);
    assert(evaluateBufferLdsBankConflicts(fixture, hw, reads));
    assert(
        std::make_pair(copyConfig->worstReadWay, copyConfig->readPhaseCost) <=
        std::make_pair(fixture.worstReadWay, fixture.readPhaseCost));
  }
  assert(!selectBufferLdsConfig(0, 32, 8, 16, hw, {}));
  assert(!selectBufferLdsConfig(255, 32, 8, 16, hw, {}));
  assert(!selectBufferLdsConfig(256, 32, 8, 3, hw, {}));
  for (unsigned banks : {32u, 64u})
    for (unsigned copy : {4u, 8u, 16u}) {
      LdsGeometry geometry = hw;
      geometry.banks = banks;
      auto reads = matrixB16Reads(16, 128);
      auto copyConfig =
          selectBufferLdsConfig(16, 256, 4, copy, geometry, reads, 512);
      assert(copyConfig);
      std::set<unsigned> physical;
      for (unsigned byte = 0; byte < copyConfig->tileBytes(); ++byte) {
        assert(copyConfig->logicalByte(copyConfig->unwrappedByte(byte)) ==
               byte);
        physical.insert(copyConfig->physicalByte(byte));
      }
      assert(physical.size() == copyConfig->tileBytes());
    }
  {
    auto reads = contiguousMmacReads(256, 16, 2, 16, 4, 64,
                                     consecutiveLdsPhases(64, 16));
    BufferLdsConfig baseline{256, 32, 8, 16, 64, 32, 1, 0};
    assert(evaluateBufferLdsBankConflicts(baseline, hw, reads));
    assert(baseline.worstReadWay == 4);
    auto copyConfig = selectMmacBufferLdsConfig(256, 16, 2, 8, 16, true, 16, 4);
    assert(copyConfig && copyConfig->worstReadWay == 2);
    assert(!selectBufferLdsConfig(256, 32, 8, 16, hw, reads, 31));
    auto invalid = hw;
    invalid.bankBytes = 0;
    assert(!selectBufferLdsConfig(256, 32, 8, 16, invalid, reads));
    assert(!evaluateBufferLdsBankConflicts(baseline, invalid, reads));
  }
  // A vector crossing a rotation boundary must be rejected, not reinterpreted
  // as a gather by changing its start address.
  BufferLdsConfig crossing{256, 32, 8, 16, 64, 32, 2, 16};
  LdsReadAccess read{8, std::vector<unsigned>(64, 2028),
                     consecutiveLdsPhases(64, 16)};
  assert(!evaluateBufferLdsBankConflicts(crossing, hw, {read}));
  for (unsigned waves : {1u, 2u, 4u, 8u, 16u}) {
    for (unsigned elem : {1u, 2u}) {
      unsigned kWidth = 8 / elem, k = 4 * kWidth;
      auto a =
          selectMmacBufferLdsConfig(128, k, elem, waves, 0, true, 16, kWidth);
      auto b =
          selectMmacBufferLdsConfig(k, 128, elem, waves, 0, false, 16, kWidth);
      assert(a && b);
      for (const auto &copyConfig : {*a, *b}) {
        std::set<unsigned> addresses;
        for (unsigned byte = 0; byte < copyConfig.tileBytes(); ++byte) {
          assert(copyConfig.logicalByte(copyConfig.unwrappedByte(byte)) ==
                 byte);
          assert(copyConfig.physicalByte(byte) < copyConfig.tileBytes());
          addresses.insert(copyConfig.physicalByte(byte));
        }
        assert(addresses.size() == copyConfig.tileBytes());
      }
    }
  }
  // gfx936 keeps FP8 storage but uses four-element FP16 MMAC fragments
  // after conversion. Model the byte reads before that conversion, including
  // the non-K-contiguous operand's scattered single-byte loads.
  for (bool kContiguous : {false, true}) {
    auto copyConfig = selectMmacBufferLdsConfig(kContiguous ? 128 : 32,
                                                kContiguous ? 32 : 128, 1, 4, 0,
                                                kContiguous, 16, 4);
    assert(copyConfig);
    std::set<unsigned> addresses;
    for (unsigned byte = 0; byte < copyConfig->tileBytes(); ++byte) {
      assert(copyConfig->logicalByte(copyConfig->unwrappedByte(byte)) == byte);
      assert(copyConfig->physicalByte(byte) < copyConfig->tileBytes());
      addresses.insert(copyConfig->physicalByte(byte));
    }
    assert(addresses.size() == copyConfig->tileBytes());
  }
  auto small = selectMmacBufferLdsConfig(16, 16, 2, 1, 0, true, 16, 4);
  assert(small && small->copyBytesPerLane == 8);
  // Source pointer/mask alignment limits the search, independently of the
  // shared encoding. Pick the widest legal candidate within that bound.
  for (unsigned bound : {4u, 8u, 16u}) {
    auto limited = selectMmacBufferLdsConfig(256, 16, 2, 8, 0, true, 16, 4, hw,
                                             true, bound);
    assert(limited && limited->copyBytesPerLane == bound);
  }
  assert(
      !selectMmacBufferLdsConfig(256, 16, 2, 8, 0, true, 16, 4, hw, true, 2));
  assert(!selectMmacBufferLdsConfig(16, 32, 2, 1, 0, false, 16, 8));
  assert(!selectMmacBufferLdsConfig(16, 16, 3, 1, 0, true, 16, 4));
  std::cout << "PASS\n";
}
