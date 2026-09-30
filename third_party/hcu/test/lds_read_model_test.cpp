// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include "TritonHCU/BufferLdsConfigSelection.h"
#include <cassert>
#include <climits>
#include <iostream>
#include <map>

using namespace mlir::triton::HCU;

// Independent bank oracle: expand individual bits, then count the unique
// physical words requested by each lane. Does not call the production scorer.
static LdsBankCost oracle(const LdsReadTrace &trace, LdsBankGeometry hw) {
  LdsBankCost result;
  for (const auto &phase : trace.phases) {
    std::map<unsigned, std::set<std::pair<unsigned, uint64_t>>> banks;
    for (const auto &request : phase)
      for (uint64_t bit = request.address.begin;
           bit < request.address.begin + request.address.size; ++bit)
        banks[(bit / hw.bankBits) % hw.banks].insert(
            {trace.broadcast == LdsBroadcast::None ? request.sourceLane : 0,
             bit / hw.bankBits});
    unsigned way = 0;
    for (const auto &bank : banks)
      way = std::max(way, unsigned(bank.second.size()));
    result.worstWay = std::max(result.worstWay, way);
    result.phaseService += way;
  }
  return result;
}

static unsigned physicalByteOracle(const BufferLdsConfig &copyConfig,
                                   unsigned logical) {
  unsigned group = copyConfig.wavesPerRowGroup;
  unsigned row = logical / copyConfig.rowBytes,
           col = logical % copyConfig.rowBytes;
  unsigned groupRows = copyConfig.rows / copyConfig.waves * group;
  unsigned wave = row / groupRows * group +
                  (row % groupRows / copyConfig.rowsPerChunk) % group;
  unsigned localRow = row % groupRows / (copyConfig.rowsPerChunk * group) *
                          copyConfig.rowsPerChunk +
                      row % copyConfig.rowsPerChunk;
  unsigned local = localRow * copyConfig.rowBytes + col;
  unsigned transferBytes = copyConfig.waveSize * copyConfig.copyBytesPerLane;
  unsigned base =
      copyConfig.transferMajor
          ? (local / transferBytes * copyConfig.waves + wave) * transferBytes
          : wave * (copyConfig.rows * copyConfig.rowBytes / copyConfig.waves) +
                local / transferBytes * transferBytes;
  return base + (local % transferBytes +
                 wave % copyConfig.wrapCount * copyConfig.wrapStepBytes) %
                    transferBytes;
}

static std::optional<LdsBankCost>
bufferLdsBankConflictOracle(const BufferLdsConfig &copyConfig,
                            const std::vector<LdsReadAccess> &reads,
                            LdsBankGeometry hw) {
  LdsReadTrace trace;
  trace.broadcast = LdsBroadcast::SameBankWord;
  for (const auto &read : reads) {
    for (unsigned logical : read.logicalBytes)
      for (unsigned byte = 0; byte < read.bytesPerLane; ++byte)
        if (physicalByteOracle(copyConfig, logical + byte) !=
            physicalByteOracle(copyConfig, logical) + byte)
          return std::nullopt;
    for (const auto &phase : read.phases) {
      trace.phases.emplace_back();
      for (unsigned lane : phase)
        trace.phases.back().push_back(
            {lane,
             {physicalByteOracle(copyConfig, read.logicalBytes[lane]) * 8,
              read.bytesPerLane * 8}});
    }
  }
  return oracle(trace, hw);
}

static std::vector<LdsReadAccess>
shaoboMmacReads(unsigned rows, unsigned columns, unsigned elementBytes,
                unsigned copyBytes, bool kContiguous) {
  constexpr unsigned instructionNonK = 16, kWidth = 4, waveSize = 64;
  if (kContiguous) {
    unsigned readBytes =
        legacyContiguousReadBytes(elementBytes, kWidth, copyBytes);
    return contiguousMmacReads(rows, columns, elementBytes, instructionNonK,
                               kWidth, waveSize,
                               scalarReadPhases(readBytes), readBytes);
  }
  std::vector<LdsReadAccess> reads;
  unsigned instructionK = waveSize / instructionNonK * kWidth;
  for (unsigned kt = 0; kt < rows; kt += instructionK)
    for (unsigned nt = 0; nt < columns; nt += instructionNonK)
      for (unsigned item = 0; item < kWidth; ++item) {
        LdsReadAccess read{elementBytes, {}, scalarReadPhases(elementBytes)};
        for (unsigned lane = 0; lane < waveSize; ++lane)
          read.logicalBytes.push_back(
              ((kt + lane / instructionNonK * kWidth + item) * columns + nt +
               lane % instructionNonK) *
              elementBytes);
        reads.push_back(std::move(read));
      }
  return reads;
}

int main() {
  // This inventory assertion checks names/shapes only, NOT hardware support.
  assert(std::size(matrixReadInstructions) == 10);
  for (const auto &inst : matrixReadInstructions) {
    assert(inst.rows * inst.columns * inst.elementBits == 8192);
    assert(&matrixReadInstruction(inst.opcode) == &inst);
  }
  assert(selectLegacyMatrixRead(16, 16, 4) == MatrixReadOpcode::M32x16B16);
  assert(selectLegacyMatrixRead(8, 16, 8) == MatrixReadOpcode::M32x32B8);
  assert(!selectLegacyMatrixRead(8, 16, 4)); // FP8 emulation, not native b8
  assert(!selectLegacyMatrixRead(16, 32, 4));
  assert(!selectLegacyMatrixRead(4, 16, 16)); // inventory is not eligibility
  auto wide = contiguousMmacReads(16, 16, 2, 16, 4, 64, scalarReadPhases(8), 8);
  auto narrow =
      contiguousMmacReads(16, 16, 2, 16, 4, 64, scalarReadPhases(4), 4);
  assert(wide.size() == 1 && narrow.size() == 2);
  assert(wide[0].phases.size() == 4 && narrow[0].phases.size() == 2);
  for (unsigned lane = 0; lane < 64; ++lane) {
    assert(narrow[0].logicalBytes[lane] == wide[0].logicalBytes[lane]);
    assert(narrow[1].logicalBytes[lane] == wide[0].logicalBytes[lane] + 4);
  }
  LdsBankGeometry hw{32, 32, 64};
  // Hardware-captured matrix permutations are independent of phase tables.
  // A synthetic coverage phase tests routing only, not bank optimality.
  std::vector<uint64_t> sourceAddresses(64);
  for (unsigned lane = 0; lane < 64; ++lane)
    sourceAddresses[lane] = lane * 256; // non-linear-to-element-index spacing
  for (const auto &inst : matrixReadInstructions) {
    auto results = matrixReadResults("gfx936", inst.opcode, sourceAddresses);
    bool measured = inst.opcode <= MatrixReadOpcode::M32x8B32;
    assert(bool(results) == measured);
    assert(!matrixReadResults("gfx950", inst.opcode, sourceAddresses));
    if (!results)
      continue;
    LdsReadTrace trace;
    trace.results = *results;
    auto phases = matrixReadBankPhases("gfx936", inst.opcode);
    assert(bool(phases) == (inst.opcode != MatrixReadOpcode::M32x64B4));
    auto complete = matrixReadTrace("gfx936", inst.opcode, sourceAddresses);
    assert(bool(complete) == bool(phases));
    if (complete)
      assert(validateLdsResultMapping(*complete, hw, 4));
    assert(!matrixReadBankPhases("gfx950", inst.opcode));
    if (!phases)
      phases = consecutiveLdsPhases(64, 64); // routing-only coverage
    for (const auto &phase : *phases) {
      trace.phases.emplace_back();
      for (unsigned lane : phase)
        trace.phases.back().push_back({lane, {sourceAddresses[lane], 128}});
    }
    assert(validateLdsResultMapping(trace, hw, 4));
    std::set<uint64_t> used;
    for (const auto &piece : trace.results)
      assert(used.insert(piece.source.begin).second);
  }
  for (std::string_view arch : {"gfx92a", "gfx938", "gfx946"}) {
    LdsBankGeometry targetHw{arch == "gfx946" ? 64u : 32u, 32, 64};
    for (const auto &inst : matrixReadInstructions) {
      bool supported = inst.opcode == MatrixReadOpcode::M32x16B16 ||
                       inst.opcode == MatrixReadOpcode::M32x32B8;
      assert(bool(matrixReadResultBasis(arch, inst.opcode)) == supported);
      assert(bool(matrixReadBankPhases(arch, inst.opcode)) == supported);
      if (supported) {
        assert(matrixReadResultBasis(arch, inst.opcode) ==
               matrixReadResultBasis("gfx936", inst.opcode));
        assert(matrixReadBankPhases(arch, inst.opcode) ==
               matrixReadBankPhases("gfx936", inst.opcode));
      }
      auto trace = matrixReadTrace(arch, inst.opcode, sourceAddresses);
      if (trace)
        assert(validateLdsResultMapping(*trace, targetHw, 4));
    }
  }
  // read2 component phases retain their scalar width. Measured on gfx936:
  // b32 strides 4/8/128 -> 1/2/32-way; b64 8/16/128 -> 1/2/16-way.
  for (unsigned bytes : {4u, 8u})
    for (bool st64 : {false, true})
      for (unsigned stride : {bytes, bytes * 2, 128u}) {
        std::vector<uint64_t> addresses(64);
        for (unsigned lane = 0; lane < 64; ++lane)
          addresses[lane] = lane * stride;
        auto trace = pairedReadTrace(
            addresses, bytes, 0, st64 ? 1 : 128 / bytes, st64,
            scalarReadPhases(bytes), LdsBroadcast::SameBankWord);
        assert(trace && validateLdsResultMapping(*trace, hw, bytes / 2));
        auto cost = scoreLdsReadTrace(*trace, hw);
        unsigned way = stride == bytes       ? 1
                       : stride == bytes * 2 ? 2
                                             : 128 / bytes;
        assert(cost && cost->worstWay == way);
        assert(cost->phaseService == (bytes == 4 ? 4 : 8) * way);
      }
  assert(!pairedReadTrace({0}, 4, 0, 256, false, {{0}}, LdsBroadcast::None));
  assert(!pairedReadTrace({0}, 4, 0, 1, false, {{0, 0}}, LdsBroadcast::None));
  assert(!pairedReadTrace({0, 4}, 4, 0, 1, false, {{0}}, LdsBroadcast::None));
  for (auto broadcast : {LdsBroadcast::None, LdsBroadcast::SameBankWord})
    for (unsigned bits : {4u, 8u, 16u, 32u, 64u, 128u})
      for (unsigned stride : {0u, 4u, 16u, 32u, 64u, 128u, 1024u})
        for (unsigned shift : {0u, 4u, 28u}) {
          LdsReadTrace trace;
          trace.broadcast = broadcast;
          // Exercise discontiguous requests and repeated source lanes in
          // different phases; neither is representable by the old lane table.
          for (unsigned phase = 0; phase < 2; ++phase) {
            trace.phases.emplace_back();
            for (unsigned lane = 0; lane < 64; ++lane) {
              trace.phases.back().push_back(
                  {lane, {lane * stride + shift + phase * 4096, bits}});
              trace.phases.back().push_back(
                  {lane, {lane * stride + shift + phase * 4096 + 16384, bits}});
            }
          }
          auto got = scoreLdsReadTrace(trace, hw);
          auto expected = oracle(trace, hw);
          assert(got && got->worstWay == expected.worstWay &&
                 got->phaseService == expected.phaseService);
        }
  LdsReadTrace sameWord{{{{0, {0, 4}}, {1, {4, 4}}}}, {}, LdsBroadcast::None};
  assert(scoreLdsReadTrace(sameWord, hw)->worstWay == 2);
  sameWord.broadcast = LdsBroadcast::SameBankWord;
  assert(scoreLdsReadTrace(sameWord, hw)->worstWay == 1);
  sameWord.phases[0][0].address = {UINT64_MAX, 2};
  assert(!scoreLdsReadTrace(sameWord, hw));
  assert(!scoreLdsReadTrace({}, hw));
  for (unsigned banks : {16u, 32u, 64u}) {
    LdsReadTrace trace;
    trace.phases.emplace_back();
    for (unsigned lane = 0; lane < 64; ++lane)
      trace.phases.back().push_back({lane, {lane * 32, 32}});
    auto cost = scoreLdsReadTrace(trace, {banks, 32, 64});
    assert(cost && cost->worstWay == 64 / banks);
  }

  // Independent golden mapping for m32x16_b16: destination lane 0 receives
  // 0,32,64,96,16,48,80,112, not the source span at lane 0's address.
  LdsReadTrace matrix;
  matrix.broadcast = LdsBroadcast::SameBankWord;
  for (const auto &phase : matrixB16PhasesWave64()) {
    matrix.phases.emplace_back();
    for (unsigned lane : phase)
      matrix.phases.back().push_back({lane, {lane * 128, 128}});
  }
  for (unsigned lane = 0; lane < 64; ++lane)
    for (unsigned item = 0; item < 8; ++item) {
      unsigned k = lane / 16 * 4 + item % 4;
      unsigned n = lane % 16 + (item / 4) * 16;
      unsigned value = k * 32 + n;
      matrix.results.push_back(
          {lane, item / 2, (item % 2) * 16, value / 8, {value * 16, 16}});
    }
  assert(validateLdsResultMapping(matrix, hw, 4));
  const unsigned laneZero[] = {0, 32, 64, 96, 16, 48, 80, 112};
  for (unsigned i = 0; i < 8; ++i)
    assert(matrix.results[i].source.begin == laneZero[i] * 16);
  matrix.results[0].sourceLane = 63;
  assert(!validateLdsResultMapping(matrix, hw, 4));
  matrix.results[0].sourceLane = 0;
  matrix.results[1].registerBit = 0; // overlaps v0.low
  assert(!validateLdsResultMapping(matrix, hw, 4));

  // Exhaustively compare the existing byte scorer against the independent
  // bit oracle on every placement in a small, explicitly bounded family.
  auto reads = matrixB16Reads(16, 32);
  for (unsigned chunk : {1u, 2u, 4u, 8u})
    for (unsigned group : {1u, 2u})
      for (unsigned wrap : {1u, 2u})
        for (unsigned step : {0u, 16u, 32u, 64u}) {
          BufferLdsConfig copyConfig{16, 64, 2, 8, 64, chunk, wrap, step};
          copyConfig.wavesPerRowGroup = group;
          assert(evaluateBufferLdsBankConflicts(copyConfig, {}, reads));
          LdsReadTrace trace;
          trace.broadcast = LdsBroadcast::SameBankWord;
          for (const auto &phase : reads[0].phases) {
            trace.phases.emplace_back();
            for (unsigned lane : phase) {
              unsigned logical = reads[0].logicalBytes[lane];
              // Independent row decomposition; do not call physicalByte().
              unsigned row = logical / 64, col = logical % 64;
              unsigned groupRows = 8 * group;
              unsigned wave =
                  row / groupRows * group + (row % groupRows / chunk) % group;
              unsigned localRow =
                  (row % groupRows / (chunk * group)) * chunk + row % chunk;
              unsigned physical =
                  wave * 512 +
                  (localRow * 64 + col + (wave % wrap) * step) % 512;
              trace.phases.back().push_back({lane, {physical * 8, 128}});
            }
          }
          auto expected = oracle(trace, hw);
          assert(expected.worstWay == copyConfig.worstReadWay &&
                 expected.phaseService == copyConfig.readPhaseCost);
        }
  // Exhaustive independent optimum over the declared small candidate family,
  // including copy width and both transfer orders. No production score or
  // address conversion is used to choose the reference winner.
  auto best = std::make_pair(UINT_MAX, UINT64_MAX);
  for (unsigned width : {4u, 8u})
    for (unsigned group : {1u, 2u})
      for (unsigned chunk : {1u, 2u, 4u, 8u})
        for (unsigned count : {1u, 2u})
          for (unsigned step = 0; step <= 496 && step < 64 * width;
               step += 16) {
            if ((count == 1) != (step == 0))
              continue;
            for (bool transferMajor : {false, true}) {
              BufferLdsConfig copyConfig{16, 64,    2,     width,
                                         64, chunk, count, step};
              copyConfig.wavesPerRowGroup = group;
              copyConfig.transferMajor = transferMajor;
              if (auto cost = bufferLdsBankConflictOracle(
                      copyConfig, reads, {32, 32, 64}))
                best = std::min(
                    best, std::make_pair(cost->worstWay, cost->phaseService));
            }
          }
  auto selected = selectMmacBufferLdsConfig(16, 32, 2, 2, 0, false, 16, 4);
  assert(selected && std::make_pair(selected->worstReadWay,
                                    selected->readPhaseCost) == best);

  // SB/gfx946 has 64 banks and a 4-byte linear wrap step. Cross-check the
  // production scorer against the independent bit oracle over the complete
  // bounded config family used by selection, for both MMAC operand layouts.
  auto sb = getBufferLdsTargetProfile("gfx946");
  assert(sb);
  LdsBankGeometry sbBanks{sb->lds.banks, sb->lds.bankBytes * 8,
                          sb->lds.waveSize};
  for (bool kContiguous : {false, true}) {
    unsigned rows = kContiguous ? 128 : 32;
    unsigned columns = kContiguous ? 32 : 128;
    for (unsigned width : {4u, 8u, 16u}) {
      auto sbReads =
          shaoboMmacReads(rows, columns, 2, width, kContiguous);
      auto sbBest = std::make_pair(UINT_MAX, UINT64_MAX);
      std::set<std::pair<unsigned, uint64_t>> observedCosts;
      unsigned validConfigs = 0;
      for (unsigned group = 1; group <= 4; group *= 2)
        for (unsigned chunk = 1; chunk <= rows / 4; chunk *= 2)
          for (unsigned count = 1; count <= 4; count *= 2) {
            unsigned limit =
                count == 1
                    ? 0
                    : std::min(sb->wrap.maxBytes / (count - 1),
                               64 * width - 1);
            for (unsigned step = count == 1 ? 0 : sb->wrap.granuleBytes;
                 step <= limit; step += sb->wrap.granuleBytes) {
              BufferLdsConfig candidate{rows, columns * 2, 4, width, 64,
                                        chunk, count, step};
              candidate.wavesPerRowGroup = group;
              auto expected =
                  bufferLdsBankConflictOracle(candidate, sbReads, sbBanks);
              BufferLdsConfig scored = candidate;
              bool accepted =
                  evaluateBufferLdsBankConflicts(scored, sb->lds, sbReads);
              assert(accepted == bool(expected));
              if (!expected)
                continue;
              ++validConfigs;
              auto expectedCost =
                  std::make_pair(expected->worstWay, expected->phaseService);
              assert(std::make_pair(scored.worstReadWay,
                                    scored.readPhaseCost) == expectedCost);
              observedCosts.insert(expectedCost);
              sbBest = std::min(sbBest, expectedCost);
            }
          }
      assert(validConfigs > 10 && observedCosts.size() > 1);
      auto sbSelected = selectMmacBufferLdsConfig(
          rows, columns, 2, 4, width, kContiguous, 16, 4, sb->lds,
          /*matrixRead=*/false, 16, sb->wrap);
      assert(sbSelected &&
             std::make_pair(sbSelected->worstReadWay,
                            sbSelected->readPhaseCost) == sbBest);
    }
  }
  std::cout << "PASS\n";
}
