// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#ifndef TRITON_HCU_LDS_READ_MODEL_H
#define TRITON_HCU_LDS_READ_MODEL_H

// Pure instruction semantics: no MLIR, pipeline policy or producer layout.
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <set>
#include <string_view>
#include <vector>

namespace mlir::triton::HCU {

enum class MatrixReadOpcode {
  M32x64B4, M32x32B8, M32x16B16, M32x16B16Alt, M32x8B32,
  M32x32B8Alt2, M64x16B8Alt4, M16x16B32, Mt16x16B32,
  Mt16x32B16Alt2
};

struct MatrixReadInstruction {
  MatrixReadOpcode opcode;
  const char *intrinsic;
  unsigned rows, columns, elementBits;
  bool transpose;
};

// Inventory from llvm/IR/IntrinsicsHCU.td. Inclusion is NOT a declaration of
// target availability, known phase semantics, or eligibility for planning.
inline constexpr MatrixReadInstruction matrixReadInstructions[] = {
    {MatrixReadOpcode::M32x64B4, "llvm.hcu.ds.read.m32x64.b4", 32, 64, 4, false},
    {MatrixReadOpcode::M32x32B8, "llvm.hcu.ds.read.m32x32.b8", 32, 32, 8, false},
    {MatrixReadOpcode::M32x16B16, "llvm.hcu.ds.read.m32x16.b16", 32, 16, 16, false},
    {MatrixReadOpcode::M32x16B16Alt, "llvm.hcu.ds.read.m32x16.b16.alt", 32, 16, 16, false},
    {MatrixReadOpcode::M32x8B32, "llvm.hcu.ds.read.m32x8.b32", 32, 8, 32, false},
    {MatrixReadOpcode::M32x32B8Alt2, "llvm.hcu.ds.read.m32x32.b8.alt2", 32, 32, 8, false},
    {MatrixReadOpcode::M64x16B8Alt4, "llvm.hcu.ds.read.m64x16.b8.alt4", 64, 16, 8, false},
    {MatrixReadOpcode::M16x16B32, "llvm.hcu.ds.read.m16x16.b32", 16, 16, 32, false},
    {MatrixReadOpcode::Mt16x16B32, "llvm.hcu.ds.read.mt16x16.b32", 16, 16, 32, true},
    {MatrixReadOpcode::Mt16x32B16Alt2, "llvm.hcu.ds.read.mt16x32.b16.alt2", 16, 32, 16, true},
};

inline const MatrixReadInstruction &matrixReadInstruction(MatrixReadOpcode op) {
  for (const auto &instruction : matrixReadInstructions)
    if (instruction.opcode == op)
      return instruction;
  // Callers use the closed enum, never a parsed integer.
  std::abort();
}

inline bool hasLegacyMatrixReadProfile(std::string_view arch,
                                       MatrixReadOpcode opcode) {
  if (arch == "gfx936")
    return opcode <= MatrixReadOpcode::M32x8B32;
  if (arch == "gfx92a" || arch == "gfx938" || arch == "gfx946")
    return opcode == MatrixReadOpcode::M32x16B16 ||
           opcode == MatrixReadOpcode::M32x32B8;
  return false;
}

// Legacy MMAC fragment compatibility only. The caller must separately prove
// target availability, shared addressing and complete tile coverage. In
// particular, an unmodelled alt/transpose variant must not inherit this map.
inline std::optional<MatrixReadOpcode>
selectLegacyMatrixRead(unsigned elementBits, unsigned instructionNonK,
                       unsigned kWidth) {
  if (instructionNonK != 16)
    return std::nullopt;
  if (elementBits == 16 && kWidth == 4)
    return MatrixReadOpcode::M32x16B16;
  if (elementBits == 8 && kWidth == 8)
    return MatrixReadOpcode::M32x32B8;
  return std::nullopt;
}

// Ordinary legacy MMAC: lowering cannot vectorize beyond the producer's copy
// width. A narrower copy splits the K fragment into several LDS instructions,
// each with its own phase table; it is not a b64 read with a smaller payload.
inline unsigned legacyContiguousReadBytes(unsigned elementBytes,
                                         unsigned kWidth,
                                         unsigned copyBytes) {
  return std::min(elementBytes * kWidth, copyBytes);
}

// An instruction can issue multiple (possibly discontiguous) spans per source
// lane. Bit units preserve b4 and sub-word accesses without rounding away data.
struct LdsBitSpan {
  uint64_t begin;
  unsigned size;
};
struct LdsPhaseRequest {
  unsigned sourceLane;
  LdsBitSpan address;
};
struct LdsResultPiece {
  unsigned destinationLane, registerIndex, registerBit;
  unsigned sourceLane;
  LdsBitSpan source;
};
enum class LdsBroadcast { None, SameBankWord };
struct LdsBankGeometry {
  unsigned banks, bankBits, waveSize;
};
struct LdsReadTrace {
  // Explicit requests, rather than a partition of lane IDs: a wide read may
  // service the same lane in several phases.
  std::vector<std::vector<LdsPhaseRequest>> phases;
  std::vector<LdsResultPiece> results;
  LdsBroadcast broadcast = LdsBroadcast::None;
};
struct LdsBankCost {
  unsigned worstWay = 0;
  uint64_t phaseService = 0;
};

// Source-element permutation measured by lds_instruction_probe.hip. Input
// bits are (element within four VGPRs, destination lane), least significant
// first. This describes routing ONLY: it cannot establish issue phases or
// authorize another GPU's instruction selection.
inline std::optional<std::vector<unsigned>>
matrixReadResultBasis(std::string_view arch, MatrixReadOpcode opcode) {
  if (!hasLegacyMatrixReadProfile(arch, opcode))
    return std::nullopt;
  switch (opcode) {
  case MatrixReadOpcode::M32x64B4:
    return {{32, 64, 128, 256, 16, 1, 2, 4, 8, 512, 1024}};
  case MatrixReadOpcode::M32x32B8:
    return {{32, 64, 128, 16, 1, 2, 4, 8, 256, 512}};
  case MatrixReadOpcode::M32x16B16:
    return {{32, 64, 16, 1, 2, 4, 8, 128, 256}};
  case MatrixReadOpcode::M32x16B16Alt:
    return {{32, 64, 1, 2, 4, 8, 16, 128, 256}};
  case MatrixReadOpcode::M32x8B32:
    return {{32, 16, 1, 2, 4, 8, 64, 128}};
  default:
    return std::nullopt;
  }
}

// Translate the instruction's routing into actual source addresses. Each
// source lane supplies one aligned 16-byte span; callers independently provide
// a validated phase model when constructing LdsReadTrace.
inline std::optional<std::vector<LdsResultPiece>>
matrixReadResults(std::string_view arch, MatrixReadOpcode opcode,
                  const std::vector<uint64_t> &sourceBitAddresses) {
  auto basis = matrixReadResultBasis(arch, opcode);
  if (!basis || sourceBitAddresses.size() != 64)
    return std::nullopt;
  for (uint64_t address : sourceBitAddresses)
    if (address % 128 || address > UINT64_MAX - 127)
      return std::nullopt;
  unsigned bits = matrixReadInstruction(opcode).elementBits;
  unsigned elements = 128 / bits;
  std::vector<LdsResultPiece> results;
  for (unsigned lane = 0; lane < 64; ++lane)
    for (unsigned item = 0; item < elements; ++item) {
      unsigned index = lane * elements + item, source = 0;
      for (unsigned bit = 0; bit < basis->size(); ++bit)
        if (index & (1u << bit))
          source ^= (*basis)[bit];
      unsigned sourceLane = source / elements;
      results.push_back({lane, item * bits / 32, item * bits % 32,
                         sourceLane,
                         {sourceBitAddresses[sourceLane] +
                              (source % elements) * bits, bits}});
    }
  return results;
}

// A read2 has two independently addressed components, NOT a single wider
// contiguous span. The caller supplies the target-validated component phase
// table. ST64 changes offset units only (64 * componentBytes).
inline std::optional<LdsReadTrace> pairedReadTrace(
    const std::vector<uint64_t> &laneByteAddresses, unsigned componentBytes,
    unsigned offset0, unsigned offset1, bool st64,
    const std::vector<std::vector<unsigned>> &componentPhases,
    LdsBroadcast broadcast) {
  if ((componentBytes != 4 && componentBytes != 8) ||
      offset0 > 255 || offset1 > 255 || laneByteAddresses.empty() ||
      componentPhases.empty())
    return std::nullopt;
  std::vector<unsigned> covered(laneByteAddresses.size());
  for (const auto &phase : componentPhases) {
    if (phase.empty())
      return std::nullopt;
    for (unsigned lane : phase)
      if (lane >= covered.size() || ++covered[lane] != 1)
        return std::nullopt;
  }
  if (std::find(covered.begin(), covered.end(), 0) != covered.end())
    return std::nullopt;
  LdsReadTrace trace;
  trace.broadcast = broadcast;
  for (unsigned component = 0; component < 2; ++component) {
    uint64_t offset = uint64_t(component ? offset1 : offset0) *
                      componentBytes * (st64 ? 64 : 1);
    for (const auto &phase : componentPhases) {
      trace.phases.emplace_back();
      for (unsigned lane : phase) {
        uint64_t base = laneByteAddresses[lane];
        if (base % componentBytes ||
            base > UINT64_MAX / 8 - offset - componentBytes)
          return std::nullopt;
        uint64_t address = (base + offset) * 8;
        trace.phases.back().push_back({lane, {address, componentBytes * 8}});
        for (unsigned word = 0; word < componentBytes / 4; ++word)
          trace.results.push_back(
              {lane, component * (componentBytes / 4) + word, 0, lane,
               {address + word * 32, 32}});
      }
    }
  }
  return trace;
}

// The caller translates logical spans to PHYSICAL LDS bit addresses first.
// Broadcast is an explicit instruction/target property, never inferred from
// element type. This function does not claim the supplied phases are hardware
// validated; target profiles own that proof.
inline std::optional<LdsBankCost>
scoreLdsReadTrace(const LdsReadTrace &trace, LdsBankGeometry hw) {
  if (!hw.banks || !hw.bankBits || !hw.waveSize || trace.phases.empty())
    return std::nullopt;
  LdsBankCost cost;
  for (const auto &phase : trace.phases) {
    if (phase.empty())
      return std::nullopt;
    std::vector<unsigned> requests(hw.banks);
    std::vector<std::pair<unsigned, uint64_t>> words;
    for (const auto &request : phase) {
      auto span = request.address;
      if (request.sourceLane >= hw.waveSize || !span.size ||
          span.begin > std::numeric_limits<uint64_t>::max() - (span.size - 1))
        return std::nullopt;
      uint64_t first = span.begin / hw.bankBits;
      uint64_t last = (span.begin + span.size - 1) / hw.bankBits;
      for (uint64_t word = first;; ++word) {
        words.emplace_back(trace.broadcast == LdsBroadcast::SameBankWord
                               ? 0 : request.sourceLane, word);
        if (word == last)
          break;
      }
    }
    std::sort(words.begin(), words.end());
    words.erase(std::unique(words.begin(), words.end()), words.end());
    for (auto word : words)
      ++requests[word.second % hw.banks];
    unsigned way = *std::max_element(requests.begin(), requests.end());
    cost.worstWay = std::max(cost.worstWay, way);
    cost.phaseService += way;
  }
  return cost;
}

// Validate an explicitly supplied result permutation independently of bank
// scheduling. A result may duplicate source bits (broadcast), but two pieces
// may not overwrite the same destination bit. Unknown mappings remain empty
// and cannot pass this validator.
inline bool validateLdsResultMapping(const LdsReadTrace &trace,
                                     LdsBankGeometry hw,
                                     unsigned registersPerLane,
                                     unsigned registerBits = 32) {
  if (!registerBits || !registersPerLane || trace.results.empty() ||
      !scoreLdsReadTrace(trace, hw))
    return false;
  std::set<std::pair<unsigned, uint64_t>> sourceBits, destinationBits;
  for (const auto &phase : trace.phases)
    for (const auto &request : phase)
      for (unsigned bit = 0; bit < request.address.size; ++bit)
        sourceBits.emplace(request.sourceLane, request.address.begin + bit);
  for (const auto &piece : trace.results) {
    if (piece.destinationLane >= hw.waveSize ||
        piece.registerIndex >= registersPerLane ||
        piece.registerBit >= registerBits || !piece.source.size ||
        piece.source.size > registerBits - piece.registerBit ||
        piece.source.begin > UINT64_MAX - (piece.source.size - 1))
      return false;
    for (unsigned bit = 0; bit < piece.source.size; ++bit) {
      if (!sourceBits.count({piece.sourceLane, piece.source.begin + bit}) ||
          !destinationBits.emplace(
              piece.destinationLane, uint64_t(piece.registerIndex) * registerBits +
                                         piece.registerBit + bit).second)
        return false;
    }
  }
  return destinationBits.size() ==
         uint64_t(hw.waveSize) * registersPerLane * registerBits;
}

} // namespace mlir::triton::HCU
#endif
