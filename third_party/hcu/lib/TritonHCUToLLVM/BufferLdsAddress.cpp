// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include "BufferLdsAddress.h"

namespace mlir::triton::HCU {
namespace {
Value emitWrap(TritonLLVMOpBuilder &b, const BufferLdsConfig &copyConfig,
               Value offset, Value producerWave) {
  // Mirror BufferLdsConfig::wrapPhysicalByte. In particular, NMZ/YY's split
  // coarse/fine M0 field still rotates within one issued wave transfer; the
  // split is an encoding detail in buffer-load lowering, not a new address
  // mode for LDS consumers.
  unsigned mask = copyConfig.wrapPeriodBytes() - 1;
  Value rotation =
      b.mul(b.and_(producerWave, b.i32_val(copyConfig.wrapCount - 1)),
            b.i32_val(copyConfig.wrapStepBytes));
  return b.add(b.and_(offset, b.i32_val(~mask)),
               b.and_(b.add(offset, rotation), b.i32_val(mask)));
}
} // namespace

Value emitBufferLdsByteOffset(TritonLLVMOpBuilder &b,
                              const BufferLdsConfig &copyConfig, Value row,
                              Value col, unsigned elementBytes) {
  unsigned groupWaves = copyConfig.rowGroupWaves();
  unsigned groupRows = copyConfig.rows / copyConfig.waves * groupWaves;
  Value group = b.udiv(row, b.i32_val(groupRows));
  row = b.and_(row, b.i32_val(groupRows - 1));
  Value wave = b.add(b.mul(group, b.i32_val(groupWaves)),
                     b.and_(b.udiv(row, b.i32_val(copyConfig.rowsPerChunk)),
                            b.i32_val(groupWaves - 1)));
  Value localRow =
      b.add(b.mul(b.udiv(row, b.i32_val(copyConfig.rowsPerChunk * groupWaves)),
                  b.i32_val(copyConfig.rowsPerChunk)),
            b.and_(row, b.i32_val(copyConfig.rowsPerChunk - 1)));
  Value local = b.add(b.mul(localRow, b.i32_val(copyConfig.rowBytes)),
                      b.mul(col, b.i32_val(elementBytes)));
  Value offset =
      copyConfig.transferMajor
          ? b.add(
                b.mul(
                    b.udiv(local, b.i32_val(copyConfig.bytesPerWaveTransfer())),
                    b.i32_val(copyConfig.waves *
                              copyConfig.bytesPerWaveTransfer())),
                b.add(b.mul(wave, b.i32_val(copyConfig.bytesPerWaveTransfer())),
                      b.and_(local,
                             b.i32_val(copyConfig.bytesPerWaveTransfer() - 1))))
          : b.add(b.mul(wave, b.i32_val(copyConfig.totalBytesPerWave())),
                  local);
  return emitWrap(b, copyConfig, offset, wave);
}

Value emitBufferLdsWrappedByteOffset(TritonLLVMOpBuilder &b,
                                     const BufferLdsConfig &copyConfig,
                                     Value interleavedByteOffset) {
  Value tileByte =
      b.and_(interleavedByteOffset, b.i32_val(copyConfig.tileBytes() - 1));
  Value wave =
      b.udiv(tileByte, b.i32_val(copyConfig.transferMajor
                                     ? copyConfig.bytesPerWaveTransfer()
                                     : copyConfig.totalBytesPerWave()));
  if (copyConfig.transferMajor)
    wave = b.and_(wave, b.i32_val(copyConfig.waves - 1));
  return emitWrap(b, copyConfig, interleavedByteOffset, wave);
}
} // namespace mlir::triton::HCU
