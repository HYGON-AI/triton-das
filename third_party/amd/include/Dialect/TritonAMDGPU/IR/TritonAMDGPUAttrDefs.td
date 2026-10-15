/*
 * Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef TRITON_AMDGPU_ATTRDEFS
#define TRITON_AMDGPU_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "TritonAMDGPUDialect.td"
include "mlir/IR/EnumAttr.td"

class TritonAMDGPU_Attr<string name, list<Trait> traits = [],
                     string baseCppClass = "::mlir::Attribute">
  : AttrDef<TritonAMDGPU_Dialect, name, traits, baseCppClass> {
}

def SetFP8Clamping : TritonAMDGPU_Attr<"SetFP8Clamping"> {
  let mnemonic = "amdgcn.set.fp8.clamping";
}

class TritonAMDGPU_I32Enum<string name, string description, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, description, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::triton::amdgpu";
}

class TritonAMDGPU_I32EnumAttr<string mnemonic, TritonAMDGPU_I32Enum enumInfo> :
    EnumAttr<TritonAMDGPU_Dialect, enumInfo, mnemonic> {
  let assemblyFormat = "`<` $value `>`";
  let cppNamespace = "::mlir::triton::amdgpu";
}

def SchedHintCaseNone : I32EnumAttrCase<"none", 0>;
def SchedHintCaseAttention : I32EnumAttrCase<"attention", 2>;

def TritonAMDGPU_SchedHintsEnum : TritonAMDGPU_I32Enum<
  "SchedHint", "Instruction Scheduling Hints for AMD GPUs", [
    SchedHintCaseNone,
    SchedHintCaseAttention,
  ]>;

def TritonAMDGPU_SchedHintVariantAttr :
  TritonAMDGPU_I32EnumAttr<"SchedHintVariant", TritonAMDGPU_SchedHintsEnum>;

def TritonAMDGPU_MlsEncoding : TritonAMDGPU_Attr<"MlsEncoding"> {
  let cppNamespace = "::mlir::triton::amdgpu";
  let mnemonic = "MlsEncoding";
  let summary = "mls encoding attribute.";
  let description = [{
    The attribute holds some information about mls.matrix_load operation.
    it include two parts: mls info(to restore matrix_load info)  and warpsPerCTA info.
      opIdx       :  the operand index for dot operation.
      mlsTile     :  the mls tile shape. opIdx == 0 ? [nonKTile, kTile] : [kTile, nonKTile]
      elemBitWidth:  the share memory elem bit width, for bit 4|6, it is 8;
      elemBitTyKind: share memory elem type kind which need to distinguish. 0: no need, 1: b4, 2: b4mix, 3: i4, 4: u4, 5: f4, 6: f6;
      alt2Kind    :  the alt2 kind of the mls encoding attribute.
      version     :  the version of the mls encoding attribute.
      order       :  the order of the tensor, the fastest-changing axis first.
      warpsPerCTA :  the number of warps per CTA. warpOrder follow the order field to fit triton framework.
  }];

  let parameters = (ins "unsigned":$opIdx,  // or nonKTileIdx
                        ArrayRefParameter<"unsigned">:$mlsTile,
                        "unsigned":$elemBitWidth,
                        "unsigned":$elemBitTyKind,
                        "unsigned":$alt2Kind,
                        "unsigned":$version,
                        ArrayRefParameter<"unsigned">:$order,
                        ArrayRefParameter<"unsigned">:$warpsPerCTA);
  let assemblyFormat = "`opIdx` `=` $opIdx `,` "
                       "`mlsTile` `=` `[` $mlsTile `]` `,` "
                       "`elemBitWidth` `=` $elemBitWidth `,` "
                       "`elemBitTyKind` `=` $elemBitTyKind `,` "
                       "`alt2Kind` `=` $alt2Kind `,` "
                       "`version` `=` $version `,` "
                       "`order` `=` `[` $order `]` `,` "
                       "`warpsPerCTA` `=` `[` $warpsPerCTA `]`";
}

def TritonAMDGPU_MlsStoreEncoding : TritonAMDGPU_Attr<"MlsStoreEncoding"> {
  let cppNamespace = "::mlir::triton::amdgpu";
  let mnemonic = "MlsStoreEncoding";
  let summary = "MLS register-to-global matrix-store encoding attribute.";
  let description = [{
    Describes an MLS C-matrix store using M/N terminology. Unlike
    MlsEncoding, this attribute has no dot operand index or K-major state.
      storeTile      : matrix-store tile shape [M, N].
      elemBitWidth   : output element width.
      interleaveKind : register interleave used by the store.
      version        : MLS ISA version.
      order          : tensor order, fastest-changing axis first.
      warpsPerCTA    : MMAC-C warp distribution across M/N.
  }];

  let parameters = (ins ArrayRefParameter<"unsigned">:$storeTile,
                        "unsigned":$elemBitWidth,
                        "unsigned":$interleaveKind,
                        "unsigned":$version,
                        ArrayRefParameter<"unsigned">:$order,
                        ArrayRefParameter<"unsigned">:$warpsPerCTA);
  let assemblyFormat = "`storeTile` `=` `[` $storeTile `]` `,` "
                       "`elemBitWidth` `=` $elemBitWidth `,` "
                       "`interleaveKind` `=` $interleaveKind `,` "
                       "`version` `=` $version `,` "
                       "`order` `=` `[` $order `]` `,` "
                       "`warpsPerCTA` `=` `[` $warpsPerCTA `]`";
}

// -------------------------------- HCU Extend --------------------------------
// Scale Buffer storage is distinct from Shared/LDS and register layouts.
def ScaleMemorySpace : TritonAMDGPU_Attr<"ScaleMemorySpace"> {
  let cppNamespace = "::mlir::triton::amdgpu";
  let mnemonic = "scale_memory";
  let summary = "Dedicated Scale Buffer memory space";
  let description = [{
    Identifies Scale Buffer storage in a memdesc. ScaleEncoding describes its
    logical-to-storage mapping; scale_write produces the descriptor and
    dot_scaled reads it. Shared/LDS source storage uses a separate memdesc.
  }];
  let assemblyFormat = "";
}

def ScaleEncoding : TritonAMDGPU_Attr<"ScaleEncoding"> {
  let cppNamespace = "::mlir::triton::amdgpu";
  let mnemonic = "scale";
  let summary = "Logical scale coordinates to Scale Buffer rows and columns";
  let description = [{
    Maps a logical E8M0 scale tensor [R,S] to Scale Buffer rows and columns.
    R is M or N; S is K/32. Each Scale Buffer row contains 16 one-byte columns.
    The tensor shape belongs to memdesc; this encoding describes its storage.

    Parameters:
      blockShape = [BR,BS]: logical instTile written by one Wave64 ds2buf
                           instruction execution, not the complete CTA tile.
      rowStride:           distance between consecutive instTile starts,
                           measured in Scale Buffer rows, not bytes.
      bankXor:             0/1 SRAM-bank polarity, default 0. A and B use
                           opposite values in plans selected for scaled MMAC.

    One Wave64 instruction writes b bytes per lane, where b is 1, 2 or 4.
    Its four 16-lane phases cover g R16 groups, where g is 1, 2 or 4:

      BR = 16*g, BS = b*(4/g), BR*BS = 64*b
      rowStride = alignTo((BR/16)*BS, 8) = alignTo(4*b, 8)

    Every R16 group fills the 16 columns. Before the bank permutation below,
    consecutive S coordinates occupy consecutive rows. An instTile writes
    (BR/16)*BS rows, each containing 16 bytes; its reservation uses rowStride
    rows. The following diagrams and coordinate examples use bankXor=0:

      Logical instTile [32,8]              Scale Buffer columns
        S ->                                0       ...       15
      R 0..15,  S=0  ----------------> row 0:  S[0,0]  ... S[15,0]
      R 0..15,  S=1  ----------------> row 1:  S[0,1]  ... S[15,1]
                  ...                      ...
      R 0..15,  S=7  ----------------> row 7:  S[0,7]  ... S[15,7]
      R 16..31, S=0  ----------------> row 8:  S[16,0] ... S[31,0]
                  ...                      ...
      R 16..31, S=7  ----------------> row 15: S[16,7] ... S[31,7]

    For the complete tensor, number instTiles along S first, then R. Example:

      !ttg.memdesc<64x16xi8,
        #amdg.scale<{blockShape = [32,8], rowStride = 16}>, #amdg.scale_memory>

      Logical tensor [64,16]              Scale Buffer relative row ranges
                    S=0..7    S=8..15
                  +---------+---------+   instTile 0 -> rows  0..15
        R=0..31   | tile 0  | tile 1  |   instTile 1 -> rows 16..31
                  +---------+---------+   instTile 2 -> rows 32..47
        R=32..63  | tile 2  | tile 3  |   instTile 3 -> rows 48..63
                  +---------+---------+

    For a logical coordinate (r,s), using integer division and remainder:

      tilesAlongS = ceilDiv(S, BS)
      tileId      = (r / BR) * tilesAlongS + s / BS
      rawRow      = ((r % BR) / 16) * BS + s % BS
      b           = (BR / 16) * BS / 4
      innerRow    = b == 1 ? rawRow + 4 * (((s / 4) % 2) XOR bankXor)
                          : rawRow XOR (4 * bankXor)
      row         = tileId * rowStride + innerRow
      col         = r % 16

    In the example above, (17,4) maps to (row=12,col=1), (1,8) to
    (row=16,col=1), and (32,0) to (row=32,col=0). Rows are relative to the
    allocation start. Late allocation assigns an 8-row-aligned
    scale_write.row_base; physical row = row_base + row. This alignment
    preserves the SRAM-bank bit. The base is not part of the encoding.

    rowStride includes required Scale Buffer alignment: blockShape=[16,4]
    uses four rows within an eight-row reservation. Its four unused rows are
    before or after the written half, depending on the selected SRAM bank.
    These gaps are in Scale Buffer, not padding in source LDS.
    Edge instTiles reserve the same stride; only valid logical coordinates
    are consumed by MMAC.

    Scale Buffer has two SRAM banks, selected by (row/4)%2. bankXor is 0/1
    and reverses their roles; A and B use opposite bankXor values. For b=1,
    the four written rows use the lower/upper half of their eight-row reserved
    block according to ((s/4)%2)^bankXor. For b=2/4, XOR the relative row with
    4*bankXor and permute copy phases accordingly. Only plans whose valid rows
    satisfy bank(row)=((s/4)%2)^bankXor are selected. Thus every MMAC reads A/B
    from opposite SRAM banks, including all M/N groups and K steps.

    The attribute verifier checks the supported block geometry, exact
    rowStride and bankXor range. Compatibility with the complete tensor, LDS
    source and MMAC consumer is checked when choosing and validating plans.

    This encoding does not specify Global/LDS order, LDS swizzle, or which
    wave writes each instTile. Those are separate source layouts and copy
    scheduling rules; the destination phase/column mapping is fixed by ds2buf.
  }];
  let parameters = (ins ArrayRefParameter<"int64_t">:$blockShape,
                        "int64_t":$rowStride,
                        DefaultValuedParameter<"int64_t", "0">:$bankXor);
  let assemblyFormat = "`<` `{` `blockShape` `=` `[` $blockShape `]` `,` "
                       "`rowStride` `=` $rowStride "
                       "(`,` `bankXor` `=` $bankXor^)? `}` `>`";
  let genVerifyDecl = 1;
}

#endif
