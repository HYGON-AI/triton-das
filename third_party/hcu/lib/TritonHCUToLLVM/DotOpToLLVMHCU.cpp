#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/InstIterator.h"
#include "Utility.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "Dialect/TritonAMDGPU/IR/Dialect.h"

using namespace mlir;

namespace mlir::triton::HCU {
LogicalResult convertMFMA(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                          const LLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter);
LogicalResult convertScaledMFMA(triton::DotScaledOp op,
                                triton::DotScaledOp::Adaptor adaptor,
                                const LLVMTypeConverter *typeConverter,
                                ConversionPatternRewriter &rewriter);
LogicalResult convertScaledMFMA(triton::amdgpu::DotScaledOp op,
                                triton::amdgpu::DotScaledOp::Adaptor adaptor,
                                const LLVMTypeConverter *typeConverter,
                                ConversionPatternRewriter &rewriter);
LogicalResult convertScaleWrite(triton::amdgpu::ScaleWriteOp op,
                                triton::amdgpu::ScaleWriteOp::Adaptor adaptor,
                                const LLVMTypeConverter *converter,
                                ConversionPatternRewriter &rewriter);
} // namespace mlir::triton::HCU

namespace {

// HCU-specific DotOp conversion that uses HCU::convertMFMA (with MMAC support).
struct HCUDotOpConversion : public ConvertOpToLLVMPattern<triton::DotOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto D = op.getResult();
    auto dEncoding = cast<RankedTensorType>(D.getType()).getEncoding();
    if (!isa<mlir::triton::gpu::AMDMfmaEncodingAttr>(dEncoding))
      return failure();
    return mlir::triton::HCU::convertMFMA(op, adaptor, getTypeConverter(),
                                          rewriter);
  }
};

template <typename ScaledDotOp>
struct ScaledDotOpConversion : public ConvertOpToLLVMPattern<ScaledDotOp> {
  using Base = ConvertOpToLLVMPattern<ScaledDotOp>;
  using OpAdaptor = typename Base::OpAdaptor;
  using Base::Base;
  LogicalResult
  matchAndRewrite(ScaledDotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto D = op.getResult();
    auto dEncoding = cast<RankedTensorType>(D.getType()).getEncoding();
    if (!isa<mlir::triton::gpu::AMDMfmaEncodingAttr>(dEncoding))
      return failure();
    return mlir::triton::HCU::convertScaledMFMA(
        op, adaptor, this->getTypeConverter(), rewriter);
  }
};

struct ScaleWriteOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::ScaleWriteOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(triton::amdgpu::ScaleWriteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return mlir::triton::HCU::convertScaleWrite(op, adaptor, getTypeConverter(),
                                                rewriter);
  }
};

// Export the m0 setup associated with each scaled MMAC to LLVM IR.
//
// Scaled MMAC takes its A/B Scale Buffer row bases from the physical m0
// register. MMAC lowering computes that i32 value and attaches it to the
// intrinsic call as an "amdg.m0" operand bundle, preserving its SSA dependency
// without changing the intrinsic's arguments. An "amdg.m0" unit attribute on
// the enclosing LLVM function selects this AMD dialect translation hook.
//
// LLVM's write_register intrinsic needs register-name metadata (!{!"m0"}).
// The generic MLIR call_intrinsic cannot supply that metadata argument, so
// this hook runs after the whole function and its PHI nodes are translated.
// It builds a register write immediately before each bundled MMAC, removes
// the temporary bundle, and replaces the call's LLVM uses. At this point no
// further lookup of the function's MLIR values is needed: ModuleTranslation's
// mapValue API is write-once and must not be used to overwrite an existing
// mapping. Only the final LLVM instruction sequence is changed here.
//
// Example LLVM IR, with MMAC arguments abbreviated:
//   Before this hook:
//     %r = call <4 x float> @llvm.hcu.mmac.scale...(...)
//          [ "amdg.m0"(i32 %m0) ]
//   After this hook:
//     call void @llvm.write_register.i32(metadata !0, i32 %m0)
//     %r = call <4 x float> @llvm.hcu.mmac.scale...(...)
//     !0 = !{!"m0"}
//
// Ordinary intrinsic translation handles MMAC itself; the LLVM backend selects
// the instruction that sets m0. No read_register call is needed here.
class HCUM0Translation : public mlir::LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;
  LogicalResult
  amendOperation(Operation *op, ArrayRef<llvm::Instruction *>,
                 NamedAttribute attribute,
                 LLVM::ModuleTranslation &translation) const override {
    if (attribute.getName().strref() != "amdg.m0")
      return success();
    auto func = dyn_cast<LLVM::LLVMFuncOp>(op);
    if (!func)
      return op->emitError("amdg.m0 requires an LLVM function");
    auto *llvmFunc = translation.lookupFunction(func.getName());
    for (auto &inst :
         llvm::make_early_inc_range(llvm::instructions(llvmFunc))) {
      auto *call = llvm::dyn_cast<llvm::CallInst>(&inst);
      if (!call)
        continue;
      auto bundle = call->getOperandBundle("amdg.m0");
      if (!bundle)
        continue;
      if (bundle->Inputs.size() != 1 ||
          !bundle->Inputs[0]->getType()->isIntegerTy(32))
        return op->emitError("amdg.m0 requires one i32 operand in its bundle");
      llvm::IRBuilder<> builder(call);
      auto &ctx = builder.getContext();
      auto *name = llvm::MetadataAsValue::get(
          ctx, llvm::MDNode::get(ctx, llvm::MDString::get(ctx, "m0")));
      builder.CreateIntrinsic(llvm::Intrinsic::write_register,
                              {builder.getInt32Ty()},
                              {name, bundle->Inputs[0]});
      auto *plain = llvm::CallBase::removeOperandBundle(
          call, bundle->getTagID(), call->getIterator());
      call->replaceAllUsesWith(plain);
      call->eraseFromParent();
    }
    return success();
  }
};

} // namespace

namespace mlir::triton::HCU {
void populateHCUDotOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                    RewritePatternSet &patterns,
                                    PatternBenefit benefit) {
  patterns.add<HCUDotOpConversion, ScaledDotOpConversion<triton::DotScaledOp>,
               ScaledDotOpConversion<triton::amdgpu::DotScaledOp>,
               ScaleWriteOpConversion>(typeConverter, benefit);
}

void registerHCUM0Translation(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *, triton::amdgpu::TritonAMDGPUDialect *dialect) {
        dialect->addInterfaces<HCUM0Translation>();
      });
}
} // namespace mlir::triton::HCU
