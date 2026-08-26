//===----------------------------------------------------------------------===//
//
// ApplyFuncHintsAndClean — materialize `tt.hint.func.*` before LLVM conversion,
// then strip leftover FuncOp / ModuleOp hint attrs.
//
// Responsibilities:
//   1. `tt.hint.func.args_ptr.no_alias` → set `llvm.noalias` on pointer args
//      (opt-in; default OFF unless the attr is present and true)
//   2. `tt.hint.func.attr.<name>` → append to FuncOp `passthrough`, so
//      convertFuncOpToLLVMFuncOp can emit LLVM function attributes
//   3. Strip remaining `tt.hint.*` attrs from each FuncOp after consumption
//   4. Strip `tt.hint.mod.flag.*` attrs from the ModuleOp (they live on the
//      module, not on FuncOp)
//
// Value-level (`tt.hint.val.*`) hints are not materialized here; they disappear
// with rewrite as usual. Any `mod.flag.*` consumer must run before this pass.
//
//===----------------------------------------------------------------------===//

#include "TritonHCU/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include <string>

using namespace mlir;
using namespace mlir::triton;

namespace mlir {

#define GEN_PASS_DEF_TRITONHCUAPPLYFUNCHINTSANDCLEAN
#include "TritonHCU/Passes.h.inc"

namespace {

// FuncOp-level hint keys / prefixes.
constexpr StringRef kHintPrefix = "tt.hint.";
constexpr StringRef kFuncAttrPrefix = "tt.hint.func.attr.";
constexpr StringRef kArgsPtrNoAlias = "tt.hint.func.args_ptr.no_alias";
// ModuleOp-level stamp prefix (frontend: set_module_attr).
constexpr StringRef kModFlagPrefix = "tt.hint.mod.flag.";

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Read a BoolAttr named `name` from `op`, or return `defaultVal` if absent.
static bool getBoolAttrOr(Operation *op, StringRef name, bool defaultVal) {
  if (auto attr = op->getAttrOfType<BoolAttr>(name))
    return attr.getValue();
  return defaultVal;
}

/// Whether pointer-arg noalias should be applied.
/// Default OFF: the attr must be present and explicitly true.
static bool shouldEnableNoAlias(FuncOp func) {
  if (!func->hasAttr(kArgsPtrNoAlias))
    return false;
  return getBoolAttrOr(func, kArgsPtrNoAlias, /*defaultVal=*/false);
}

/// Remove attrs whose names start with `prefix` from `op`.
static void dropAttrsWithPrefix(Operation *op, StringRef prefix) {
  SmallVector<StringAttr> toRemove;
  for (NamedAttribute named : op->getAttrs()) {
    if (named.getName().strref().starts_with(prefix))
      toRemove.push_back(named.getName());
  }
  for (StringAttr name : toRemove)
    op->removeAttr(name);
}

//===----------------------------------------------------------------------===//
// Hint materialization
//===----------------------------------------------------------------------===//

/// Mark every pointer argument of `func` with LLVM `noalias` (UnitAttr).
static void applyArgsPtrNoAlias(FuncOp func) {
  if (!shouldEnableNoAlias(func))
    return;

  MLIRContext *ctx = func.getContext();
  for (unsigned i = 0, e = func.getNumArguments(); i < e; ++i) {
    if (!isa<PointerType>(func.getArgument(i).getType()))
      continue;
    func.setArgAttr(i, LLVM::LLVMDialect::getNoAliasAttrName(),
                    UnitAttr::get(ctx));
  }
}

/// Fold `tt.hint.func.attr.*` into FuncOp `passthrough`.
///
/// Encoding rules (matches convertFuncOpToLLVMFuncOp expectations):
///   - BoolAttr true  → unit-style entry:  "attr"
///   - BoolAttr false → drop (do not emit)
///   - IntegerAttr    → key/value entry:   [["attr", "N"]]
///   - StringAttr     → key/value entry:   [["attr", "str"]]
///
/// Consumed `tt.hint.func.attr.*` attrs are removed from the FuncOp.
static void materializeFuncAttrs(FuncOp func) {
  MLIRContext *ctx = func.getContext();

  // Preserve any existing passthrough entries.
  SmallVector<Attribute> passThru;
  if (auto existing = func->getAttrOfType<ArrayAttr>("passthrough"))
    passThru.append(existing.begin(), existing.end());

  SmallVector<StringAttr> toRemove;
  for (NamedAttribute named : func->getAttrs()) {
    StringRef name = named.getName().strref();
    if (!name.starts_with(kFuncAttrPrefix))
      continue;

    StringRef llvmAttrName = name.drop_front(kFuncAttrPrefix.size());
    Attribute value = named.getValue();

    if (auto b = dyn_cast<BoolAttr>(value)) {
      if (b.getValue())
        passThru.push_back(StringAttr::get(ctx, llvmAttrName));
    } else if (auto i = dyn_cast<IntegerAttr>(value)) {
      passThru.push_back(ArrayAttr::get(
          ctx, {StringAttr::get(ctx, llvmAttrName),
                StringAttr::get(ctx, std::to_string(i.getInt()))}));
    } else if (auto s = dyn_cast<StringAttr>(value)) {
      passThru.push_back(
          ArrayAttr::get(ctx, {StringAttr::get(ctx, llvmAttrName), s}));
    }

    toRemove.push_back(named.getName());
  }

  for (StringAttr name : toRemove)
    func->removeAttr(name);

  if (!passThru.empty())
    func->setAttr("passthrough", ArrayAttr::get(ctx, passThru));
}

/// Apply FuncOp-level hint materializations, then strip FuncOp `tt.hint.*`.
static void applyFuncHints(FuncOp func) {
  applyArgsPtrNoAlias(func);
  materializeFuncAttrs(func);
  dropAttrsWithPrefix(func, kHintPrefix);
}

/// Drop ModuleOp-level `tt.hint.mod.flag.*` stamps.
static void cleanModFlagAttrs(ModuleOp mod) {
  dropAttrsWithPrefix(mod, kModFlagPrefix);
}

} // namespace

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

class ApplyFuncHintsAndCleanPass
    : public impl::TritonHCUApplyFuncHintsAndCleanBase<
          ApplyFuncHintsAndCleanPass> {
public:
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    mod.walk([&](FuncOp func) { applyFuncHints(func); });
    // mod.flag.* lives on the ModuleOp, not on FuncOp.
    cleanModFlagAttrs(mod);
  }
};

} // namespace mlir
