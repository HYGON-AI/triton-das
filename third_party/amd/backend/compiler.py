from triton.backends.compiler import BaseBackend, GPUTarget, AttrsDescriptor, register_descriptor
from triton._C.libtriton import ir, passes, llvm, amd
from dataclasses import dataclass
from typing import Any, Dict, Tuple
from types import ModuleType
import hashlib
import tempfile
import os
import re
import subprocess
import functools
from pathlib import Path


def min_dot_size(target: GPUTarget):
    arch_str = target.arch
    # CDNA 3.0 supports k==8 in all mfma variants except for int8
    # (where the smallest `k` supported is 16)
    if "gfx94" in arch_str:
        return lambda lhsType, rhsType: (16, 16, 16) if (lhsType.is_int8() or rhsType.is_int8()) else (16, 16, 8)
    # CDNA 2.0 always supports `k==8`
    if "gfx9" in arch_str:
        return lambda lhsType, rhsType: (16, 16, 8)
    # Other architectures will only support 16,16,16
    return lambda lhsType, rhsType: (16, 16, 16)


@dataclass(frozen=True)
class HIPOptions:
    num_warps: int = 4
    waves_per_eu: int = 1
    num_stages: int = 2
    num_ctas: int = 1
    extern_libs: dict = None
    cluster_dims: tuple = (1, 1, 1)
    debug: bool = False
    sanitize_overflow: bool = True
    arch: str = None
    supported_fp8_dtypes: Tuple[str] = ("fp8e5", )
    deprecated_fp8_dtypes: Tuple[str] = ()
    default_dot_input_precision: str = "ieee"
    allowed_dot_input_precisions: Tuple[str] = ("ieee", )
    enable_fp_fusion: bool = True
    matrix_instr_nonkdim: int = 0
    kpack: int = 1
    allow_flush_denorm: bool = False
    max_num_imprecise_acc_default: int = 0
    backend_name: str = 'hip'

    # The following option provides hints to the AMDGPU backend regarding instruction scheduling
    # for all `tt.dot` operations in a kernel. The "default" variant preserves the default
    # instruction scheduling of the AMDGPU backend which aims at maximizing occupancy.
    # The option is experimental and may change at any time regarding its semantics and/or may
    # be gone entirely anytime.
    instruction_sched_variant: str = 'default'

    def __post_init__(self):
        default_libdir = Path(__file__).parent / 'lib'
        extern_libs = {} if self.extern_libs is None else dict(self.extern_libs)
        # Ignore user-defined warp size for gfx9
        warp_size = 32 if 'gfx10' in self.arch or 'gfx11' in self.arch or 'gfx12' in self.arch else 64
        object.__setattr__(self, 'warp_size', warp_size)
        libs = ["ocml", "ockl"]
        for lib in libs:
            extern_libs[lib] = str(default_libdir / f'{lib}.bc')
        object.__setattr__(self, 'extern_libs', tuple(extern_libs.items()))
        assert self.num_warps > 0 and (self.num_warps & (self.num_warps - 1)) == 0, \
               "num_warps must be a power of 2"

    def hash(self):
        key = '_'.join([f'{name}-{val}' for name, val in self.__dict__.items()])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


@register_descriptor
class HIPAttrsDescriptor(AttrsDescriptor):
    # This property asserts if the underlying storage area of a given pointer
    # can be resepresented as a 32 bit integer. When this is true, we can be
    # sure that all indices into the tensor behind that pointer can use 32-bit
    # indexing. That opens the door for the AMD backend to use buffer load/store
    # instrinsics, which requires this property. Buffer load/store intrinsics
    # gives direct out-of-bound support and simplifies index calculation for
    # lower register pressure.
    __slots__ = ("pointer_range_32")

    def _add_backend_properties(self, params=None, values=None):
        self.property_values["tt.pointer_range"] = 32
        if params is None or values is None:
            return

        self.arg_properties["tt.pointer_range"] = [
            param.num for param, arg in zip(params, values) if HIPAttrsDescriptor.is_within2gb(arg)
            and not param.do_not_specialize and not param.do_not_specialize_on_alignment
        ]

    @staticmethod
    def is_within2gb(arg):
        if hasattr(arg, "ptr_range"):
            return arg.ptr_range() <= 2**31 - 1
        if "torch.Tensor" in str(type(arg)) and hasattr(arg, "untyped_storage"):
            # Please note that 2**31-1 is the max int32 positive limit
            return arg.untyped_storage().size() <= 2**31 - 1
        return False

    @staticmethod
    def get_property_key(val, align):
        generic_key = AttrsDescriptor.get_property_key(val, align)
        hip_key = "S" if HIPAttrsDescriptor.is_within2gb(val) else "N"
        key = (generic_key + hip_key).replace("N", "")
        return key if key else "N"


class HIPBackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'hip'

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        assert isinstance(target.arch, str)
        self.binary_ext = "hsaco"

    def parse_options(self, opts) -> Any:
        args = {'arch': self.target.arch}

        if "supported_fp8_dtypes" not in opts:
            supported_fp8_dtypes = set(HIPOptions.supported_fp8_dtypes)
            if self.target.arch in ('gfx940', 'gfx941', 'gfx942'):
                supported_fp8_dtypes.update({'fp8e4b8', 'fp8e5b16'})
            args["supported_fp8_dtypes"] = tuple(sorted(supported_fp8_dtypes))

        if "enable_fp_fusion" not in opts:
            args["enable_fp_fusion"] = os.getenv("TRITON_DEFAULT_FP_FUSION", "1") == "1"
        args.update({k: opts[k] for k in HIPOptions.__dataclass_fields__.keys() if k in opts})
        return HIPOptions(**args)

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
            metadata.cluster_dims[0],
            metadata.cluster_dims[1],
            metadata.cluster_dims[2],
        )

    def get_codegen_implementation(self):
        codegen_fns = {"min_dot_size": min_dot_size(self.target)}
        return codegen_fns

    def get_module_map(self) -> Dict[str, ModuleType]:
        from triton.language.extra.hip import libdevice
        return {"triton.language.extra.libdevice": libdevice}

    def load_dialects(self, ctx):
        amd.load_dialects(ctx)

    def get_attrs_descriptor(self, params, args):
        return HIPAttrsDescriptor(params, args)

    @staticmethod
    def compute_spec_key(arg, align):
        return HIPAttrsDescriptor.get_property_key(arg, align)

    @staticmethod
    def path_to_rocm_lld():
        # Check env path for ld.lld
        lld_env_path = os.getenv("TRITON_HIP_LLD_PATH")
        if lld_env_path is not None:
            lld = Path(lld_env_path)
            if lld.is_file():
                return lld
        # Check backend for ld.lld (used for pytorch wheels)
        lld = Path(__file__).parent / "llvm/bin/ld.lld"
        if lld.is_file():
            return lld
        lld = Path("/opt/rocm/llvm/bin/ld.lld")
        if lld.is_file():
            return lld
        lld = Path("/usr/bin/ld.lld")
        if lld.is_file():
            return lld
        raise Exception("ROCm linker /opt/rocm/llvm/bin/ld.lld not found. Set 'TRITON_HIP_LLD_PATH' to its path.")

    @staticmethod
    def path_to_rocm_clang():
        # Check env path for clang
        clang_env_path = os.getenv("TRITON_HIP_CLANG_PATH")
        if clang_env_path is not None:
            clang = Path(clang_env_path)
            if clang.is_file():
                return clang
        clang = Path("/opt/rocm/llvm/bin/clang")
        if clang.is_file():
            return clang
        clang = Path("/usr/bin/clang")
        if clang.is_file():
            return clang
        raise Exception("ROCm compiler /opt/rocm/llvm/bin/clang not found. Set 'TRITON_HIP_CLANG_PATH' to its path.")

    @staticmethod
    def make_ttir(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_pointer(pm)
        passes.ttir.add_combine(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod)
        return mod

    @staticmethod
    def make_ttgir(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.ttir.add_convert_to_ttgpuir(pm, f"hip:{options.arch}", options.num_warps, options.warp_size,
                                           options.num_ctas)
        pm.run(mod)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.ttgpuir.add_coalesce(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_thread_locality(pm)
        amd.passes.ttgpuir.add_accelerate_matmul(pm, options.arch, options.matrix_instr_nonkdim, options.kpack)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        amd.passes.ttgpuir.add_optimize_epilogue(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm, True)
        if amd.has_matrix_core_feature(options.arch):
            assert options.num_stages != 0, ("Triton AMD backend pipeliner has been updated. "
                                             "We used to trigger software pipelining with "
                                             "num_stages == 0. Now it will not happen anymore; "
                                             "please update to use num_stages == 2 for "
                                             "equivalent behavior in the past.")
            amd.passes.ttgpuir.add_stream_pipelinev2(pm, options.num_stages)
            passes.common.add_canonicalizer(pm)
        amd.passes.ttgpuir.insert_instruction_sched_hints(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm, True)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_reduce_data_duplication(pm)
        if amd.has_matrix_core_feature(options.arch):
            amd.passes.ttgpuir.add_reorder_instructions(pm)
        amd.passes.ttgpuir.add_canonicalize_pointers(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        pm.run(mod)
        return mod

    @staticmethod
    def make_llir(src, metadata, options):
        mod = src
        # TritonGPU -> LLVM-IR (MLIR)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        amd.passes.ttgpuir.add_decompose_unsupported_conversions(pm, options.arch)
        # custom_lds_size is an experimental parameter that defines amount of LDS available
        # for one thread block. Measured in bytes.
        #
        # If custom_lds_size = 0, pass will consider all LDS is available for one threads block,
        # LDS size is determined by provided arch name.
        custom_lds_size = 0
        amd.passes.ttgpuir.add_optimize_lds_usage(pm, options.arch, custom_lds_size)
        passes.convert.add_scf_to_cf(pm)
        passes.convert.add_index_to_llvmir(pm)

        passes.ttgpuir.add_allocate_shared_memory(pm)
        ## __HIP_FTZ is used to control the denorm flushing behavior of exp2 op as follows:
        ## 1. If __HIP_FTZ = 1, exp2 flushes denorms in input and output regardless
        ##    of the value of kernel arg `allow_flush_denorm`.
        ## 2. If __HIP_FTZ = 0, whether exp2 flushes denorms in input and output
        ##    depends on the value of kernel arg `allow_flush_denorm`.
        ## 3. __HIP_FTZ is default to 1 and not exposed as a kernel argument.
        ##    For now it is used as a controller for developers only.
        __HIP_FTZ = True
        amd.passes.ttgpuir.add_to_llvmir(pm, options.arch, __HIP_FTZ)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)

        passes.convert.add_cf_to_llvmir(pm)
        passes.convert.add_arith_to_llvmir(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        amd.passes.ttgpuir.lower_instruction_sched_hints(pm, options.instruction_sched_variant)
        if os.environ.get("TRITON_DISABLE_LINE_INFO", "0") == "0":
            passes.llvmir.add_di_scope(pm)
        # This pass (`add_builtin_func_to_llvmir`) serves as a temporary workaround to address the issue of excessive basic block
        # count caused by predicated loads/stores. In certain kernels, the addition of these blocks can cause the MLIR
        # canonicalizer to never finish when attempting to merge blocks. The permanent solution under consideration
        # involves using MUBUF instructions that have built-in out-of-bounds checks, which would eliminate the need
        # for conditional branching around memory accesses.
        amd.passes.ttgpuir.add_builtin_func_to_llvmir(pm)
        pm.run(mod)

        # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
        llvm.init_targets()
        context = llvm.context()
        llvm_mod = llvm.to_module(mod, context)
        amd.attach_target_triple(llvm_mod)
        llvm.attach_datalayout(llvm_mod, amd.TARGET_TRIPLE, options.arch, '')

        # Set various control constants on the LLVM module so that device
        # libraries can resolve references to them.
        amd.set_isa_version(llvm_mod, options.arch)
        amd.set_abi_version(llvm_mod, 500)
        amd.set_bool_control_constant(llvm_mod, "__oclc_finite_only_opt", False)
        amd.set_bool_control_constant(llvm_mod, "__oclc_correctly_rounded_sqrt32", True)
        amd.set_bool_control_constant(llvm_mod, "__oclc_unsafe_math_opt", False)
        amd.set_bool_control_constant(llvm_mod, "__oclc_wavefrontsize64", options.warp_size == 64)

        # Set kernel attributes first given this may affect later optimizations.
        fns = [fn for fn in llvm_mod.get_functions() if not fn.is_declaration()]
        # The public kernel should be kernel 0.
        fns[0].set_calling_conv(amd.CALLING_CONV_AMDGPU_KERNEL)
        fns[0].add_fn_attr("amdgpu-flat-work-group-size", f"1,{options.num_warps*options.warp_size}")
        fns[0].add_fn_attr("amdgpu-waves-per-eu", f"{options.waves_per_eu}")
        denormal_mode = "preserve-sign" if options.allow_flush_denorm else "ieee"
        fns[0].add_fn_attr("denormal-fp-math-f32", denormal_mode)

        # Hint the compiler that we'd like the firmware to set the kernel arguments
        # to user SGPRs so that the kernel does not need to s_load its arguments
        # from memory.
        amd.set_all_fn_arg_inreg(fns[0])

        if options.extern_libs:
            paths = [path for (name, path) in options.extern_libs if amd.need_extern_lib(llvm_mod, name)]
            llvm.link_extern_libs(llvm_mod, paths)

        llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3, options.arch, '', [], options.enable_fp_fusion)

        # Get some metadata
        metadata["shared"] = src.get_int_attr("triton_gpu.shared")

        amd.cleanup_bitcode_metadata(llvm_mod)
        return str(llvm_mod)

    @staticmethod
    def make_amdgcn_gfx928(src, metadata, options):
        # Find kernel names (there should only be one)
        # We get the name at the last possible step to accomodate `triton.compile`
        # on user-provided LLVM
        names = re.findall(r"define amdgpu_kernel void @([a-zA-Z_][a-zA-Z0-9_]*)", src)
        assert len(names) == 1
        metadata["name"] = names[0]
        if os.environ.get("DEBUG_DUMP", "0") == "1":
            with open(f"./{metadata['name']}_{metadata['hash'][0:8:1]}.ll.ir", "w") as f:
                f.write(str(src))

        def llvmIRRewriter(src):
            src = src.replace(f"or disjoint", f"or")
            src = src.replace(f"zext nneg", f"zext")
            src = src.replace(f"uitofp nneg", f"uitofp")
            src = src.replace(f"trunc nuw", f"trunc")
            src = src.replace(f"trunc nsw", f"trunc")
            src = src.replace(f"llvm.amdgcn.exp2.f32", f"llvm.exp2.f32")
            src = src.replace(f"llvm.ldexp.f32.i32", f"llvm.amdgcn.ldexp.f32.i32")
            src = src.replace(f"llvm.ldexp.f64.i32", f"llvm.amdgcn.ldexp.f64.i32")
            src = src.replace(f"llvm.amdgcn.readfirstlane.i32", f"llvm.amdgcn.readfirstlane")

            # mfma -> mmac
            src = src.replace(f"@llvm.amdgcn.mfma", f"@llvm.amdgcn.mmac")
            # MxNxKf16 -> MxNxK.f16
            src = re.sub(r'(@llvm\.amdgcn\.mmac\.(f32|i32)\.(\d+x\d+x\d+))([a-z]*\d+)', r'\1.\4', src)
            # erase i32 0, i32 0, i32 0
            src = re.sub(r'(@llvm\.amdgcn\.mmac[^\(]+\([^\)]+),\s*i32\s+0,\s*i32\s+0,\s*i32\s+0(\))', r'\1\2', src)
            # modify declare erase i32 immarg, i32 immarg, i32 immarg
            src = re.sub(r',\s*i32 immarg,\s*i32 immarg,\s*i32 immarg', r'', src)

            # rename f32 mmac declare
            src = src.replace(f"llvm.amdgcn.mmac.f32.16x16x4.f32", f"llvm.amdgcn.mmac.16x16x4.f32")
            # mmac.16x16x4.f32 add i32 immarg
            src = re.sub(r'(declare <4 x float> @llvm\.amdgcn\.mmac\.16x16x4\.f32\(float, float, <4 x float>)',
                         r'\1, i32 immarg', src)

            # mmac.16x16x4.f32 add i32 i32 0
            src = re.sub(r'(@llvm\.amdgcn\.mmac\.16x16x4\.f32\(float [^,]+, float [^,]+, <4 x float> [^)]+)',
                         r'\1, i32 0', src)

            # llvm.amdgcn.mmac.f32.16x16x16.bf16.1k -> llvm.amdgcn.mmac.f32.16x16x16.bf16
            src = src.replace(f"llvm.amdgcn.mmac.f32.16x16x16.bf16.1k", f"llvm.amdgcn.mmac.f32.16x16x16.bf16")

            def rewriteRange(input_ll_ir):
                lines = input_ll_ir.splitlines()
                output_lines = []
                metadata_index = 666666  # magic counter
                metadata_definitions = {}  # store meta data
                for line in lines:
                    match = re.search(r'range\((\w+) (\d+), (\d+)\)', line)
                    if match:
                        data_type, start, end = match.groups()
                        line = re.sub(r'range\(\w+ \d+, \d+\) ', '', line)
                        metadata_key = metadata_index
                        line = re.sub(r'!dbg', f'!range !{metadata_key}, !dbg', line)
                        metadata_definitions[metadata_key] = (data_type, start, end)
                        metadata_index += 1

                    output_lines.append(line)

                output_ll_ir = '\n'.join(output_lines)
                for key, (data_type, start, end) in metadata_definitions.items():
                    output_ll_ir += f'\n!{key} = !{{{data_type} {start}, {data_type} {end}}}'

                return output_ll_ir

            src = rewriteRange(src)
            return src

        # rewrite llvm ir
        src = llvmIRRewriter(src)
        if os.environ.get("DEBUG_DUMP", "0") == "1":
            with open(f"./{metadata['name']}_{metadata['hash'][0:8:1]}.rewrite.ll.ir", "w") as f:
                f.write(str(src))
        if not os.path.exists("/tmp/triton3"):
            os.makedirs("/tmp/triton3")

        # llvm -> hsaco
        rocm_base_path = "/opt/rocm"
        bin_path = f"{rocm_base_path}/llvm/bin"
        bitcode_path = f"{rocm_base_path}/amdgcn/bitcode"
        llc_path = f"{bin_path}/llc"
        llvm_as_path = f"{bin_path}/llvm-as"
        llvm_link_path = f"{bin_path}/llvm-link"
        llirIn = tempfile.NamedTemporaryFile(prefix="_", suffix=".ll", dir="/tmp/triton3").name
        llir_BC = tempfile.NamedTemporaryFile(prefix="_", suffix=".bc", dir="/tmp/triton3").name
        llir_linked_BC = tempfile.NamedTemporaryFile(prefix="_", suffix=".linked.bc", dir="/tmp/triton3").name
        amdgcnOut = tempfile.NamedTemporaryFile(prefix="_", suffix=".amdgcn", dir="/tmp/triton3").name
        with open(llirIn, "w") as fd_in:
            fd_in.write(src)

        subprocess.check_call([
            llvm_as_path,
            '-o',
            llir_BC,
            llirIn,
        ])

        subprocess.check_call([
            llvm_link_path,
            '--only-needed',
            '-o',
            llir_linked_BC,
            llir_BC,
            f"{bitcode_path}/hip.bc",
            f"{bitcode_path}/ocml.bc",
            f"{bitcode_path}/ockl.bc",
            f"{bitcode_path}/oclc_daz_opt_off.bc",
            f"{bitcode_path}/oclc_unsafe_math_off.bc",
            f"{bitcode_path}/oclc_finite_only_off.bc",
            f"{bitcode_path}/oclc_correctly_rounded_sqrt_on.bc",
            f"{bitcode_path}/oclc_wavefrontsize64_on.bc",
            f"{bitcode_path}/oclc_isa_version_928.bc",
            f"{bitcode_path}/oclc_abi_version_500.bc",
        ])

        subprocess.check_call([
            llc_path,
            '-filetype=asm',
            '-march=amdgcn',
            '-mcpu=gfx928',
            '-o',
            amdgcnOut,
            llir_linked_BC,
        ])
        with open(amdgcnOut, "r") as fd_out:
            amdgcn = fd_out.read()

        if os.environ.get("DEBUG_DUMP", "0") == "1":
            with open(f"./{metadata['name']}_{metadata['hash'][0:8:1]}.amdgcn", "w") as f:
                f.write(str(amdgcn))
        return amdgcn

    @staticmethod
    def make_hsaco_gfx928(src, metadata, options):
        base = f"/opt/rocm"
        clang_path = f"{base}/llvm/bin/clang"
        lld_path = f"{base}/llvm/bin/lld"
        bundler_path = f"{base}/llvm/bin/clang-offload-bundler"
        amdgcnIn = tempfile.NamedTemporaryFile(prefix="input_", suffix=".amdgcn", dir="/tmp/triton3").name
        compileOut = tempfile.NamedTemporaryFile(prefix="compiled_", suffix=".hsaco", dir="/tmp/triton3").name
        lldOut = tempfile.NamedTemporaryFile(prefix="lld_", suffix=".hsaco", dir="/tmp/triton3").name
        bundleOut = tempfile.NamedTemporaryFile(prefix="bundle_", suffix=".hsaco", dir="/tmp/triton3").name
        with open(amdgcnIn, "w") as fd_in:
            fd_in.write(src)
        subprocess.check_call([
            clang_path,
            '-O0',
            '-x',
            'assembler',
            '--target=amdgcn-amd-amdhsa',
            '-mcpu=gfx928',
            '-c',
            '-o',
            compileOut,
            amdgcnIn,
        ])
        subprocess.check_call([
            lld_path,
            '-flavor',
            'gnu',
            '-m',
            'elf64_amdgpu',
            '--no-undefined',
            '-shared',
            '-plugin-opt=-amdgpu-internalize-symbols',
            '-plugin-opt=mcpu=gfx928',
            '-plugin-opt=O3',
            '--lto-CGO3',
            '-plugin-opt=-amdgpu-early-inline-all=true',
            '-plugin-opt=-amdgpu-function-calls=false',
            '--whole-archive',
            '-o',
            lldOut,
            compileOut,
            '--no-whole-archive',
        ])
        subprocess.check_call([
            bundler_path,
            '-type=o',
            '-bundle-align=4096',
            '-targets=host-x86_64-unknown-linux,hipv4-amdgcn-amd-amdhsa--gfx928',
            '-input=/dev/null',
            f'-input={lldOut}',
            f'-output={bundleOut}',
        ])

        with open(bundleOut, "rb") as fd_out:
            ret = fd_out.read()
        return ret

    def add_stages(self, stages, options):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)
        stages["amdgcn"] = lambda src, metadata: self.make_amdgcn_gfx928(src, metadata, options)
        stages["hsaco"] = lambda src, metadata: self.make_hsaco_gfx928(src, metadata, options)

    @functools.lru_cache()
    def hash(self):
        version = subprocess.check_output([HIPBackend.path_to_rocm_lld(), "--version"], encoding='utf-8')
        return f'{version}-{self.target}'
