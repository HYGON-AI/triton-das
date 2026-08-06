; ModuleID = 'hcu-max-mem-cluster.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define amdgpu_kernel void @same_base_cluster_kernel(ptr addrspace(1) inreg nocapture readonly %0, ptr addrspace(1) inreg nocapture writeonly %1, i32 inreg %2, ptr addrspace(1) inreg nocapture readnone %3, ptr addrspace(1) inreg nocapture readnone %4) local_unnamed_addr #0 {
  %6 = tail call i32 @llvm.amdgcn.workgroup.id.x()
  %7 = shl i32 %6, 12
  %8 = tail call i32 @llvm.amdgcn.workitem.id.x()
  %9 = shl nuw nsw i32 %8, 2
  %10 = and i32 %9, 252
  %11 = or i32 %10, %7
  %12 = or i32 %11, 256
  %13 = or i32 %11, 512
  %14 = or i32 %11, 768
  %15 = or i32 %11, 1024
  %16 = or i32 %11, 1280
  %17 = or i32 %11, 1536
  %18 = or i32 %11, 1792
  %19 = or i32 %11, 2048
  %20 = or i32 %11, 2304
  %21 = or i32 %11, 2560
  %22 = or i32 %11, 2816
  %23 = or i32 %11, 3072
  %24 = or i32 %11, 3328
  %25 = or i32 %11, 3584
  %26 = or i32 %11, 3840
  %27 = icmp slt i32 %11, %2
  %28 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p1(ptr addrspace(1) %0, i16 0, i64 -8, i32 159744)
  %29 = shl i32 %11, 2
  %30 = select i1 %27, i32 %29, i32 -8
  %31 = tail call <4 x float> @llvm.amdgcn.raw.ptr.buffer.load.v4f32(ptr addrspace(8) %28, i32 %30, i32 0, i32 0)
  %32 = extractelement <4 x float> %31, i64 0
  %33 = extractelement <4 x float> %31, i64 1
  %34 = extractelement <4 x float> %31, i64 2
  %35 = extractelement <4 x float> %31, i64 3
  %36 = icmp slt i32 %12, %2
  %37 = or i32 %29, 1024
  %38 = select i1 %36, i32 %37, i32 -8
  %39 = tail call <4 x float> @llvm.amdgcn.raw.ptr.buffer.load.v4f32(ptr addrspace(8) %28, i32 %38, i32 0, i32 0)
  %40 = extractelement <4 x float> %39, i64 0
  %41 = extractelement <4 x float> %39, i64 1
  %42 = extractelement <4 x float> %39, i64 2
  %43 = extractelement <4 x float> %39, i64 3
  %44 = icmp slt i32 %13, %2
  %45 = or i32 %29, 2048
  %46 = select i1 %44, i32 %45, i32 -8
  %47 = tail call <4 x float> @llvm.amdgcn.raw.ptr.buffer.load.v4f32(ptr addrspace(8) %28, i32 %46, i32 0, i32 0)
  %48 = extractelement <4 x float> %47, i64 0
  %49 = extractelement <4 x float> %47, i64 1
  %50 = extractelement <4 x float> %47, i64 2
  %51 = extractelement <4 x float> %47, i64 3
  %52 = icmp slt i32 %14, %2
  %53 = or i32 %29, 3072
  %54 = select i1 %52, i32 %53, i32 -8
  %55 = tail call <4 x float> @llvm.amdgcn.raw.ptr.buffer.load.v4f32(ptr addrspace(8) %28, i32 %54, i32 0, i32 0)
  %56 = extractelement <4 x float> %55, i64 0
  %57 = extractelement <4 x float> %55, i64 1
  %58 = extractelement <4 x float> %55, i64 2
  %59 = extractelement <4 x float> %55, i64 3
  %60 = icmp slt i32 %15, %2
  %61 = or i32 %29, 4096
  %62 = select i1 %60, i32 %61, i32 -8
  %63 = tail call <4 x float> @llvm.amdgcn.raw.ptr.buffer.load.v4f32(ptr addrspace(8) %28, i32 %62, i32 0, i32 0)
  %64 = extractelement <4 x float> %63, i64 0
  %65 = extractelement <4 x float> %63, i64 1
  %66 = extractelement <4 x float> %63, i64 2
  %67 = extractelement <4 x float> %63, i64 3
  %68 = icmp slt i32 %16, %2
  %69 = or i32 %29, 5120
  %70 = select i1 %68, i32 %69, i32 -8
  %71 = tail call <4 x float> @llvm.amdgcn.raw.ptr.buffer.load.v4f32(ptr addrspace(8) %28, i32 %70, i32 0, i32 0)
  %72 = extractelement <4 x float> %71, i64 0
  %73 = extractelement <4 x float> %71, i64 1
  %74 = extractelement <4 x float> %71, i64 2
  %75 = extractelement <4 x float> %71, i64 3
  %76 = icmp slt i32 %17, %2
  %77 = or i32 %29, 6144
  %78 = select i1 %76, i32 %77, i32 -8
  %79 = tail call <4 x float> @llvm.amdgcn.raw.ptr.buffer.load.v4f32(ptr addrspace(8) %28, i32 %78, i32 0, i32 0)
  %80 = extractelement <4 x float> %79, i64 0
  %81 = extractelement <4 x float> %79, i64 1
  %82 = extractelement <4 x float> %79, i64 2
  %83 = extractelement <4 x float> %79, i64 3
  %84 = icmp slt i32 %18, %2
  %85 = or i32 %29, 7168
  %86 = select i1 %84, i32 %85, i32 -8
  %87 = tail call <4 x float> @llvm.amdgcn.raw.ptr.buffer.load.v4f32(ptr addrspace(8) %28, i32 %86, i32 0, i32 0)
  %88 = extractelement <4 x float> %87, i64 0
  %89 = extractelement <4 x float> %87, i64 1
  %90 = extractelement <4 x float> %87, i64 2
  %91 = extractelement <4 x float> %87, i64 3
  %92 = icmp slt i32 %19, %2
  %93 = or i32 %29, 8192
  %94 = select i1 %92, i32 %93, i32 -8
  %95 = tail call <4 x float> @llvm.amdgcn.raw.ptr.buffer.load.v4f32(ptr addrspace(8) %28, i32 %94, i32 0, i32 0)
  %96 = extractelement <4 x float> %95, i64 0
  %97 = extractelement <4 x float> %95, i64 1
  %98 = extractelement <4 x float> %95, i64 2
  %99 = extractelement <4 x float> %95, i64 3
  %100 = icmp slt i32 %20, %2
  %101 = or i32 %29, 9216
  %102 = select i1 %100, i32 %101, i32 -8
  %103 = tail call <4 x float> @llvm.amdgcn.raw.ptr.buffer.load.v4f32(ptr addrspace(8) %28, i32 %102, i32 0, i32 0)
  %104 = extractelement <4 x float> %103, i64 0
  %105 = extractelement <4 x float> %103, i64 1
  %106 = extractelement <4 x float> %103, i64 2
  %107 = extractelement <4 x float> %103, i64 3
  %108 = icmp slt i32 %21, %2
  %109 = or i32 %29, 10240
  %110 = select i1 %108, i32 %109, i32 -8
  %111 = tail call <4 x float> @llvm.amdgcn.raw.ptr.buffer.load.v4f32(ptr addrspace(8) %28, i32 %110, i32 0, i32 0)
  %112 = extractelement <4 x float> %111, i64 0
  %113 = extractelement <4 x float> %111, i64 1
  %114 = extractelement <4 x float> %111, i64 2
  %115 = extractelement <4 x float> %111, i64 3
  %116 = icmp slt i32 %22, %2
  %117 = or i32 %29, 11264
  %118 = select i1 %116, i32 %117, i32 -8
  %119 = tail call <4 x float> @llvm.amdgcn.raw.ptr.buffer.load.v4f32(ptr addrspace(8) %28, i32 %118, i32 0, i32 0)
  %120 = extractelement <4 x float> %119, i64 0
  %121 = extractelement <4 x float> %119, i64 1
  %122 = extractelement <4 x float> %119, i64 2
  %123 = extractelement <4 x float> %119, i64 3
  %124 = icmp slt i32 %23, %2
  %125 = or i32 %29, 12288
  %126 = select i1 %124, i32 %125, i32 -8
  %127 = tail call <4 x float> @llvm.amdgcn.raw.ptr.buffer.load.v4f32(ptr addrspace(8) %28, i32 %126, i32 0, i32 0)
  %128 = extractelement <4 x float> %127, i64 0
  %129 = extractelement <4 x float> %127, i64 1
  %130 = extractelement <4 x float> %127, i64 2
  %131 = extractelement <4 x float> %127, i64 3
  %132 = icmp slt i32 %24, %2
  %133 = or i32 %29, 13312
  %134 = select i1 %132, i32 %133, i32 -8
  %135 = tail call <4 x float> @llvm.amdgcn.raw.ptr.buffer.load.v4f32(ptr addrspace(8) %28, i32 %134, i32 0, i32 0)
  %136 = extractelement <4 x float> %135, i64 0
  %137 = extractelement <4 x float> %135, i64 1
  %138 = extractelement <4 x float> %135, i64 2
  %139 = extractelement <4 x float> %135, i64 3
  %140 = icmp slt i32 %25, %2
  %141 = or i32 %29, 14336
  %142 = select i1 %140, i32 %141, i32 -8
  %143 = tail call <4 x float> @llvm.amdgcn.raw.ptr.buffer.load.v4f32(ptr addrspace(8) %28, i32 %142, i32 0, i32 0)
  %144 = extractelement <4 x float> %143, i64 0
  %145 = extractelement <4 x float> %143, i64 1
  %146 = extractelement <4 x float> %143, i64 2
  %147 = extractelement <4 x float> %143, i64 3
  %148 = icmp slt i32 %26, %2
  %149 = or i32 %7, %9
  %150 = shl i32 %149, 2
  %151 = or i32 %150, 15360
  %152 = select i1 %148, i32 %151, i32 -8
  %153 = tail call <4 x float> @llvm.amdgcn.raw.ptr.buffer.load.v4f32(ptr addrspace(8) %28, i32 %152, i32 0, i32 0)
  %154 = extractelement <4 x float> %153, i64 0
  %155 = extractelement <4 x float> %153, i64 1
  %156 = extractelement <4 x float> %153, i64 2
  %157 = extractelement <4 x float> %153, i64 3
  %158 = fadd float %32, %40
  %159 = fadd float %33, %41
  %160 = fadd float %34, %42
  %161 = fadd float %35, %43
  %162 = fadd float %158, %48
  %163 = fadd float %159, %49
  %164 = fadd float %160, %50
  %165 = fadd float %161, %51
  %166 = fadd float %162, %56
  %167 = fadd float %163, %57
  %168 = fadd float %164, %58
  %169 = fadd float %165, %59
  %170 = fadd float %166, %64
  %171 = fadd float %167, %65
  %172 = fadd float %168, %66
  %173 = fadd float %169, %67
  %174 = fadd float %170, %72
  %175 = fadd float %171, %73
  %176 = fadd float %172, %74
  %177 = fadd float %173, %75
  %178 = fadd float %174, %80
  %179 = fadd float %175, %81
  %180 = fadd float %176, %82
  %181 = fadd float %177, %83
  %182 = fadd float %178, %88
  %183 = fadd float %179, %89
  %184 = fadd float %180, %90
  %185 = fadd float %181, %91
  %186 = fadd float %182, %96
  %187 = fadd float %183, %97
  %188 = fadd float %184, %98
  %189 = fadd float %185, %99
  %190 = fadd float %186, %104
  %191 = fadd float %187, %105
  %192 = fadd float %188, %106
  %193 = fadd float %189, %107
  %194 = fadd float %190, %112
  %195 = fadd float %191, %113
  %196 = fadd float %192, %114
  %197 = fadd float %193, %115
  %198 = fadd float %194, %120
  %199 = fadd float %195, %121
  %200 = fadd float %196, %122
  %201 = fadd float %197, %123
  %202 = fadd float %198, %128
  %203 = fadd float %199, %129
  %204 = fadd float %200, %130
  %205 = fadd float %201, %131
  %206 = fadd float %202, %136
  %207 = fadd float %203, %137
  %208 = fadd float %204, %138
  %209 = fadd float %205, %139
  %210 = fadd float %206, %144
  %211 = fadd float %207, %145
  %212 = fadd float %208, %146
  %213 = fadd float %209, %147
  %214 = fadd float %210, %154
  %215 = fadd float %211, %155
  %216 = fadd float %212, %156
  %217 = fadd float %213, %157
  %218 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p1(ptr addrspace(1) %1, i16 0, i64 -8, i32 159744)
  %219 = insertelement <4 x float> poison, float %214, i64 0
  %220 = insertelement <4 x float> %219, float %215, i64 1
  %221 = insertelement <4 x float> %220, float %216, i64 2
  %222 = insertelement <4 x float> %221, float %217, i64 3
  %223 = shl i32 %6, 10
  %224 = shl nuw nsw i32 %10, 2
  %225 = or i32 %224, %223
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.v4f32(<4 x float> %222, ptr addrspace(8) %218, i32 %225, i32 0, i32 0)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.amdgcn.workgroup.id.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare <4 x float> @llvm.amdgcn.raw.ptr.buffer.load.v4f32(ptr addrspace(8) nocapture readonly, i32, i32, i32 immarg) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: write)
declare void @llvm.amdgcn.raw.ptr.buffer.store.v4f32(<4 x float>, ptr addrspace(8) nocapture writeonly, i32, i32, i32 immarg) #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p1(ptr addrspace(1) readnone, i16, i64, i32) #1

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "amdgpu-agpr-alloc"="0" "amdgpu-flat-work-group-size"="1,64" "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-waves-per-eu"="1" "denormal-fp-math-f32"="ieee" "uniform-work-group-size"="true" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"amdhsa_code_object_version", i32 500}
