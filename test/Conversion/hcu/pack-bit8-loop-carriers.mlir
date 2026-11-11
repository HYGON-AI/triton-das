// RUN: triton-opt %s -tritonhcu-pack-bit8-loop-carriers -verify-each | FileCheck %s
// RUN: triton-opt %s -tritonhcu-pack-bit8-loop-carriers -tritonhcu-pack-bit8-loop-carriers -verify-each | FileCheck %s

// CHECK: #[[UNROLL:.*]] = #llvm.loop_unroll<disable = true>
// CHECK: #[[$LOOP_ANNOTATION:.*]] = #llvm.loop_annotation<unroll = #[[UNROLL]]>
module {
  // CHECK-LABEL: llvm.func @pack_loop
  llvm.func @pack_loop(%init: !llvm.struct<(i8, i8, i8, i8)>, %cond: i1, %out: !llvm.ptr) {
    llvm.br ^loop(%init : !llvm.struct<(i8, i8, i8, i8)>)
  // CHECK: ^[[LOOP:bb[0-9]+]](%[[CARRIER:.*]]: !llvm.struct<(i32)>):
  ^loop(%carrier: !llvm.struct<(i8, i8, i8, i8)>):
    llvm.cond_br %cond, ^body, ^exit
  ^body:
    // CHECK: llvm.extractvalue %[[CARRIER]][0] : !llvm.struct<(i32)>
    // CHECK: llvm.bitcast {{.*}} : i32 to vector<4xi8>
    %byte = llvm.extractvalue %carrier[2] : !llvm.struct<(i8, i8, i8, i8)>
    llvm.store %byte, %out : i8, !llvm.ptr
    // CHECK: llvm.br ^[[LOOP]]({{.*}} : !llvm.struct<(i32)>) {loop_annotation = #[[$LOOP_ANNOTATION]]}
    llvm.br ^loop(%init : !llvm.struct<(i8, i8, i8, i8)>) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>}
  ^exit:
    llvm.return
  }

  // CHECK-LABEL: llvm.func @forward_to_remainder
  llvm.func @forward_to_remainder(%init: !llvm.struct<(i8, i8, i8, i8)>, %cond: i1, %out: !llvm.ptr) {
    llvm.br ^main(%init : !llvm.struct<(i8, i8, i8, i8)>)
  // CHECK: ^[[MAIN:bb[0-9]+]]({{.*}}: !llvm.struct<(i32)>):
  ^main(%carrier: !llvm.struct<(i8, i8, i8, i8)>):
    llvm.cond_br %cond, ^main_body, ^handoff
  ^main_body:
    %byte = llvm.extractvalue %carrier[0] : !llvm.struct<(i8, i8, i8, i8)>
    llvm.store %byte, %out : i8, !llvm.ptr
    // CHECK: llvm.br ^[[MAIN]]({{.*}} : !llvm.struct<(i32)>)
    llvm.br ^main(%init : !llvm.struct<(i8, i8, i8, i8)>)
  ^handoff:
    llvm.br ^tail(%carrier : !llvm.struct<(i8, i8, i8, i8)>)
  // CHECK: ^[[TAIL:bb[0-9]+]]({{.*}}: !llvm.struct<(i32)>):
  ^tail(%remainder: !llvm.struct<(i8, i8, i8, i8)>):
    llvm.cond_br %cond, ^tail_body, ^exit
  ^tail_body:
    %tail_byte = llvm.extractvalue %remainder[3] : !llvm.struct<(i8, i8, i8, i8)>
    llvm.store %tail_byte, %out : i8, !llvm.ptr
    // CHECK: llvm.br ^[[TAIL]]({{.*}} : !llvm.struct<(i32)>)
    llvm.br ^tail(%init : !llvm.struct<(i8, i8, i8, i8)>)
  ^exit:
    llvm.return
  }

  // CHECK-LABEL: llvm.func @mixed_elements
  llvm.func @mixed_elements(%init: !llvm.struct<(i8, i8, i8, i32)>, %cond: i1) {
    llvm.br ^loop(%init : !llvm.struct<(i8, i8, i8, i32)>)
  // CHECK: ^{{bb[0-9]+}}({{.*}}: !llvm.struct<(i8, i8, i8, i32)>):
  ^loop(%carrier: !llvm.struct<(i8, i8, i8, i32)>):
    llvm.cond_br %cond, ^body, ^exit
  ^body:
    llvm.br ^loop(%init : !llvm.struct<(i8, i8, i8, i32)>)
  ^exit:
    llvm.return
  }

  // CHECK-LABEL: llvm.func @partial_word
  llvm.func @partial_word(%init: !llvm.struct<(i8, i8)>, %cond: i1) {
    llvm.br ^loop(%init : !llvm.struct<(i8, i8)>)
  // CHECK: ^{{bb[0-9]+}}({{.*}}: !llvm.struct<(i8, i8)>):
  ^loop(%carrier: !llvm.struct<(i8, i8)>):
    llvm.cond_br %cond, ^body, ^exit
  ^body:
    llvm.br ^loop(%init : !llvm.struct<(i8, i8)>)
  ^exit:
    llvm.return
  }

  // CHECK-LABEL: llvm.func @whole_aggregate_use
  llvm.func @whole_aggregate_use(%init: !llvm.struct<(i8, i8, i8, i8)>, %cond: i1) -> !llvm.struct<(i8, i8, i8, i8)> {
    llvm.br ^loop(%init : !llvm.struct<(i8, i8, i8, i8)>)
  // CHECK: ^{{bb[0-9]+}}({{.*}}: !llvm.struct<(i8, i8, i8, i8)>):
  ^loop(%carrier: !llvm.struct<(i8, i8, i8, i8)>):
    llvm.cond_br %cond, ^body, ^exit
  ^body:
    llvm.br ^loop(%init : !llvm.struct<(i8, i8, i8, i8)>)
  ^exit:
    llvm.return %carrier : !llvm.struct<(i8, i8, i8, i8)>
  }
}
