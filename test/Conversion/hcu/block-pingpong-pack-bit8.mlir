// RUN: triton-opt %s -tritonhcu-pack-block-pingpong-bit8 | FileCheck %s

// CHECK-NOT: hcu.block_pingpong.pack_bit8_carriers
module attributes {hcu.block_pingpong.pack_bit8_carriers} {
  // CHECK-LABEL: llvm.func @pack_bit8_loop
  llvm.func @pack_bit8_loop(
      %ptr: !llvm.ptr<3>, %cond: i1,
      %init: !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8)>)
      attributes {hcu.block_pingpong.pack_bit8_carriers} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c5 = llvm.mlir.constant(5 : i32) : i32
    %c6 = llvm.mlir.constant(6 : i32) : i32
    %c7 = llvm.mlir.constant(7 : i32) : i32
    llvm.br ^loop(%init : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8)>)

  // CHECK: ^[[LOOP:bb[0-9]+]](%[[CARRIER:.*]]: !llvm.struct<(i32, i32)>)
  ^loop(%carrier: !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8)>):
    llvm.cond_br %cond, ^body, ^exit

  ^body:
    %b0 = llvm.extractvalue %carrier[0] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8)>
    %b1 = llvm.extractvalue %carrier[1] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8)>
    %b2 = llvm.extractvalue %carrier[2] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8)>
    %b3 = llvm.extractvalue %carrier[3] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8)>
    %b4 = llvm.extractvalue %carrier[4] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8)>
    %b5 = llvm.extractvalue %carrier[5] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8)>
    %b6 = llvm.extractvalue %carrier[6] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8)>
    %b7 = llvm.extractvalue %carrier[7] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8)>
    %v0 = llvm.mlir.undef : vector<8xi8>
    %v1 = llvm.insertelement %b0, %v0[%c0 : i32] : vector<8xi8>
    %v2 = llvm.insertelement %b1, %v1[%c1 : i32] : vector<8xi8>
    %v3 = llvm.insertelement %b2, %v2[%c2 : i32] : vector<8xi8>
    %v4 = llvm.insertelement %b3, %v3[%c3 : i32] : vector<8xi8>
    %v5 = llvm.insertelement %b4, %v4[%c4 : i32] : vector<8xi8>
    %v6 = llvm.insertelement %b5, %v5[%c5 : i32] : vector<8xi8>
    %v7 = llvm.insertelement %b6, %v6[%c6 : i32] : vector<8xi8>
    %v8 = llvm.insertelement %b7, %v7[%c7 : i32] : vector<8xi8>
    // CHECK-NOT: llvm.store {{.*}}vector<8xi8>
    // CHECK: llvm.store {{.*}} : vector<2xi32>, !llvm.ptr<3>
    llvm.store %v8, %ptr {alignment = 8 : i64} : vector<8xi8>, !llvm.ptr<3>

    %next0 = llvm.mlir.undef : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8)>
    %next1 = llvm.insertvalue %b0, %next0[0] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8)>
    %next2 = llvm.insertvalue %b1, %next1[1] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8)>
    %next3 = llvm.insertvalue %b2, %next2[2] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8)>
    %next4 = llvm.insertvalue %b3, %next3[3] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8)>
    %next5 = llvm.insertvalue %b4, %next4[4] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8)>
    %next6 = llvm.insertvalue %b5, %next5[5] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8)>
    %next7 = llvm.insertvalue %b6, %next6[6] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8)>
    %next8 = llvm.insertvalue %b7, %next7[7] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK: llvm.br ^[[LOOP]]({{.*}} : !llvm.struct<(i32, i32)>)
    llvm.br ^loop(%next8 : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8)>)

  ^exit:
    llvm.return
  }

  // CHECK-LABEL: llvm.func @leave_bit16_loop_unchanged
  llvm.func @leave_bit16_loop_unchanged(
      %cond: i1, %init: !llvm.struct<(i16, i16, i16, i16)>) {
    llvm.br ^loop(%init : !llvm.struct<(i16, i16, i16, i16)>)
  // CHECK: ^{{.*}}(%{{.*}}: !llvm.struct<(i16, i16, i16, i16)>)
  ^loop(%carrier: !llvm.struct<(i16, i16, i16, i16)>):
    llvm.cond_br %cond, ^body, ^exit
  ^body:
    llvm.br ^loop(%carrier : !llvm.struct<(i16, i16, i16, i16)>)
  ^exit:
    llvm.return
  }

  // A Bit8 carrier in an unmarked function is unrelated to BlockPingpong and
  // must not be rewritten merely because another function marked the module.
  // CHECK-LABEL: llvm.func @leave_untagged_bit8_loop_unchanged
  llvm.func @leave_untagged_bit8_loop_unchanged(
      %cond: i1, %init: !llvm.struct<(i8, i8, i8, i8)>) {
    llvm.br ^loop(%init : !llvm.struct<(i8, i8, i8, i8)>)
  // CHECK: ^{{.*}}(%{{.*}}: !llvm.struct<(i8, i8, i8, i8)>)
  ^loop(%carrier: !llvm.struct<(i8, i8, i8, i8)>):
    llvm.cond_br %cond, ^body, ^exit
  ^body:
    llvm.br ^loop(%carrier : !llvm.struct<(i8, i8, i8, i8)>)
  ^exit:
    llvm.return
  }

  // A dynamic trip count leaves a main pipelined loop followed by a remainder
  // loop. The main carrier initially has a whole-struct use on the edge to the
  // remainder loop, so the successor must be packed first.
  // CHECK-LABEL: llvm.func @pack_bit8_chained_loops
  llvm.func @pack_bit8_chained_loops(
      %main_cond: i1, %tail_cond: i1,
      %init: !llvm.struct<(i8, i8, i8, i8)>)
      attributes {hcu.block_pingpong.pack_bit8_carriers} {
    llvm.br ^main(%init : !llvm.struct<(i8, i8, i8, i8)>)

  // CHECK: ^[[MAIN:bb[0-9]+]](%[[MAIN_CARRIER:.*]]: !llvm.struct<(i32)>)
  ^main(%main_carrier: !llvm.struct<(i8, i8, i8, i8)>):
    llvm.cond_br %main_cond, ^main_body, ^handoff

  ^main_body:
    %m0 = llvm.extractvalue %main_carrier[0] : !llvm.struct<(i8, i8, i8, i8)>
    %m1 = llvm.extractvalue %main_carrier[1] : !llvm.struct<(i8, i8, i8, i8)>
    %m2 = llvm.extractvalue %main_carrier[2] : !llvm.struct<(i8, i8, i8, i8)>
    %m3 = llvm.extractvalue %main_carrier[3] : !llvm.struct<(i8, i8, i8, i8)>
    %mn0 = llvm.mlir.undef : !llvm.struct<(i8, i8, i8, i8)>
    %mn1 = llvm.insertvalue %m0, %mn0[0] : !llvm.struct<(i8, i8, i8, i8)>
    %mn2 = llvm.insertvalue %m1, %mn1[1] : !llvm.struct<(i8, i8, i8, i8)>
    %mn3 = llvm.insertvalue %m2, %mn2[2] : !llvm.struct<(i8, i8, i8, i8)>
    %mn4 = llvm.insertvalue %m3, %mn3[3] : !llvm.struct<(i8, i8, i8, i8)>
    // CHECK: llvm.br ^[[MAIN]]({{.*}} : !llvm.struct<(i32)>)
    llvm.br ^main(%mn4 : !llvm.struct<(i8, i8, i8, i8)>)

  ^handoff:
    llvm.br ^tail(%main_carrier : !llvm.struct<(i8, i8, i8, i8)>)

  // CHECK: ^[[TAIL:bb[0-9]+]](%[[TAIL_CARRIER:.*]]: !llvm.struct<(i32)>)
  ^tail(%tail_carrier: !llvm.struct<(i8, i8, i8, i8)>):
    llvm.cond_br %tail_cond, ^tail_body, ^exit

  ^tail_body:
    %t0 = llvm.extractvalue %tail_carrier[0] : !llvm.struct<(i8, i8, i8, i8)>
    %t1 = llvm.extractvalue %tail_carrier[1] : !llvm.struct<(i8, i8, i8, i8)>
    %t2 = llvm.extractvalue %tail_carrier[2] : !llvm.struct<(i8, i8, i8, i8)>
    %t3 = llvm.extractvalue %tail_carrier[3] : !llvm.struct<(i8, i8, i8, i8)>
    %tn0 = llvm.mlir.undef : !llvm.struct<(i8, i8, i8, i8)>
    %tn1 = llvm.insertvalue %t0, %tn0[0] : !llvm.struct<(i8, i8, i8, i8)>
    %tn2 = llvm.insertvalue %t1, %tn1[1] : !llvm.struct<(i8, i8, i8, i8)>
    %tn3 = llvm.insertvalue %t2, %tn2[2] : !llvm.struct<(i8, i8, i8, i8)>
    %tn4 = llvm.insertvalue %t3, %tn3[3] : !llvm.struct<(i8, i8, i8, i8)>
    // CHECK: llvm.br ^[[TAIL]]({{.*}} : !llvm.struct<(i32)>)
    llvm.br ^tail(%tn4 : !llvm.struct<(i8, i8, i8, i8)>)

  ^exit:
    llvm.return
  }
}
