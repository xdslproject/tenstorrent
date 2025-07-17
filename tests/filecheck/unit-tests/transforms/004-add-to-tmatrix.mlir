// RUN: python3.13 tenstorrent/tools/tt-opt %s -p linalg-to-tt | filecheck %s

builtin.module {
  func.func @"example_add"() {
    %0 = "memref.alloca"() <{"operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<32x32xf32>
    %1 = "memref.alloca"() <{"operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<32x32xf32>
    %2 = "memref.alloca"() <{"operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<32x32xf32>
    %3 = "memref.alloca"() <{"operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<32x32xf32>
    linalg.add ins(%0, %1 : memref<32x32xf32>, memref<32x32xf32>) outs(%3 : memref<32x32xf32>) -> ()
    "memref.copy"(%3, %2) : (memref<32x32xf32>, memref<32x32xf32>) -> ()
    func.return
  }
  func.func @main() {
    "memref.alloca_scope"() ({
      func.call @example_add() : () -> ()
      "memref.alloca_scope.return"() : () -> ()
    }) : () -> ()
    func.return
  }
}

// CHECK: "comp.binary_op_init_common"(%2, %3, %4) : (i32, i32, i32) -> ()
// CHECK: "comp.add_tiles_init"(%2, %3, %6) : (i32, i32, i1) -> ()
// CHECK: "comp.tile_regs_acquire"() : () -> ()
// CHECK: "comp.add_tiles"(%2, %3, %0, %0, %0) : (i32, i32, i32, i32, i32) -> ()
