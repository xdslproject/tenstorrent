// RUN: python3.13 tenstorrent/tools/tt-opt %s -p linalg-to-tt | filecheck %s

builtin.module {
  func.func @"example_add"() {
    %0 = "memref.alloca"() <{"operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<32x32xi32>
    %1 = "memref.alloca"() <{"operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<32x32xi32>
    %2 = "memref.alloca"() <{"operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<32x32xi32>
    %3 = "memref.alloca"() <{"operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<32x32xi32>
    linalg.add ins(%0, %1 : memref<32x32xi32>, memref<32x32xi32>) outs(%3 : memref<32x32xi32>) -> ()
    "memref.copy"(%3, %2) : (memref<32x32xi32>, memref<32x32xi32>) -> ()
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

// CHECK:     "comp.init_sfpu"
// CHECK:     "comp.add_int32_tile_init"
// CHECK:     "comp.copy_tile"
// CHECK:     "comp.copy_tile"
// CHECK:     "comp.add_int32_tile"