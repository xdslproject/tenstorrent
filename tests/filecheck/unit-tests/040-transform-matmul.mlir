// RUN: python3.13 tenstorrent/tools/tt-opt %s -p rewrite-matmul-to-tt | filecheck %s

builtin.module {
  func.func @"_QMproblem_modPentry"() {
    %0 = "memref.alloca"() <{"operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<10x10xi32>
    %1 = "memref.alloca"() <{"operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<10x10xi32>
    %2 = "memref.alloca"() <{"operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<10x10xi32>
    %3 = "memref.alloca"() <{"operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<10x10xi32>
    linalg.matmul ins(%0, %1 : memref<10x10xi32>, memref<10x10xi32>) outs(%3 : memref<10x10xi32>) -> ()
    "memref.copy"(%3, %2) : (memref<10x10xi32>, memref<10x10xi32>) -> ()
    func.return
  }
  func.func @main() {
    "memref.alloca_scope"() ({
      func.call @_QMproblem_modPentry() : () -> ()
      "memref.alloca_scope.return"() : () -> ()
    }) : () -> ()
    func.return
  }
}

// CHECK:      %3 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<10x10xi32>
// CHECK-NEXT: func.call @host_entry(%0, %1, %3) : (memref<10x10xi32>, memref<10x10xi32>, memref<10x10xi32>) -> ()
// CHECK:      func.func private @host_entry(memref<10x10xi32>, memref<10x10xi32>, memref<10x10xi32>) -> ()