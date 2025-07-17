// RUN: python3.13 tenstorrent/tools/tt-opt %s -p extract-driver | filecheck %s

builtin.module {
  builtin.module {
    func.func @example_matmul() {
      %0 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x32xf32>
      %1 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x32xf32>
      %2 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x32xf32>
      %3 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x32xf32>
      func.call @host_entry(%0, %1, %3) : (memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) -> ()
      "memref.copy"(%3, %2) : (memref<32x32xf32>, memref<32x32xf32>) -> ()
      func.return
    }
    func.func @main() {
      "memref.alloca_scope"() ({
        func.call @example_matmul() : () -> ()
        "memref.alloca_scope.return"() : () -> ()
      }) : () -> ()
      func.return
    }
    func.func private @host_entry(memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) -> ()
  }
  builtin.module attributes {kernel_type = "host", vis = "external"} {
    func.func @host_entry(%0 : memref<32x32xf32>, %1 : memref<32x32xf32>, %2 : memref<32x32xf32>) {
      %zero = arith.constant 0 : i32
      func.return
    }
  }
  builtin.module attributes {kernel_type = "data_in"} {
    func.func @kernel_main(%bank_id : i32, %bank_id_1 : i32, %mem_addr : i32, %mem_addr_1 : i32, %size_bytes : i32, %size_bytes_1 : i32) {
      %0 = arith.constant 1 : i32
      func.return
    }
  }
  builtin.module attributes {kernel_type = "compute"} {
    func.func @MAIN() {
      %0 = arith.constant 0 : i32
      func.return
    }
  }
  builtin.module attributes {kernel_type = "data_out"} {
    func.func @kernel_main(%bank_id : i32, %mem_addr : i32, %size_bytes : i32) {
      %0 = "dm.get_noc_addr_from_bank_id"(%bank_id, %mem_addr) <{dram = true}> : (i32, i32) -> i64
      func.return
    }
  }
}

// CHECK:      builtin.module {
// CHECK-NEXT:   func.func @example_matmul() {
// CHECK-NEXT:     %0 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x32xf32>
// CHECK-NEXT:     %1 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x32xf32>
// CHECK-NEXT:     %2 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x32xf32>
// CHECK-NEXT:     %3 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x32xf32>
// CHECK-NEXT:     func.call @host_entry(%0, %1, %3) : (memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) -> ()
// CHECK-NEXT:     "memref.copy"(%3, %2) : (memref<32x32xf32>, memref<32x32xf32>) -> ()
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @main() {
// CHECK-NEXT:     "memref.alloca_scope"() ({
// CHECK-NEXT:       func.call @example_matmul() : () -> ()
// CHECK-NEXT:       "memref.alloca_scope.return"() : () -> ()
// CHECK-NEXT:     }) : () -> ()
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func private @host_entry(memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) -> ()
// CHECK-NEXT: }
