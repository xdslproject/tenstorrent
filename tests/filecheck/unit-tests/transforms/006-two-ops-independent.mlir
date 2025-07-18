// RUN: python3.13 tenstorrent/tools/tt-opt %s -p linalg-to-tt | filecheck %s

builtin.module {
  func.func @"example_independent"() {
    %0 = "memref.alloca"() <{"operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<32x32xf32>
    %1 = "memref.alloca"() <{"operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<32x32xf32>
    %2 = "memref.alloca"() <{"operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<32x32xf32>
    %3 = "memref.alloca"() <{"operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<32x32xf32>
    linalg.add ins(%0, %1 : memref<32x32xf32>, memref<32x32xf32>) outs(%3 : memref<32x32xf32>) -> ()
    "memref.copy"(%3, %2) : (memref<32x32xf32>, memref<32x32xf32>) -> ()

    %4 = "memref.alloca"() <{"operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<32x32xf32>
    %5 = "memref.alloca"() <{"operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<32x32xf32>
    %6 = "memref.alloca"() <{"operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<32x32xf32>
    %7 = "memref.alloca"() <{"operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<32x32xf32>
    linalg.matmul ins(%4, %5 : memref<32x32xf32>, memref<32x32xf32>) outs(%7 : memref<32x32xf32>) -> ()
    "memref.copy"(%7, %6) : (memref<32x32xf32>, memref<32x32xf32>) -> ()
    func.return
  }
  func.func @main() {
    "memref.alloca_scope"() ({
      func.call @example_independent() : () -> ()
      "memref.alloca_scope.return"() : () -> ()
    }) : () -> ()
    func.return
  }
}



// CHECK:       func.call @host_entry(%0, %1, %3) : (memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) -> ()
// CHECK:       func.call @host_entry_1(%4, %5, %7) : (memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) -> ()
// CHECK:     func.func private @host_entry(memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) -> ()
// CHECK:     func.func private @host_entry_1(memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) -> ()
// CHECK:   builtin.module attributes {kernel_type = "host", vis = "external"} {
// CHECK:     func.func @host_entry(%0 : memref<32x32xf32>, %1 : memref<32x32xf32>, %2 : memref<32x32xf32>) {
// CHECK:   builtin.module attributes {kernel_type = "data_in"} {
// CHECK:   builtin.module attributes {kernel_type = "compute"} {
// CHECK:   builtin.module attributes {kernel_type = "data_out"} {
// CHECK:   builtin.module attributes {kernel_type = "host", vis = "external"} {
// CHECK:     func.func @host_entry_1(%0 : memref<32x32xf32>, %1 : memref<32x32xf32>, %2 : memref<32x32xf32>) {
// CHECK:   builtin.module attributes {kernel_type = "data_in"} {
// CHECK:     func.func @kernel_main(%bank_id : i32, %bank_id_1 : i32, %mem_addr : i32, %mem_addr_1 : i32, %size_bytes : i32, %size_bytes_1 : i32) {
// CHECK:   builtin.module attributes {kernel_type = "compute"} {
// CHECK:   builtin.module attributes {kernel_type = "data_out"} {
