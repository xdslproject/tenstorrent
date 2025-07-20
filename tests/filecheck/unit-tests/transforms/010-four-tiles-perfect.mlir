// RUN: python3.13 tenstorrent/tools/tt-opt %s -p linalg-to-tt | filecheck %s

builtin.module {
  func.func @"example_matmul"() {
    %0 = "memref.alloca"() <{"operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<64x64xf32>
    %1 = "memref.alloca"() <{"operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<64x64xf32>
    %2 = "memref.alloca"() <{"operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<64x64xf32>
    %3 = "memref.alloca"() <{"operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<64x64xf32>
    linalg.matmul ins(%0, %1 : memref<64x64xf32>, memref<64x64xf32>) outs(%3 : memref<64x64xf32>) -> ()
    "memref.copy"(%3, %2) : (memref<64x64xf32>, memref<64x64xf32>) -> ()
    func.return
  }
  func.func @main() {
    "memref.alloca_scope"() ({
      func.call @example_matmul() : () -> ()
      "memref.alloca_scope.return"() : () -> ()
    }) : () -> ()
    func.return
  }
}

// CHECK:      %2 = arith.constant 4 : i32
// CHECK-NEXT: scf.for %3 = %0 to %2 step %1  : i32 {
// CHECK-NEXT:   %4 = arith.constant 1 : i32
// CHECK-NEXT:   %5 = arith.constant 0 : i8
// CHECK-NEXT:   %6 = arith.constant 0 : i32
// CHECK-NEXT:   %7 = "dm.get_noc_addr_from_bank_id"(%bank_id, %mem_addr, %5) <{dram = true}> : (i32, i32, i8) -> i64
// CHECK-NEXT:   %8 = "cb.get_write_pointer"(%6) : (i32) -> i32
// CHECK-NEXT:   "cb.reserve_back"(%6, %4) : (i32, i32) -> ()
// CHECK-NEXT:   "dm.noc_async_read"(%7, %8, %size_bytes) : (i64, i32, i32) -> ()
// CHECK-NEXT:   "dm.noc_async_read_barrier"() : () -> ()
// CHECK-NEXT:   "cb.push_back"(%6, %4) : (i32, i32) -> ()
// CHECK-NEXT:   %9 = arith.constant 1 : i32
// CHECK-NEXT:   %10 = arith.constant 0 : i8
// CHECK-NEXT:   %11 = arith.constant 1 : i32
// CHECK-NEXT:   %12 = "dm.get_noc_addr_from_bank_id"(%bank_id_1, %mem_addr_1, %10) <{dram = true}> : (i32, i32, i8) -> i64
// CHECK-NEXT:   %13 = "cb.get_write_pointer"(%11) : (i32) -> i32
// CHECK-NEXT:   "cb.reserve_back"(%11, %9) : (i32, i32) -> ()
// CHECK-NEXT:   "dm.noc_async_read"(%12, %13, %size_bytes_1) : (i64, i32, i32) -> ()
// CHECK-NEXT:   "dm.noc_async_read_barrier"() : () -> ()
// CHECK-NEXT:   "cb.push_back"(%11, %9) : (i32, i32) -> ()
// CHECK-NEXT: }

// CHECK:      %9 = arith.constant 4 : i32
// CHECK-NEXT: scf.for %10 = %7 to %9 step %8  : i32 {
// CHECK-NEXT:   "cb.wait_front"
// CHECK-NEXT:   "cb.wait_front"
// CHECK-NEXT:   "comp.tile_regs_acquire"() : () -> ()
// CHECK-NEXT:   "comp.matmul_tiles"
// CHECK-NEXT:   "comp.tile_regs_commit"() : () -> ()
// CHECK-NEXT:   "comp.tile_regs_wait"() : () -> ()
// CHECK-NEXT:   "comp.pack_tile"
// CHECK-NEXT:   "comp.tile_regs_release"() : () -> ()
// CHECK-NEXT:   "cb.pop_front"
// CHECK-NEXT:   "cb.pop_front"
// CHECK-NEXT:   "cb.push_back"

// CHECK:      %5 = arith.constant 4 : i32
// CHECK-NEXT: scf.for %6 = %4 to %5 step %1  : i32 {
// CHECK-NEXT:   "cb.wait_front"
// CHECK-NEXT:   "dm.noc_async_write"
// CHECK-NEXT:   "dm.noc_async_write_barrier"
// CHECK-NEXT:   "cb.pop_front"
// CHECK-NEXT: }