// RUN: python3.13 tenstorrent/tools/tt-opt %s -t tt-metalium | filecheck %s

builtin.module {
  builtin.module attributes {kernel_type = "host"} {
    func.func @main() -> i32 {
      %0 = arith.constant 1 : i32
      %1 = arith.constant 2 : i32
      %2 = arith.constant 3 : i32
      %3 = arith.muli %1, %2 : i32
      %4 = arith.addi %0, %3 : i32
      %5 = arith.constant 4 : i32
      %6 = arith.addi %4, %5 : i32
      %a = memref.alloc() : memref<i32>
      memref.store %6, %a[] : memref<i32>
      %7 = arith.constant 0 : i32
      func.return %7 : i32
    }
  }
}

// CHECK: std::int32_t main() {
// CHECK-NEXT:     std::int32_t a = 1 + 2 * 3 + 4;
// CHECK-NEXT:     return 0;
// CHECK-NEXT: }
