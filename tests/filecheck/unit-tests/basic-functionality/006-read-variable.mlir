// RUN: python3.13 tenstorrent/tools/tt-opt %s -t tt-metalium | filecheck %s

builtin.module {
  builtin.module attributes {kernel_type = "host"} {
    func.func @main() -> i32 {
      %0 = arith.constant 5 : i32
      %a = memref.alloc() : memref<i32>
      memref.store %0, %a[] : memref<i32>
      %1 = memref.load %a[] : memref<i32>
      %2 = arith.constant 1 : i32
      %3 = arith.addi %1, %2 : i32
      %b = memref.alloc() : memref<i32>
      memref.store %3, %b[] : memref<i32>
      %4 = arith.constant 0 : i32
      func.return %4 : i32
    }
  }
}

// CHECK:      std::int32_t main() {
// CHECK-NEXT:     std::int32_t a = 5;
// CHECK-NEXT:     std::int32_t b = a + 1;
// CHECK-NEXT:     return 0;
// CHECK-NEXT: }
