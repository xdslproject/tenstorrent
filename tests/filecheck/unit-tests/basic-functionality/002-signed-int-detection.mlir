// RUN: python3.13 tenstorrent/tools/tt-opt %s -t tt-metalium | filecheck %s

builtin.module {
  builtin.module attributes {kernel_type = "host"} {
    func.func @main() -> i32 {
      %0 = arith.constant 0 : i32
      %1 = arith.constant 5 : i32
      %2 = arith.subi %0, %1 : i32
      %a = memref.alloc() : memref<i32>
      memref.store %2, %a[] : memref<i32>
      %3 = arith.constant 0 : i32
      func.return %3 : i32
    }
  }
}

// CHECK:      std::int32_t main() {
// CHECK-NEXT:     std::int32_t a = 0 - 5;
// CHECK-NEXT:     return 0;
// CHECK-NEXT: }