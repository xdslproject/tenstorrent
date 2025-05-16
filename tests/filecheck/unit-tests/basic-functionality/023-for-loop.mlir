// RUN: python3.13 tenstorrent/tools/tt-opt %s -t tt-metalium | filecheck %s

builtin.module {
  builtin.module attributes {kernel_type = "host"} {
    func.func @main() -> i32 {
      %a = memref.alloc() : memref<i32>
      %0 = arith.constant 0 : i32
      %1 = arith.constant 50 : i32
      %2 = arith.constant 1 : i32
      %i = memref.alloc() : memref<i32>
      scf.for %3 = %0 to %1 step %2  : i32 {
        memref.store %3, %i[] : memref<i32>
        %4 = arith.constant 5 : i32
        memref.store %4, %a[] : memref<i32>
      }
      %5 = arith.constant 0 : i32
      func.return %5 : i32
    }
  }
}

// CHECK:      std::int32_t main() {
// CHECK-NEXT:     std::int32_t a;
// CHECK-NEXT:     std::int32_t i;
// CHECK-NEXT:     for (i = 0; i < 50; i += 1) {
// CHECK-NEXT:         a = 5;
// CHECK-NEXT:     }
// CHECK-NEXT:     return 0;
// CHECK-NEXT: }
