// RUN: python3.13 tenstorrent/tools/tt-opt %s -t tt-metalium | filecheck %s

builtin.module {
  builtin.module attributes {kernel_type = "host"} {
    func.func @main() -> i32 {
      %0 = arith.constant 0 : i32
      %a = memref.alloc() : memref<i32>
      memref.store %0, %a[] : memref<i32>
      %1 = arith.constant 0 : i32
      %2 = arith.constant 50 : i32
      %3 = arith.constant 1 : i32
      %i = memref.alloc() : memref<i32>
      scf.for %4 = %1 to %2 step %3  : i32 {
        memref.store %4, %i[] : memref<i32>
        %5 = memref.load %i[] : memref<i32>
        memref.store %5, %a[] : memref<i32>
      }
      %6 = arith.constant 0 : i32
      func.return %6 : i32
    }
  }
}

// CHECK:      std::int32_t main() {
// CHECK-NEXT:     std::int32_t a = 0;
// CHECK-NEXT:     std::int32_t i;
// CHECK-NEXT:     for (i = 0; i < 50; i += 1) {
// CHECK-NEXT:         a = i;
// CHECK-NEXT:     }
// CHECK-NEXT:     return 0;
// CHECK-NEXT: }
