// RUN: python3.13 tenstorrent/tools/tt-opt %s -t tt-metalium | filecheck %s

builtin.module {
  builtin.module attributes {kernel_type = "host"} {
    func.func @main() -> i32 {
      %0 = arith.constant 1 : i32
      %a = memref.alloc() : memref<i32>
      memref.store %0, %a[] : memref<i32>
      %1 = arith.constant true
      scf.if %1 {
        %2 = arith.constant 5 : i32
        memref.store %2, %a[] : memref<i32>
      } else {
        %3 = arith.constant false
        scf.if %3 {
          %4 = arith.constant 6 : i32
          memref.store %4, %a[] : memref<i32>
        }
      }
      %5 = arith.constant 0 : i32
      func.return %5 : i32
    }
  }
}

// CHECK:      std::int32_t main() {
// CHECK-NEXT:     std::int32_t a = 1;
// CHECK-NEXT:     if (true) {
// CHECK-NEXT:         a = 5;
// CHECK-NEXT:     } else {
// CHECK-NEXT:         if (false) {
// CHECK-NEXT:             a = 6;
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     return 0;
// CHECK-NEXT: }
