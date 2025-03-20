// RUN: python3.13 tenstorrent/tools/tt-opt %s -t tt-metalium | filecheck %s

builtin.module {
  builtin.module attributes {kernel_type = "host"} {
    func.func @main() -> i32 {
      %0 = arith.constant true
      %a = memref.alloc() : memref<i1>
      memref.store %0, %a[] : memref<i1>
      %1 = arith.constant 1 : i32
      %b = memref.alloc() : memref<i32>
      memref.store %1, %b[] : memref<i32>
      %2 = memref.load %a[] : memref<i1>
      scf.if %2 {
        %3 = arith.constant 2 : i32
        memref.store %3, %b[] : memref<i32>
      } else {
        %4 = memref.load %a[] : memref<i1>
        scf.if %4 {
          %5 = arith.constant 3 : i32
          memref.store %5, %b[] : memref<i32>
        } else {
          %6 = arith.constant 4 : i32
          memref.store %6, %b[] : memref<i32>
        }
      }
      %7 = arith.constant 0 : i32
      func.return %7 : i32
    }
  }
}

// CHECK:      std::int32_t main() {
// CHECK-NEXT:     bool a = true;
// CHECK-NEXT:     std::int32_t b = 1;
// CHECK-NEXT:     if (a) {
// CHECK-NEXT:         b = 2;
// CHECK-NEXT:     } else {
// CHECK-NEXT:         if (a) {
// CHECK-NEXT:             b = 3;
// CHECK-NEXT:         } else {
// CHECK-NEXT:             b = 4;
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     return 0;
// CHECK-NEXT: }