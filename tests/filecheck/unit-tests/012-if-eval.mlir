// RUN: python3.13 tenstorrent/tools/tt-opt %s -t tt-metalium | filecheck %s

builtin.module {
  builtin.module attributes {kernel_type = "host"} {
    func.func @main() -> i32 {
      %0 = arith.constant 5 : i32
      %a = memref.alloc() : memref<i32>
      memref.store %0, %a[] : memref<i32>
      %1 = arith.constant 3 : i32
      %b = memref.alloc() : memref<i32>
      memref.store %1, %b[] : memref<i32>
      %c = memref.alloc() : memref<i32>
      %2 = memref.load %a[] : memref<i32>
      %3 = memref.load %b[] : memref<i32>
      %4 = arith.cmpi ult, %2, %3 : i32
      scf.if %4 {
        %5 = arith.constant 5 : i32
        memref.store %5, %c[] : memref<i32>
      }
      %6 = arith.constant 0 : i32
      func.return %6 : i32
    }
  }
}

// CHECK:      std::int32_t main() {
// CHECK-NEXT:     std::int32_t a = 5;
// CHECK-NEXT:     std::int32_t b = 3;
// CHECK-NEXT:     std::int32_t c;
// CHECK-NEXT:     if (a < b) {
// CHECK-NEXT:         c = 5;
// CHECK-NEXT:     }
// CHECK-NEXT:     return 0;
// CHECK-NEXT: }