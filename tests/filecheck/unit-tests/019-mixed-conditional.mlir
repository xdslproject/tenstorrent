// RUN: python3.13 tenstorrent/tools/tt-opt %s -t tt-metalium | filecheck %s

// TODO: sometimes breaks printing body outside loop and/or switching assigns to b
builtin.module {
  builtin.module attributes {kernel_type = "host"} {
    func.func @main() -> i32 {
      %0 = arith.constant 1 : i32
      %a = memref.alloc() : memref<i32>
      memref.store %0, %a[] : memref<i32>
      %1 = arith.constant 2 : i32
      %b = memref.alloc() : memref<i32>
      memref.store %1, %b[] : memref<i32>
      %2 = arith.constant 5 : i32
      %c = memref.alloc() : memref<i32>
      memref.store %2, %c[] : memref<i32>
      %3 = memref.load %a[] : memref<i32>
      %4 = arith.constant 1 : i32
      %5 = arith.cmpi eq, %3, %4 : i32
      %6 = memref.load %b[] : memref<i32>
      %7 = arith.constant 2 : i32
      %8 = arith.cmpi uge, %6, %7 : i32
      %9 = arith.andi %5, %8 : i1
      %10 = memref.load %c[] : memref<i32>
      %11 = memref.load %b[] : memref<i32>
      %12 = memref.load %a[] : memref<i32>
      %13 = arith.addi %11, %12 : i32
      %14 = arith.cmpi ult, %10, %13 : i32
      %15 = arith.ori %9, %14 : i1
      scf.if %15 {
        %16 = arith.constant 9 : i32
        memref.store %16, %b[] : memref<i32>
      }
      %17 = arith.constant 0 : i32
      func.return %17 : i32
    }
  }
}

// CHECK:      std::int32_t main() {
// CHECK-NEXT:     std::int32_t a = 1;
// CHECK-NEXT:     std::int32_t b = 2;
// CHECK-NEXT:     std::int32_t c = 5;
// CHECK-NEXT:     if (a == 1 && b >= 2 || c < b + a) {
// CHECK-NEXT:         b = 9;
// CHECK-NEXT:     }
// CHECK-NEXT:     return 0;
// CHECK-NEXT: }