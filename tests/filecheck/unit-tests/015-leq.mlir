// RUN: python3.13 tenstorrent/tools/tt-opt %s -t tt-metalium | filecheck %s

builtin.module {
  builtin.module attributes {kernel_type = "host"} {
    func.func @main() -> i32 {
      %0 = arith.constant 5 : i32
      %a = memref.alloc() : memref<i32>
      memref.store %0, %a[] : memref<i32>
      %1 = arith.constant 18 : i32
      %b = memref.alloc() : memref<i32>
      memref.store %1, %b[] : memref<i32>
      %2 = memref.load %a[] : memref<i32>
      %3 = memref.load %b[] : memref<i32>
      %4 = arith.cmpi ule, %2, %3 : i32
      %c = memref.alloc() : memref<i1>
      memref.store %4, %c[] : memref<i1>
      %5 = arith.constant 0 : i32
      func.return %5 : i32
    }
  }
}

// CHECK:      std::int32_t main() {
// CHECK-NEXT:     std::int32_t a = 5;
// CHECK-NEXT:     std::int32_t b = 18;
// CHECK-NEXT:     bool c = a <= b;
// CHECK-NEXT:     return 0;
// CHECK-NEXT: }