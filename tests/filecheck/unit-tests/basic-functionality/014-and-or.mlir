// RUN: python3.13 tenstorrent/tools/tt-opt %s -t tt-metalium | filecheck %s

builtin.module {
  builtin.module attributes {kernel_type = "host"} {
    func.func @main() -> i32 {
      %0 = arith.constant true
      %a = memref.alloc() : memref<i1>
      memref.store %0, %a[] : memref<i1>
      %1 = arith.constant false
      %b = memref.alloc() : memref<i1>
      memref.store %1, %b[] : memref<i1>
      %2 = memref.load %a[] : memref<i1>
      %3 = memref.load %b[] : memref<i1>
      %4 = arith.andi %2, %3 : i1
      %c = memref.alloc() : memref<i1>
      memref.store %4, %c[] : memref<i1>
      %5 = memref.load %a[] : memref<i1>
      %6 = memref.load %b[] : memref<i1>
      %7 = arith.ori %5, %6 : i1
      %d = memref.alloc() : memref<i1>
      memref.store %7, %d[] : memref<i1>
      %8 = arith.constant 0 : i32
      func.return %8 : i32
    }
  }
}

// CHECK:      std::int32_t main() {
// CHECK-NEXT:     bool a = true;
// CHECK-NEXT:     bool b = false;
// CHECK-NEXT:     bool c = a && b;
// CHECK-NEXT:     bool d = a || b;
// CHECK-NEXT:     return 0;
// CHECK-NEXT: }