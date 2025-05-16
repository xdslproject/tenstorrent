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
      %2 = arith.constant 0 : i32
      func.return %2 : i32
    }
  }
}

// CHECK:      std::int32_t main() {
// CHECK-NEXT:     bool a = true;
// CHECK-NEXT:     bool b = false;
// CHECK-NEXT:     return 0;
// CHECK-NEXT: }
