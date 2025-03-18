// RUN: python3.13 tenstorrent/tools/tt-opt %s -t tt-metalium | filecheck %s

builtin.module {
  builtin.module attributes {kernel_type = "host"} {
    func.func @main() -> i32 {
      %a = memref.alloc() : memref<i32>
      %0 = arith.constant true
      scf.if %0 {
        %1 = arith.constant 5 : i32
        memref.store %1, %a[] : memref<i32>
      }
      %2 = arith.constant 0 : i32
      func.return %2 : i32
    }
  }
}

// CHECK:      std::int32_t main() {
// CHECK-NEXT:     std::int32_t a;
// CHECK-NEXT:     if (true) {
// CHECK-NEXT:         a = 5
// CHECK-NEXT:     }
// CHECK-NEXT:     return 0;
// CHECK-NEXT: }
