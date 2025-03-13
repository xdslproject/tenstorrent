// RUN: python3.13 tenstorrent/tools/tt-opt %s -t tt-metalium | filecheck %s

builtin.module {
  builtin.module attributes {kernel_type = "host"} {
    func.func @main() -> i32 {
      %0 = arith.constant true
      %1 = arith.constant true
      %2 = arith.xori %0, %1 : i1
      %a = memref.alloc() : memref<i1>
      memref.store %2, %a[] : memref<i1>
      %3 = arith.constant 0 : i32
      func.return %3 : i32
    }
  }
}

// CHECK-NEXT: std::int32_t main() {
// CHECK-NEXT:     bool a = !true;
// CHECK-NEXT:     return 0;
// CHECK-NEXT: }
