// RUN: python3.13 tenstorrent/tools/tt-opt %s -t tt-metalium | filecheck %s

builtin.module {
  builtin.module attributes {kernel_type = "host"} {
    func.func @main() -> i32 {
      %0 = arith.constant 4.000000e+00 : f32
      %a = memref.alloc() : memref<f32>
      memref.store %0, %a[] : memref<f32>
      %1 = memref.load %a[] : memref<f32>
      %2 = arith.constant 2.000000e+00 : f32
      %3 = arith.divf %1, %2 : f32
      memref.store %3, %a[] : memref<f32>
      %4 = arith.constant 0 : i32
      func.return %4 : i32
    }
  }
}

// CHECK:      std::int32_t main() {
// CHECK-NEXT:     float a = 4.0;
// CHECK-NEXT:     a = a / 2.0;
// CHECK-NEXT:     return 0;
// CHECK-NEXT: }
