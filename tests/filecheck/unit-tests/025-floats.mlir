// RUN: python3.13 tenstorrent/tools/tt-opt %s -t tt-metalium | filecheck %s

builtin.module {
  builtin.module attributes {kernel_type = "host"} {
    func.func @main() -> i32 {
      %0 = arith.constant 3.700000e+00 : f32
      %a = memref.alloc() : memref<f32>
      memref.store %0, %a[] : memref<f32>
      %1 = memref.load %a[] : memref<f32>
      %2 = arith.constant 1.020000e+01 : f32
      %3 = arith.subf %1, %2 : f32
      memref.store %3, %a[] : memref<f32>
      %4 = arith.constant 0 : i32
      func.return %4 : i32
    }
  }
}

// CHECK:      std::int32_t main() {
// CHECK-NEXT:     float a = 3.700000047683716;
// CHECK-NEXT:     a = a - 10.199999809265137;
// CHECK-NEXT:     return 0;
// CHECK-NEXT: }
