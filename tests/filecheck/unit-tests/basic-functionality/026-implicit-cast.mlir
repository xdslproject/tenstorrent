// RUN: python3.13 tenstorrent/tools/tt-opt %s -t tt-metalium | filecheck %s

builtin.module {
  builtin.module attributes {kernel_type = "host"} {
    func.func @main() -> i32 {
      %0 = arith.constant 7 : i32
      %1 = arith.sitofp %0 : i32 to f32
      %a = memref.alloc() : memref<f32>
      memref.store %1, %a[] : memref<f32>
      %2 = arith.constant 1.340000e+01 : f32
      %3 = arith.constant 3 : i32
      %4 = arith.sitofp %3 : i32 to f32
      %5 = arith.mulf %2, %4 : f32
      memref.store %5, %a[] : memref<f32>
      %6 = arith.constant 0 : i32
      func.return %6 : i32
    }
  }
}

// CHECK:      std::int32_t main() {
// CHECK-NEXT:     float a = static_cast<float>(7);
// CHECK-NEXT:     a = 13.399999618530273 * static_cast<float>(3);
// CHECK-NEXT:     return 0;
// CHECK-NEXT: }