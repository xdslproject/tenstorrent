# RUN: python3.13 %s | filecheck %s
from tenstorrent.frontend import tt


@tt.host
def implicit_cast():
    # our lang is statically typed, a should be assigned 'float' due to 13.4
    a = 7
    a = 13.4 * 3


# CHECK:      builtin.module {
# CHECK-NEXT:   builtin.module attributes {kernel_type = "host"} {
# CHECK-NEXT:     func.func @main() -> i32 {
# CHECK-NEXT:       %0 = arith.constant 7 : i32
# CHECK-NEXT:       %1 = arith.sitofp %0 : i32 to f32
# CHECK-NEXT:       %a = memref.alloc() : memref<f32>
# CHECK-NEXT:       memref.store %1, %a[] : memref<f32>
# CHECK-NEXT:       %2 = arith.constant 1.340000e+01 : f32
# CHECK-NEXT:       %3 = arith.constant 3 : i32
# CHECK-NEXT:       %4 = arith.sitofp %3 : i32 to f32
# CHECK-NEXT:       %5 = arith.mulf %2, %4 : f32
# CHECK-NEXT:       memref.store %5, %a[] : memref<f32>
# CHECK-NEXT:       %6 = arith.constant 0 : i32
# CHECK-NEXT:       func.return %6 : i32
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT: }
