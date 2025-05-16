# RUN: python3.13 %s | filecheck %s
from tenstorrent.frontend import tt


@tt.host
def division():
    a = 4.0
    a = a / 2.0


# CHECK:      builtin.module {
# CHECK-NEXT:   builtin.module attributes {kernel_type = "host"} {
# CHECK-NEXT:     func.func @main() -> i32 {
# CHECK-NEXT:       %0 = arith.constant 4.000000e+00 : f32
# CHECK-NEXT:       %a = memref.alloc() : memref<f32>
# CHECK-NEXT:       memref.store %0, %a[] : memref<f32>
# CHECK-NEXT:       %1 = memref.load %a[] : memref<f32>
# CHECK-NEXT:       %2 = arith.constant 2.000000e+00 : f32
# CHECK-NEXT:       %3 = arith.divf %1, %2 : f32
# CHECK-NEXT:       memref.store %3, %a[] : memref<f32>
# CHECK-NEXT:       %4 = arith.constant 0 : i32
# CHECK-NEXT:       func.return %4 : i32
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT: }
