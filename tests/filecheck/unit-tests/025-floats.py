# RUN: python3.13 %s | filecheck %s
from tenstorrent.frontend import tt


@tt.host
def floats():
    a = 3.7
    a = a - 10.2

# CHECK:      builtin.module {
# CHECK-NEXT:   builtin.module attributes {kernel_type = "host"} {
# CHECK-NEXT:     func.func @main() -> i32 {
# CHECK-NEXT:       %0 = arith.constant 3.700000e+00 : f32
# CHECK-NEXT:       %a = memref.alloc() : memref<f32>
# CHECK-NEXT:       memref.store %0, %a[] : memref<f32>
# CHECK-NEXT:       %1 = memref.load %a[] : memref<f32>
# CHECK-NEXT:       %2 = arith.constant 1.020000e+01 : f32
# CHECK-NEXT:       %3 = arith.subf %1, %2 : f32
# CHECK-NEXT:       memref.store %3, %a[] : memref<f32>
# CHECK-NEXT:       %4 = arith.constant 0 : i32
# CHECK-NEXT:       func.return %4 : i32
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT: }
