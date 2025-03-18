# RUN: python3.13 %s | filecheck %s
from tenstorrent.frontend import tt


@tt.host
def for_loop_iref():
    a = 0

    for i in range(0, 50):
        a = i


# CHECK:      builtin.module {
# CHECK-NEXT:   builtin.module attributes {kernel_type = "host"} {
# CHECK-NEXT:     func.func @main() -> i32 {
# CHECK-NEXT:       %0 = arith.constant 0 : i32
# CHECK-NEXT:       %a = memref.alloc() : memref<i32>
# CHECK-NEXT:       memref.store %0, %a[] : memref<i32>
# CHECK-NEXT:       %1 = arith.constant 0 : i32
# CHECK-NEXT:       %2 = arith.constant 50 : i32
# CHECK-NEXT:       %3 = arith.constant 1 : i32
# CHECK-NEXT:       %i = memref.alloc() : memref<i32>
# CHECK-NEXT:       scf.for %4 = %1 to %2 step %3  : i32 {
# CHECK-NEXT:         memref.store %4, %i[] : memref<i32>
# CHECK-NEXT:         %5 = memref.load %i[] : memref<i32>
# CHECK-NEXT:         memref.store %5, %a[] : memref<i32>
# CHECK-NEXT:       }
# CHECK-NEXT:       %6 = arith.constant 0 : i32
# CHECK-NEXT:       func.return %6 : i32
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT: }
