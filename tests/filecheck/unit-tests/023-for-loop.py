# RUN: python3.13 %s | filecheck %s
from tenstorrent.frontend import tt


@tt.host
def for_loop():
    for i in range(0, 50):
        a = 5


# CHECK:      builtin.module {
# CHECK-NEXT:   builtin.module attributes {kernel_type = "host"} {
# CHECK-NEXT:     func.func @main() -> i32 {
# CHECK-NEXT:       %a = memref.alloc() : memref<i32>
# CHECK-NEXT:       %0 = arith.constant 0 : i32
# CHECK-NEXT:       %1 = arith.constant 50 : i32
# CHECK-NEXT:       %2 = arith.constant 1 : i32
# CHECK-NEXT:       %i = memref.alloc() : memref<i32>
# CHECK-NEXT:       scf.for %3 = %0 to %1 step %2  : i32 {
# CHECK-NEXT:         memref.store %3, %i[] : memref<i32>
# CHECK-NEXT:         %4 = arith.constant 5 : i32
# CHECK-NEXT:         memref.store %4, %a[] : memref<i32>
# CHECK-NEXT:       }
# CHECK-NEXT:       %5 = arith.constant 0 : i32
# CHECK-NEXT:       func.return %5 : i32
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT: }
