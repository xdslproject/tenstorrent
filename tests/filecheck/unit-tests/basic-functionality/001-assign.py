# RUN: python3.13 %s | filecheck %s
from tenstorrent.frontend import tt


@tt.host
def assign():
    a = 13


# CHECK:      builtin.module {
# CHECK-NEXT:   builtin.module attributes {kernel_type = "host"} {
# CHECK-NEXT:     func.func @main() -> i32 {
# CHECK-NEXT:       %0 = arith.constant 13 : i32
# CHECK-NEXT:       %a = memref.alloc() : memref<i32>
# CHECK-NEXT:       memref.store %0, %a[] : memref<i32>
# CHECK-NEXT:       %1 = arith.constant 0 : i32
# CHECK-NEXT:       func.return %1 : i32
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT: }
