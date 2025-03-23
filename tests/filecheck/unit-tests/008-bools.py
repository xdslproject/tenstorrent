# RUN: python3.13 %s | filecheck %s
from tenstorrent.frontend import tt


@tt.host
def bools():
    a = True
    b = False


# CHECK:      builtin.module {
# CHECK-NEXT:   builtin.module attributes {kernel_type = "host"} {
# CHECK-NEXT:     func.func @main() -> i32 {
# CHECK-NEXT:       %0 = arith.constant true
# CHECK-NEXT:       %a = memref.alloc() : memref<i1>
# CHECK-NEXT:       memref.store %0, %a[] : memref<i1>
# CHECK-NEXT:       %1 = arith.constant false
# CHECK-NEXT:       %b = memref.alloc() : memref<i1>
# CHECK-NEXT:       memref.store %1, %b[] : memref<i1>
# CHECK-NEXT:       %2 = arith.constant 0 : i32
# CHECK-NEXT:       func.return %2 : i32
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT: }
