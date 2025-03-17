# RUN: python3.13 %s | filecheck %s
from tenstorrent.frontend import tt


@tt.host
def nope():
    a = not False

# CHECK:      builtin.module {
# CHECK-NEXT:   builtin.module attributes {kernel_type = "host"} {
# CHECK-NEXT:     func.func @main() -> i32 {
# CHECK-NEXT:       %0 = arith.constant false
# CHECK-NEXT:       %1 = arith.constant true
# CHECK-NEXT:       %2 = arith.xori %0, %1 : i1
# CHECK-NEXT:       %a = memref.alloc() : memref<i1>
# CHECK-NEXT:       memref.store %2, %a[] : memref<i1>
# CHECK-NEXT:       %3 = arith.constant 0 : i32
# CHECK-NEXT:       func.return %3 : i32
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT: }
