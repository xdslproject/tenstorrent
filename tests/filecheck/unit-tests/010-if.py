# RUN: python3.13 %s | filecheck %s
from tenstorrent.frontend import tt


@tt.host
def if_statement():
    if True:
        a = 5

# CHECK:      builtin.module {
# CHECK-NEXT:   builtin.module attributes {kernel_type = "host"} {
# CHECK-NEXT:     func.func @main() -> i32 {
# CHECK-NEXT:       %a = memref.alloc() : memref<i32>
# CHECK-NEXT:       %0 = arith.constant true
# CHECK-NEXT:       scf.if %0 {
# CHECK-NEXT:         %1 = arith.constant 5 : i32
# CHECK-NEXT:         memref.store %1, %a[] : memref<i32>
# CHECK-NEXT:       }
# CHECK-NEXT:       %2 = arith.constant 0 : i32
# CHECK-NEXT:       func.return %2 : i32
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT: }
