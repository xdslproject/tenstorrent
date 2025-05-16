# RUN: python3.13 %s | filecheck %s
from tenstorrent.frontend import tt


@tt.host
def if_block():
    if True:
        a = 1
        a = 2
        a = 3
        a = 4


# CHECK:      builtin.module {
# CHECK-NEXT:   builtin.module attributes {kernel_type = "host"} {
# CHECK-NEXT:     func.func @main() -> i32 {
# CHECK-NEXT:       %a = memref.alloc() : memref<i32>
# CHECK-NEXT:       %0 = arith.constant true
# CHECK-NEXT:       scf.if %0 {
# CHECK-NEXT:         %1 = arith.constant 1 : i32
# CHECK-NEXT:         memref.store %1, %a[] : memref<i32>
# CHECK-NEXT:         %2 = arith.constant 2 : i32
# CHECK-NEXT:         memref.store %2, %a[] : memref<i32>
# CHECK-NEXT:         %3 = arith.constant 3 : i32
# CHECK-NEXT:         memref.store %3, %a[] : memref<i32>
# CHECK-NEXT:         %4 = arith.constant 4 : i32
# CHECK-NEXT:         memref.store %4, %a[] : memref<i32>
# CHECK-NEXT:       }
# CHECK-NEXT:       %5 = arith.constant 0 : i32
# CHECK-NEXT:       func.return %5 : i32
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT: }
