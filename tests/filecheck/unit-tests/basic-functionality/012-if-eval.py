# RUN: python3.13 %s | filecheck %s
from tenstorrent.frontend import tt


@tt.host
def if_eval():
    a = 5
    b = 3

    if a < b:
        a = 5


# CHECK:      builtin.module {
# CHECK-NEXT:   builtin.module attributes {kernel_type = "host"} {
# CHECK-NEXT:     func.func @main() -> i32 {
# CHECK-NEXT:       %0 = arith.constant 5 : i32
# CHECK-NEXT:       %a = memref.alloc() : memref<i32>
# CHECK-NEXT:       memref.store %0, %a[] : memref<i32>
# CHECK-NEXT:       %1 = arith.constant 3 : i32
# CHECK-NEXT:       %b = memref.alloc() : memref<i32>
# CHECK-NEXT:       memref.store %1, %b[] : memref<i32>
# CHECK-NEXT:       %2 = memref.load %a[] : memref<i32>
# CHECK-NEXT:       %3 = memref.load %b[] : memref<i32>
# CHECK-NEXT:       %4 = arith.cmpi ult, %2, %3 : i32
# CHECK-NEXT:       scf.if %4 {
# CHECK-NEXT:         %5 = arith.constant 5 : i32
# CHECK-NEXT:         memref.store %5, %a[] : memref<i32>
# CHECK-NEXT:       }
# CHECK-NEXT:       %6 = arith.constant 0 : i32
# CHECK-NEXT:       func.return %6 : i32
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT: }
