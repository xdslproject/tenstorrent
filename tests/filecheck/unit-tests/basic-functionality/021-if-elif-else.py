# RUN: python3.13 %s | filecheck %s
from tenstorrent.frontend import tt


@tt.host
def if_elif_else():
    a = True
    b = 1

    if a:
        b = 2
    elif a:
        b = 3
    else:
        b = 4


# CHECK:      builtin.module {
# CHECK-NEXT:   builtin.module attributes {kernel_type = "host"} {
# CHECK-NEXT:     func.func @main() -> i32 {
# CHECK-NEXT:       %0 = arith.constant true
# CHECK-NEXT:       %a = memref.alloc() : memref<i1>
# CHECK-NEXT:       memref.store %0, %a[] : memref<i1>
# CHECK-NEXT:       %1 = arith.constant 1 : i32
# CHECK-NEXT:       %b = memref.alloc() : memref<i32>
# CHECK-NEXT:       memref.store %1, %b[] : memref<i32>
# CHECK-NEXT:       %2 = memref.load %a[] : memref<i1>
# CHECK-NEXT:       scf.if %2 {
# CHECK-NEXT:         %3 = arith.constant 2 : i32
# CHECK-NEXT:         memref.store %3, %b[] : memref<i32>
# CHECK-NEXT:       } else {
# CHECK-NEXT:         %4 = memref.load %a[] : memref<i1>
# CHECK-NEXT:         scf.if %4 {
# CHECK-NEXT:           %5 = arith.constant 3 : i32
# CHECK-NEXT:           memref.store %5, %b[] : memref<i32>
# CHECK-NEXT:         } else {
# CHECK-NEXT:           %6 = arith.constant 4 : i32
# CHECK-NEXT:           memref.store %6, %b[] : memref<i32>
# CHECK-NEXT:         }
# CHECK-NEXT:       }
# CHECK-NEXT:       %7 = arith.constant 0 : i32
# CHECK-NEXT:       func.return %7 : i32
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT: }
