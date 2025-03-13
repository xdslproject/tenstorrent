# RUN: python3.13 %s | filecheck %s
from tenstorrent.frontend import tt


@tt.host
def mixed_conditional():
    a = 1
    b = 2
    c = 5

    if a == 1 and b >= 2 or c < b + a:
        b = 9

# CHECK:      builtin.module {
# CHECK-NEXT:   builtin.module attributes {kernel_type = "host"} {
# CHECK-NEXT:     func.func @main() -> i32 {
# CHECK-NEXT:       %0 = arith.constant 1 : i32
# CHECK-NEXT:       %a = memref.alloc() : memref<i32>
# CHECK-NEXT:       memref.store %0, %a[] : memref<i32>
# CHECK-NEXT:       %1 = arith.constant 2 : i32
# CHECK-NEXT:       %b = memref.alloc() : memref<i32>
# CHECK-NEXT:       memref.store %1, %b[] : memref<i32>
# CHECK-NEXT:       %2 = arith.constant 5 : i32
# CHECK-NEXT:       %c = memref.alloc() : memref<i32>
# CHECK-NEXT:       memref.store %2, %c[] : memref<i32>
# CHECK-NEXT:       %3 = memref.load %a[] : memref<i32>
# CHECK-NEXT:       %4 = arith.constant 1 : i32
# CHECK-NEXT:       %5 = arith.cmpi eq, %3, %4 : i32
# CHECK-NEXT:       %6 = memref.load %b[] : memref<i32>
# CHECK-NEXT:       %7 = arith.constant 2 : i32
# CHECK-NEXT:       %8 = arith.cmpi uge, %6, %7 : i32
# CHECK-NEXT:       %9 = arith.andi %5, %8 : i1
# CHECK-NEXT:       %10 = memref.load %c[] : memref<i32>
# CHECK-NEXT:       %11 = memref.load %b[] : memref<i32>
# CHECK-NEXT:       %12 = memref.load %a[] : memref<i32>
# CHECK-NEXT:       %13 = arith.addi %11, %12 : i32
# CHECK-NEXT:       %14 = arith.cmpi ult, %10, %13 : i32
# CHECK-NEXT:       %15 = arith.ori %9, %14 : i1
# CHECK-NEXT:       scf.if %15 {
# CHECK-NEXT:         %16 = arith.constant 9 : i32
# CHECK-NEXT:         memref.store %16, %b[] : memref<i32>
# CHECK-NEXT:       }
# CHECK-NEXT:       %17 = arith.constant 0 : i32
# CHECK-NEXT:       func.return %17 : i32
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT: }
