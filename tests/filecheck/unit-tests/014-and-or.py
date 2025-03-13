# RUN: python3.13 %s | filecheck %s
from tenstorrent.frontend import tt


@tt.host
def and_or():
    a = True
    b = False
    c = a and b
    d = a or b

# CHECK:      builtin.module {
# CHECK-NEXT:   builtin.module attributes {kernel_type = "host"} {
# CHECK-NEXT:     func.func @main() -> i32 {
# CHECK-NEXT:       %0 = arith.constant true
# CHECK-NEXT:       %a = memref.alloc() : memref<i1>
# CHECK-NEXT:       memref.store %0, %a[] : memref<i1>
# CHECK-NEXT:       %1 = arith.constant false
# CHECK-NEXT:       %b = memref.alloc() : memref<i1>
# CHECK-NEXT:       memref.store %1, %b[] : memref<i1>
# CHECK-NEXT:       %2 = memref.load %a[] : memref<i1>
# CHECK-NEXT:       %3 = memref.load %b[] : memref<i1>
# CHECK-NEXT:       %4 = arith.andi %2, %3 : i1
# CHECK-NEXT:       %c = memref.alloc() : memref<i1>
# CHECK-NEXT:       memref.store %4, %c[] : memref<i1>
# CHECK-NEXT:       %5 = memref.load %a[] : memref<i1>
# CHECK-NEXT:       %6 = memref.load %b[] : memref<i1>
# CHECK-NEXT:       %7 = arith.ori %5, %6 : i1
# CHECK-NEXT:       %d = memref.alloc() : memref<i1>
# CHECK-NEXT:       memref.store %7, %d[] : memref<i1>
# CHECK-NEXT:       %8 = arith.constant 0 : i32
# CHECK-NEXT:       func.return %8 : i32
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT: }
