# RUN: python3.13 %s | filecheck %s
from tenstorrent.frontend import tt


@tt.host
def read_variable():
    a = 5
    b = a + 1

# CHECK:      builtin.module {
# CHECK-NEXT:   builtin.module attributes {kernel_type = "host"} {
# CHECK-NEXT:     func.func @main() -> i32 {
# CHECK-NEXT:       %0 = arith.constant 5 : i32
# CHECK-NEXT:       %a = memref.alloc() : memref<i32>
# CHECK-NEXT:       memref.store %0, %a[] : memref<i32>
# CHECK-NEXT:       %1 = memref.load %a[] : memref<i32>
# CHECK-NEXT:       %2 = arith.constant 1 : i32
# CHECK-NEXT:       %3 = arith.addi %1, %2 : i32
# CHECK-NEXT:       %b = memref.alloc() : memref<i32>
# CHECK-NEXT:       memref.store %3, %b[] : memref<i32>
# CHECK-NEXT:       %4 = arith.constant 0 : i32
# CHECK-NEXT:       func.return %4 : i32
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT: }
