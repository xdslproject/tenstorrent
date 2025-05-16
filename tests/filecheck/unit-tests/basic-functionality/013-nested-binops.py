# RUN: python3.13 %s | filecheck %s
from tenstorrent.frontend import tt


@tt.host
def nested_binops():
    a = 1 + 2 * 3 + 4


# CHECK:      builtin.module {
# CHECK-NEXT:   builtin.module attributes {kernel_type = "host"} {
# CHECK-NEXT:     func.func @main() -> i32 {
# CHECK-NEXT:       %0 = arith.constant 1 : i32
# CHECK-NEXT:       %1 = arith.constant 2 : i32
# CHECK-NEXT:       %2 = arith.constant 3 : i32
# CHECK-NEXT:       %3 = arith.muli %1, %2 : i32
# CHECK-NEXT:       %4 = arith.addi %0, %3 : i32
# CHECK-NEXT:       %5 = arith.constant 4 : i32
# CHECK-NEXT:       %6 = arith.addi %4, %5 : i32
# CHECK-NEXT:       %a = memref.alloc() : memref<i32>
# CHECK-NEXT:       memref.store %6, %a[] : memref<i32>
# CHECK-NEXT:       %7 = arith.constant 0 : i32
# CHECK-NEXT:       func.return %7 : i32
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT: }
