# # RUN: python3.13 %s | filecheck %s
from tenstorrent.frontend import tt


@tt.host
def simple_binop_add():
    a = 2 + 3

# CHECK:      builtin.module {
# CHECK-NEXT:   builtin.module attributes {kernel_type = "host"} {
# CHECK-NEXT:     func.func @main() -> i32 {
# CHECK-NEXT:       %0 = arith.constant 2 : i32
# CHECK-NEXT:       %1 = arith.constant 3 : i32
# CHECK-NEXT:       %2 = arith.addi %0, %1 : i32
# CHECK-NEXT:       %a = memref.alloc() : memref<i32>
# CHECK-NEXT:       memref.store %2, %a[] : memref<i32>
# CHECK-NEXT:       %3 = arith.constant 0 : i32
# CHECK-NEXT:       func.return %3 : i32
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT: }
