# RUN: python3.13 %s | filecheck %s
from tenstorrent.frontend import tt


@tt.data_in
def constexpr():
    a = tt.get_compile_time_arg_val(5)


# CHECK:      builtin.module {
# CHECK-NEXT:   builtin.module attributes {kernel_type = "data_in"} {
# CHECK-NEXT:     func.func @kernel_main() {
# CHECK-NEXT:       %0 = arith.constant 5 : i32
# CHECK-NEXT:       %1 = builtin.unrealized_conversion_cast %0 : i32 to ui32
# CHECK-NEXT:       %2 = "ttshared.get_compile_time_arg_val"(%1) : (ui32) -> !ttshared.constexpr<ui32>
# CHECK-NEXT:       %a = memref.alloc() : memref<!ttshared.constexpr<ui32>>
# CHECK-NEXT:       memref.store %2, %a[] : memref<!ttshared.constexpr<ui32>>
# CHECK-NEXT:       func.return
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT: }
