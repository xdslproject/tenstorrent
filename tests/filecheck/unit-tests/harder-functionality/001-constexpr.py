# RUN: python3.13 %s | filecheck %s
from tenstorrent.frontend import tt


@tt.data_in
def constexpr():
    a = tt.get_compile_time_arg_val(5)


# CHECK:            %0 = arith.constant 5 : i32
# CHECK-NEXT:       %1 = "ttshared.get_compile_time_arg_val"(%0) : (i32) -> !ttshared.constexpr<i32>
# CHECK-NEXT:       %a = memref.alloc() : memref<!ttshared.constexpr<i32>>
# CHECK-NEXT:       memref.store %1, %a[] : memref<!ttshared.constexpr<i32>>
