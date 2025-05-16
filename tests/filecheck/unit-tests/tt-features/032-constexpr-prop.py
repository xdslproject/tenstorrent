# RUN: python3.13 %s | filecheck %s
from tenstorrent.frontend import tt


@tt.data_in
def constexpr_prop_compare():
    # should be a constexpr<i1>
    a = tt.get_compile_time_arg_val(1) == 0


@tt.data_in
def constexpr_prop_add():
    b = tt.get_compile_time_arg_val(2) + 5


@tt.data_in
def constexpr_prop_div():
    c = tt.get_compile_time_arg_val(3) / 7.5


# CHECK:      builtin.module {
# CHECK-NEXT:   builtin.module attributes {kernel_type = "data_in"} {
# CHECK-NEXT:     func.func @kernel_main() {
# CHECK-NEXT:       %0 = arith.constant 1 : i32
# CHECK-NEXT:       %1 = builtin.unrealized_conversion_cast %0 : i32 to ui32
# CHECK-NEXT:       %2 = "ttshared.get_compile_time_arg_val"(%1) : (ui32) -> !ttshared.constexpr<ui32>
# CHECK-NEXT:       %3 = arith.constant 0 : i32
# CHECK-NEXT:       %4 = builtin.unrealized_conversion_cast %2 : !ttshared.constexpr<ui32> to ui32
# CHECK-NEXT:       %5 = builtin.unrealized_conversion_cast %4 : ui32 to i32
# CHECK-NEXT:       %6 = arith.cmpi eq, %5, %3 : i32
# CHECK-NEXT:       %7 = builtin.unrealized_conversion_cast %6 : i1 to !ttshared.constexpr<i1>
# CHECK-NEXT:       %a = memref.alloc() : memref<!ttshared.constexpr<i1>>
# CHECK-NEXT:       memref.store %7, %a[] : memref<!ttshared.constexpr<i1>>
# CHECK-NEXT:       func.return
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT:   builtin.module attributes {kernel_type = "data_in"} {
# CHECK-NEXT:     func.func @kernel_main() {
# CHECK-NEXT:       %0 = arith.constant 2 : i32
# CHECK-NEXT:       %1 = builtin.unrealized_conversion_cast %0 : i32 to ui32
# CHECK-NEXT:       %2 = "ttshared.get_compile_time_arg_val"(%1) : (ui32) -> !ttshared.constexpr<ui32>
# CHECK-NEXT:       %3 = arith.constant 5 : i32
# CHECK-NEXT:       %4 = builtin.unrealized_conversion_cast %2 : !ttshared.constexpr<ui32> to ui32
# CHECK-NEXT:       %5 = builtin.unrealized_conversion_cast %4 : ui32 to i32
# CHECK-NEXT:       %6 = arith.addi %5, %3 : i32
# CHECK-NEXT:       %7 = builtin.unrealized_conversion_cast %6 : i32 to !ttshared.constexpr<i32>
# CHECK-NEXT:       %b = memref.alloc() : memref<!ttshared.constexpr<i32>>
# CHECK-NEXT:       memref.store %7, %b[] : memref<!ttshared.constexpr<i32>>
# CHECK-NEXT:       func.return
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT:   builtin.module attributes {kernel_type = "data_in"} {
# CHECK-NEXT:     func.func @kernel_main() {
# CHECK-NEXT:       %0 = arith.constant 3 : i32
# CHECK-NEXT:       %1 = builtin.unrealized_conversion_cast %0 : i32 to ui32
# CHECK-NEXT:       %2 = "ttshared.get_compile_time_arg_val"(%1) : (ui32) -> !ttshared.constexpr<ui32>
# CHECK-NEXT:       %3 = arith.constant 7.500000e+00 : f32
# CHECK-NEXT:       %4 = builtin.unrealized_conversion_cast %2 : !ttshared.constexpr<ui32> to ui32
# CHECK-NEXT:       %5 = builtin.unrealized_conversion_cast %4 : ui32 to si32
# CHECK-NEXT:       %6 = arith.sitofp %5 : si32 to f32
# CHECK-NEXT:       %7 = arith.divf %6, %3 : f32
# CHECK-NEXT:       %8 = builtin.unrealized_conversion_cast %7 : f32 to !ttshared.constexpr<f32>
# CHECK-NEXT:       %c = memref.alloc() : memref<!ttshared.constexpr<f32>>
# CHECK-NEXT:       memref.store %8, %c[] : memref<!ttshared.constexpr<f32>>
# CHECK-NEXT:       func.return
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT: }
