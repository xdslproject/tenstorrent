# RUN: python3.13 %s | filecheck %s
from tenstorrent.frontend import tt


@tt.compute
def constexpr_prop_compare():
    # should be a constexpr<i1>
    a = tt.get_compile_time_arg_val(1) == 0


@tt.compute
def constexpr_prop_add():
    b = tt.get_compile_time_arg_val(2) + 5


@tt.compute
def constexpr_prop_div():
    c = tt.get_compile_time_arg_val(3) / 7.5


# CHECK:      %0 = arith.constant 1 : i32
# CHECK-NEXT: %1 = "ttshared.get_compile_time_arg_val"(%0) : (i32) -> !ttshared.constexpr<i32>
# CHECK-NEXT: %2 = arith.constant 0 : i32
# CHECK-NEXT: %3 = builtin.unrealized_conversion_cast %1 : !ttshared.constexpr<i32> to i32
# CHECK-NEXT: %4 = arith.cmpi eq, %3, %2 : i32
# CHECK-NEXT: %5 = builtin.unrealized_conversion_cast %4 : i1 to !ttshared.constexpr<i1>

# CHECK:      %0 = arith.constant 2 : i32
# CHECK-NEXT: %1 = "ttshared.get_compile_time_arg_val"(%0) : (i32) -> !ttshared.constexpr<i32>
# CHECK-NEXT: %2 = arith.constant 5 : i32
# CHECK-NEXT: %3 = builtin.unrealized_conversion_cast %1 : !ttshared.constexpr<i32> to i32
# CHECK-NEXT: %4 = arith.addi %3, %2 : i32
# CHECK-NEXT: %5 = builtin.unrealized_conversion_cast %4 : i32 to !ttshared.constexpr<i32>

# CHECK:      %0 = arith.constant 3 : i32
# CHECK-NEXT: %1 = "ttshared.get_compile_time_arg_val"(%0) : (i32) -> !ttshared.constexpr<i32>
# CHECK-NEXT: %2 = arith.constant 7.500000e+00 : f32
# CHECK-NEXT: %3 = builtin.unrealized_conversion_cast %1 : !ttshared.constexpr<i32> to i32
# CHECK-NEXT: %4 = arith.sitofp %3 : i32 to f32
# CHECK-NEXT: %5 = arith.divf %4, %2 : f32
# CHECK-NEXT: %6 = builtin.unrealized_conversion_cast %5 : f32 to !ttshared.constexpr<f32>
