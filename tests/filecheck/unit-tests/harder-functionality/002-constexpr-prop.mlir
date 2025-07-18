// RUN: python3.13 tenstorrent/tools/tt-opt %s -t tt-metalium | filecheck %s

builtin.module {
  builtin.module attributes {kernel_type = "compute"} {
    func.func @constexpr_prop_compare() {
      %0 = arith.constant 1 : i32
      %1 = "ttshared.get_compile_time_arg_val"(%0) : (i32) -> !ttshared.constexpr<i32>
      %2 = arith.constant 0 : i32
      %3 = builtin.unrealized_conversion_cast %1 : !ttshared.constexpr<i32> to i32
      %4 = arith.cmpi eq, %3, %2 : i32
      %5 = builtin.unrealized_conversion_cast %4 : i1 to !ttshared.constexpr<i1>
      %a = memref.alloc() : memref<!ttshared.constexpr<i1>>
      memref.store %5, %a[] : memref<!ttshared.constexpr<i1>>
      func.return
    }
  }
  builtin.module attributes {kernel_type = "compute"} {
    func.func @constexpr_prop_add() {
      %0 = arith.constant 2 : i32
      %1 = "ttshared.get_compile_time_arg_val"(%0) : (i32) -> !ttshared.constexpr<i32>
      %2 = arith.constant 5 : i32
      %3 = builtin.unrealized_conversion_cast %1 : !ttshared.constexpr<i32> to i32
      %4 = arith.addi %3, %2 : i32
      %5 = builtin.unrealized_conversion_cast %4 : i32 to !ttshared.constexpr<i32>
      %b = memref.alloc() : memref<!ttshared.constexpr<i32>>
      memref.store %5, %b[] : memref<!ttshared.constexpr<i32>>
      func.return
    }
  }
  builtin.module attributes {kernel_type = "compute"} {
    func.func @constexpr_prop_div() {
      %0 = arith.constant 3 : i32
      %1 = "ttshared.get_compile_time_arg_val"(%0) : (i32) -> !ttshared.constexpr<i32>
      %2 = arith.constant 7.500000e+00 : f32
      %3 = builtin.unrealized_conversion_cast %1 : !ttshared.constexpr<i32> to i32
      %4 = arith.sitofp %3 : i32 to f32
      %5 = arith.divf %4, %2 : f32
      %6 = builtin.unrealized_conversion_cast %5 : f32 to !ttshared.constexpr<f32>
      %c = memref.alloc() : memref<!ttshared.constexpr<f32>>
      memref.store %6, %c[] : memref<!ttshared.constexpr<f32>>
      func.return
    }
  }
}

// CHECK: constexpr bool a = get_compile_time_arg_val(1) == 0;
// CHECK: constexpr std::int32_t b = get_compile_time_arg_val(2) + 5;
// CHECK: constexpr float c = static_cast<float>(get_compile_time_arg_val(3)) / 7.5;