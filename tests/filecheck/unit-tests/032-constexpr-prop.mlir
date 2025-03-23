// RUN: python3.13 tenstorrent/tools/tt-opt %s -t tt-metalium | filecheck %s

builtin.module {
  builtin.module attributes {kernel_type = "data_in"} {
    func.func @kernel_main() {
      %0 = arith.constant 1 : i32
      %1 = builtin.unrealized_conversion_cast %0 : i32 to ui32
      %2 = "ttshared.get_compile_time_arg_val"(%1) : (ui32) -> !ttshared.constexpr<ui32>
      %3 = arith.constant 0 : i32
      %4 = builtin.unrealized_conversion_cast %2 : !ttshared.constexpr<ui32> to ui32
      %5 = builtin.unrealized_conversion_cast %4 : ui32 to i32
      %6 = arith.cmpi eq, %5, %3 : i32
      %7 = builtin.unrealized_conversion_cast %6 : i1 to !ttshared.constexpr<i1>
      %a = memref.alloc() : memref<!ttshared.constexpr<i1>>
      memref.store %7, %a[] : memref<!ttshared.constexpr<i1>>
      func.return
    }
  }
  builtin.module attributes {kernel_type = "data_in"} {
    func.func @kernel_main() {
      %0 = arith.constant 2 : i32
      %1 = builtin.unrealized_conversion_cast %0 : i32 to ui32
      %2 = "ttshared.get_compile_time_arg_val"(%1) : (ui32) -> !ttshared.constexpr<ui32>
      %3 = arith.constant 5 : i32
      %4 = builtin.unrealized_conversion_cast %2 : !ttshared.constexpr<ui32> to ui32
      %5 = builtin.unrealized_conversion_cast %4 : ui32 to i32
      %6 = arith.addi %5, %3 : i32
      %7 = builtin.unrealized_conversion_cast %6 : i32 to !ttshared.constexpr<i32>
      %b = memref.alloc() : memref<!ttshared.constexpr<i32>>
      memref.store %7, %b[] : memref<!ttshared.constexpr<i32>>
      func.return
    }
  }
  builtin.module attributes {kernel_type = "data_in"} {
    func.func @kernel_main() {
      %0 = arith.constant 3 : i32
      %1 = builtin.unrealized_conversion_cast %0 : i32 to ui32
      %2 = "ttshared.get_compile_time_arg_val"(%1) : (ui32) -> !ttshared.constexpr<ui32>
      %3 = arith.constant 7.500000e+00 : f32
      %4 = builtin.unrealized_conversion_cast %2 : !ttshared.constexpr<ui32> to ui32
      %5 = builtin.unrealized_conversion_cast %4 : ui32 to si32
      %6 = arith.sitofp %5 : si32 to f32
      %7 = arith.divf %6, %3 : f32
      %8 = builtin.unrealized_conversion_cast %7 : f32 to !ttshared.constexpr<f32>
      %c = memref.alloc() : memref<!ttshared.constexpr<f32>>
      memref.store %8, %c[] : memref<!ttshared.constexpr<f32>>
      func.return
    }
  }
}

// CHECK: constexpr bool a = static_cast<std::int32_t>(get_compile_time_arg_val(static_cast<std::uint32_t>(1))) == 0;
// CHECK: constexpr std::int32_t b = static_cast<std::int32_t>(get_compile_time_arg_val(static_cast<std::uint32_t>(2))) + 5;
// CHECK: constexpr float c = static_cast<float>(static_cast<std::int32_t>(get_compile_time_arg_val(static_cast<std::uint32_t>(3)))) / 7.5;