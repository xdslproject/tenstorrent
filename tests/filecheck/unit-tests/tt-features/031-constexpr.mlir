// RUN: python3.13 tenstorrent/tools/tt-opt %s -t tt-metalium | filecheck %s

builtin.module {
  builtin.module attributes {kernel_type = "data_in"} {
    func.func @kernel_main() {
      %0 = arith.constant 5 : i32
      %1 = builtin.unrealized_conversion_cast %0 : i32 to ui32
      %2 = "ttshared.get_compile_time_arg_val"(%1) : (ui32) -> !ttshared.constexpr<ui32>
      %a = memref.alloc() : memref<!ttshared.constexpr<ui32>>
      memref.store %2, %a[] : memref<!ttshared.constexpr<ui32>>
      func.return
    }
  }
}

// CHECK:      void kernel_main() {
// CHECK-NEXT:     constexpr uint32_t a = get_compile_time_arg_val(static_cast<std::uint32_t>(5));
// CHECK-NEXT: }