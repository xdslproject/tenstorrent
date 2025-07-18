// RUN: python3.13 tenstorrent/tools/tt-opt %s -t tt-metalium | filecheck %s

builtin.module {
  builtin.module attributes {kernel_type = "data_in"} {
    func.func @kernel_main() {
      %0 = arith.constant 5 : i32
      %2 = "ttshared.get_compile_time_arg_val"(%0) : (i32) -> !ttshared.constexpr<i32>
      %a = memref.alloc() : memref<!ttshared.constexpr<i32>>
      memref.store %2, %a[] : memref<!ttshared.constexpr<i32>>
      func.return
    }
  }
}

// CHECK: constexpr {{[a-zA-Z0-9_:]*}} a = get_compile_time_arg_val(5);
