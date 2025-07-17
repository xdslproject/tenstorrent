// RUN: python3.13 tenstorrent/tools/tt-opt %s -t tt-metalium | filecheck %s

builtin.module {
  builtin.module attributes {kernel_type = "compute"} {
    func.func @call_with_template() {
      %0 = arith.constant 3 : i32
      "comp.exp_tile"(%0) <{fast_and_approx = true}> : (i32) -> ()
      func.return
    }
  }
}

// CHECK: exp_tile<true>(3);
