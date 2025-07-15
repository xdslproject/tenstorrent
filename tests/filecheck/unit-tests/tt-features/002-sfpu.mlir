// RUN: python3.13 tenstorrent/tools/tt-opt %s -t tt-metalium | filecheck %s

builtin.module {
  builtin.module attributes {kernel_type = "compute"} {
    func.func @MAIN() {
      %cb_in = arith.constant 0 : i32
      %cb_out = arith.constant 1 : i32
      %0 = arith.constant 0 : i32
      %u_0 = builtin.unrealized_conversion_cast %0 : i32 to ui32
      %1 = arith.constant 1 : i32
      %u_1 = builtin.unrealized_conversion_cast %1 : i32 to ui32

      // initialise the sfpu with the right operation
      "comp.init_sfpu"(%cb_in, %cb_out) : (i32, i32) -> ()
      "comp.exp_tile_init"() <{ fast_and_approx = false }> : () -> ()

      // grab lock on (I think...) dst
      "comp.tile_regs_acquire"() : () -> ()
      "cb.wait_front"(%cb_in, %1) : (i32, i32) -> ()

      // manually copy the tile from the circular buffer to (i think) dst[0]
      "comp.copy_tile"(%u_0, %u_0, %u_0): (ui32, ui32, ui32) -> ()

      // perform sfpu function
      "comp.exp_tile"(%u_0) <{ fast_and_approx = false }> : (ui32) -> ()
      "comp.tile_regs_commit"() : () -> ()
      "comp.tile_regs_wait"() : () -> ()

      // dst[0] -> cb_out and cleanup
      "cb.reserve_back"(%cb_out, %1) : (i32, i32) -> ()
      "comp.pack_tile"(%u_0, %u_1, %u_0) <{out_of_order_output = false}> : (ui32, ui32, ui32) -> ()
      "cb.pop_front"(%cb_in, %1) : (i32, i32) -> ()
      "comp.tile_regs_release"() : () -> ()
      "cb.push_back"(%cb_out, %1) : (i32, i32) -> ()

      func.return
    }
  }
}

// CHECK:      #include <cstdint>
// CHECK:      #include "compute_kernel_api/common.h"
// CHECK:      #include "compute_kernel_api/tile_move_copy.h"
// CHECK:      #include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
// CHECK:      #include "compute_kernel_api/eltwise_unary/exp.h"

// CHECK:      namespace NAMESPACE {
// CHECK:      void MAIN {
// CHECK-NEXT:     init_sfpu(0, 1);
// CHECK-NEXT:     exp_tile_init<false>();
// CHECK-NEXT:     tile_regs_acquire();
// CHECK-NEXT:     cb_wait_front(0, 1);
// CHECK-NEXT:     copy_tile(static_cast<std::uint32_t>(0), static_cast<std::uint32_t>(0), static_cast<std::uint32_t>(0));
// CHECK-NEXT:     exp_tile<false>(static_cast<std::uint32_t>(0));
// CHECK-NEXT:     tile_regs_commit();
// CHECK-NEXT:     tile_regs_wait();
// CHECK-NEXT:     cb_reserve_back(1, 1);
// CHECK-NEXT:     pack_tile<false>(static_cast<std::uint32_t>(0), static_cast<std::uint32_t>(1), static_cast<std::uint32_t>(0));
// CHECK-NEXT:     cb_pop_front(0, 1);
// CHECK-NEXT:     tile_regs_release();
// CHECK-NEXT:     cb_push_back(1, 1);
// CHECK-NEXT: }
// CHECK:      }
