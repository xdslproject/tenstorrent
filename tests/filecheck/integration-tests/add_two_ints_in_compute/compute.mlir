// RUN: python3.13 tenstorrent/tools/tt-opt %s -t tt-metalium | filecheck %s

builtin.module {
  builtin.module attributes {kernel_type = "compute"} {
    func.func @add_two_int_tiles() {
      %0 = arith.constant 0 : i32
      %cb0 = memref.alloc() : memref<i32>
      memref.store %0, %cb0[] : memref<i32>
      %1 = arith.constant 1 : i32
      %cb1 = memref.alloc() : memref<i32>
      memref.store %1, %cb1[] : memref<i32>
      %2 = arith.constant 16 : i32
      %cb_out = memref.alloc() : memref<i32>
      memref.store %2, %cb_out[] : memref<i32>
      %3 = memref.load %cb0[] : memref<i32>
      %4 = memref.load %cb1[] : memref<i32>
      %5 = memref.load %cb_out[] : memref<i32>
      "comp.binary_op_init_common"(%3, %4, %5) : (i32, i32, i32) -> ()
      %6 = memref.load %cb0[] : memref<i32>
      %7 = memref.load %cb1[] : memref<i32>
      %8 = arith.constant false
      "comp.add_tiles_init"(%6, %7, %8) : (i32, i32, i1) -> ()
      %9 = memref.load %cb0[] : memref<i32>
      %10 = arith.constant 1 : i32
      "cb.wait_front"(%9, %10) : (i32, i32) -> ()
      %11 = memref.load %cb1[] : memref<i32>
      %12 = arith.constant 1 : i32
      "cb.wait_front"(%11, %12) : (i32, i32) -> ()
      "comp.tile_regs_acquire"() : () -> ()
      %13 = memref.load %cb0[] : memref<i32>
      %14 = memref.load %cb1[] : memref<i32>
      %15 = arith.constant 0 : i32
      %16 = arith.constant 0 : i32
      %17 = arith.constant 0 : i32
      "comp.add_tiles"(%13, %14, %15, %16, %17) : (i32, i32, i32, i32, i32) -> ()
      "comp.tile_regs_commit"() : () -> ()
      "comp.tile_regs_wait"() : () -> ()
      %18 = arith.constant 0 : i32
      %19 = memref.load %cb_out[] : memref<i32>
      %20 = arith.constant 0 : i32
      "comp.pack_tile"(%18, %19, %20) <{out_of_order_output = false}> : (i32, i32, i32) -> ()
      "comp.tile_regs_release"() : () -> ()
      %21 = memref.load %cb0[] : memref<i32>
      %22 = arith.constant 1 : i32
      "cb.pop_front"(%21, %22) : (i32, i32) -> ()
      %23 = memref.load %cb1[] : memref<i32>
      %24 = arith.constant 1 : i32
      "cb.pop_front"(%23, %24) : (i32, i32) -> ()
      %25 = memref.load %cb_out[] : memref<i32>
      %26 = arith.constant 1 : i32
      "cb.push_back"(%25, %26) : (i32, i32) -> ()
      func.return
    }
  }
}

// CHECK:      #include <cstdint>
// CHECK:      #include "compute_kernel_api/common.h"
// CHECK:      #include "compute_kernel_api/tile_move_copy.h"
// CHECK:      #include "compute_kernel_api/eltwise_binary.h"
// CHECK:      namespace NAMESPACE {
// CHECK:      void MAIN {
// CHECK-NEXT:     std::int32_t cb0 = 0;
// CHECK-NEXT:     std::int32_t cb1 = 1;
// CHECK-NEXT:     std::int32_t cb_out = 16;
// CHECK-NEXT:     binary_op_init_common(cb0, cb1, cb_out);
// CHECK-NEXT:     add_tiles_init(cb0, cb1, false);
// CHECK-NEXT:     cb_wait_front(cb0, 1);
// CHECK-NEXT:     cb_wait_front(cb1, 1);
// CHECK-NEXT:     tile_regs_acquire();
// CHECK-NEXT:     add_tiles(cb0, cb1, 0, 0, 0);
// CHECK-NEXT:     tile_regs_commit();
// CHECK-NEXT:     tile_regs_wait();
// CHECK-NEXT:     pack_tile<false>(0, cb_out, 0);
// CHECK-NEXT:     tile_regs_release();
// CHECK-NEXT:     cb_pop_front(cb0, 1);
// CHECK-NEXT:     cb_pop_front(cb1, 1);
// CHECK-NEXT:     cb_push_back(cb_out, 1);
// CHECK-NEXT: }
// CHECK: }
