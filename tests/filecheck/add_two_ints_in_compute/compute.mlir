// RUN: python3.13 tenstorrent/tools/tt-opt %s -t tt-metalium | filecheck %s

builtin.module {
  builtin.module attributes  {"kernel_type" = "compute"} {
    func.func @add_two_ints_vectorised() {
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
      %6 = builtin.unrealized_conversion_cast %3 : i32 to ui32
      %7 = builtin.unrealized_conversion_cast %4 : i32 to ui32
      %8 = builtin.unrealized_conversion_cast %5 : i32 to ui32
      "comp.binary_op_init_common"(%6, %7, %8) : (ui32, ui32, ui32) -> ()
      %9 = memref.load %cb0[] : memref<i32>
      %10 = memref.load %cb1[] : memref<i32>
      %11 = arith.constant false
      %12 = builtin.unrealized_conversion_cast %9 : i32 to ui32
      %13 = builtin.unrealized_conversion_cast %10 : i32 to ui32
      "comp.add_tiles_init"(%12, %13, %11) : (ui32, ui32, i1) -> ()
      %14 = memref.load %cb0[] : memref<i32>
      %15 = arith.constant 1 : i32
      "cb.wait_front"(%14, %15) : (i32, i32) -> ()
      %16 = memref.load %cb1[] : memref<i32>
      %17 = arith.constant 1 : i32
      "cb.wait_front"(%16, %17) : (i32, i32) -> ()
      "comp.tile_regs_acquire"() : () -> ()
      %18 = memref.load %cb0[] : memref<i32>
      %19 = memref.load %cb1[] : memref<i32>
      %20 = arith.constant 0 : i32
      %21 = arith.constant 0 : i32
      %22 = arith.constant 0 : i32
      %23 = builtin.unrealized_conversion_cast %18 : i32 to ui32
      %24 = builtin.unrealized_conversion_cast %19 : i32 to ui32
      %25 = builtin.unrealized_conversion_cast %20 : i32 to ui32
      %26 = builtin.unrealized_conversion_cast %21 : i32 to ui32
      %27 = builtin.unrealized_conversion_cast %22 : i32 to ui32
      "comp.add_tiles"(%23, %24, %25, %26, %27) : (ui32, ui32, ui32, ui32, ui32) -> ()
      "comp.tile_regs_commit"() : () -> ()
      "comp.tile_regs_wait"() : () -> ()
      %28 = arith.constant 0 : i32
      %29 = memref.load %cb_out[] : memref<i32>
      %30 = arith.constant 0 : i32
      %31 = builtin.unrealized_conversion_cast %28 : i32 to ui32
      %32 = builtin.unrealized_conversion_cast %29 : i32 to ui32
      %33 = builtin.unrealized_conversion_cast %30 : i32 to ui32
      "comp.pack_tile"(%31, %32, %33) <{"out_of_order_output" = false}> : (ui32, ui32, ui32) -> ()
      "comp.tile_regs_release"() : () -> ()
      %34 = memref.load %cb0[] : memref<i32>
      %35 = arith.constant 1 : i32
      "cb.pop_front"(%34, %35) : (i32, i32) -> ()
      %36 = memref.load %cb1[] : memref<i32>
      %37 = arith.constant 1 : i32
      "cb.pop_front"(%36, %37) : (i32, i32) -> ()
      %38 = memref.load %cb0[] : memref<i32>
      %39 = arith.constant 1 : i32
      "cb.push_back"(%38, %39) : (i32, i32) -> ()
      func.return
    }
  }
}

// CHECK:      #include <cstdint>
// CHECK-NEXT: #include "compute_kernel_api/tile_move_copy.h"
// CHECK-NEXT: #include "compute_kernel_api/eltwise_binary.h" 
// CHECK:      void MAIN {
// CHECK-NEXT:     std::int32_t cb0 = 0;
// CHECK-NEXT:     std::int32_t cb1 = 1;
// CHECK-NEXT:     std::int32_t cb_out = 16;
// CHECK-NEXT:     binary_op_init_common(static_cast<std::uint32_t>(cb0), static_cast<std::uint32_t>(cb1), static_cast<std::uint32_t>(cb_out));
// CHECK-NEXT:     add_tiles_init(static_cast<std::uint32_t>(cb0), static_cast<std::uint32_t>(cb1), false);
// CHECK-NEXT:     cb_wait_front(cb0, 1);
// CHECK-NEXT:     cb_wait_front(cb1, 1);
// CHECK-NEXT:     tile_regs_acquire();
// CHECK-NEXT:     add_tiles(static_cast<std::uint32_t>(cb0), static_cast<std::uint32_t>(cb1), static_cast<std::uint32_t>(0), static_cast<std::uint32_t>(0), static_cast<std::uint32_t>(0));
// CHECK-NEXT:     tile_regs_commit();
// CHECK-NEXT:     tile_regs_wait();
// CHECK-NEXT:     pack_tile<false>(static_cast<std::uint32_t>(0), static_cast<std::uint32_t>(cb_out), static_cast<std::uint32_t>(0));
// CHECK-NEXT:     tile_regs_release();
// CHECK-NEXT:     cb_pop_front(cb0, 1);
// CHECK-NEXT:     cb_pop_front(cb1, 1);
// CHECK-NEXT:     cb_push_back(cb0, 1);
// CHECK-NEXT: }