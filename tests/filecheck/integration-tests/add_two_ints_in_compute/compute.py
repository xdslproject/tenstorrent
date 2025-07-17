# RUN: python3.13 %s | filecheck %s
from tests import run_from_examples

run_from_examples(__file__)

# CHECK:      builtin.module {
# CHECK-NEXT:   builtin.module attributes {kernel_type = "compute"} {
# CHECK-NEXT:     func.func @add_two_int_tiles() {
# CHECK-NEXT:       %0 = arith.constant 0 : i32
# CHECK-NEXT:       %cb0 = memref.alloc() : memref<i32>
# CHECK-NEXT:       memref.store %0, %cb0[] : memref<i32>
# CHECK-NEXT:       %1 = arith.constant 1 : i32
# CHECK-NEXT:       %cb1 = memref.alloc() : memref<i32>
# CHECK-NEXT:       memref.store %1, %cb1[] : memref<i32>
# CHECK-NEXT:       %2 = arith.constant 16 : i32
# CHECK-NEXT:       %cb_out = memref.alloc() : memref<i32>
# CHECK-NEXT:       memref.store %2, %cb_out[] : memref<i32>
# CHECK-NEXT:       %3 = memref.load %cb0[] : memref<i32>
# CHECK-NEXT:       %4 = memref.load %cb1[] : memref<i32>
# CHECK-NEXT:       %5 = memref.load %cb_out[] : memref<i32>
# CHECK-NEXT:       "comp.binary_op_init_common"(%3, %4, %5) : (i32, i32, i32) -> ()
# CHECK-NEXT:       %6 = memref.load %cb0[] : memref<i32>
# CHECK-NEXT:       %7 = memref.load %cb1[] : memref<i32>
# CHECK-NEXT:       %8 = arith.constant false
# CHECK-NEXT:       "comp.add_tiles_init"(%6, %7, %8) : (i32, i32, i1) -> ()
# CHECK-NEXT:       %9 = memref.load %cb0[] : memref<i32>
# CHECK-NEXT:       %10 = arith.constant 1 : i32
# CHECK-NEXT:       "cb.wait_front"(%9, %10) : (i32, i32) -> ()
# CHECK-NEXT:       %11 = memref.load %cb1[] : memref<i32>
# CHECK-NEXT:       %12 = arith.constant 1 : i32
# CHECK-NEXT:       "cb.wait_front"(%11, %12) : (i32, i32) -> ()
# CHECK-NEXT:       "comp.tile_regs_acquire"() : () -> ()
# CHECK-NEXT:       %13 = memref.load %cb0[] : memref<i32>
# CHECK-NEXT:       %14 = memref.load %cb1[] : memref<i32>
# CHECK-NEXT:       %15 = arith.constant 0 : i32
# CHECK-NEXT:       %16 = arith.constant 0 : i32
# CHECK-NEXT:       %17 = arith.constant 0 : i32
# CHECK-NEXT:       "comp.add_tiles"(%13, %14, %15, %16, %17) : (i32, i32, i32, i32, i32) -> ()
# CHECK-NEXT:       "comp.tile_regs_commit"() : () -> ()
# CHECK-NEXT:       "comp.tile_regs_wait"() : () -> ()
# CHECK-NEXT:       %18 = arith.constant 0 : i32
# CHECK-NEXT:       %19 = memref.load %cb_out[] : memref<i32>
# CHECK-NEXT:       %20 = arith.constant 0 : i32
# CHECK-NEXT:       "comp.pack_tile"(%18, %19, %20) <{out_of_order_output = false}> : (i32, i32, i32) -> ()
# CHECK-NEXT:       "comp.tile_regs_release"() : () -> ()
# CHECK-NEXT:       %21 = memref.load %cb0[] : memref<i32>
# CHECK-NEXT:       %22 = arith.constant 1 : i32
# CHECK-NEXT:       "cb.pop_front"(%21, %22) : (i32, i32) -> ()
# CHECK-NEXT:       %23 = memref.load %cb1[] : memref<i32>
# CHECK-NEXT:       %24 = arith.constant 1 : i32
# CHECK-NEXT:       "cb.pop_front"(%23, %24) : (i32, i32) -> ()
# CHECK-NEXT:       %25 = memref.load %cb_out[] : memref<i32>
# CHECK-NEXT:       %26 = arith.constant 1 : i32
# CHECK-NEXT:       "cb.push_back"(%25, %26) : (i32, i32) -> ()
# CHECK-NEXT:       func.return
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT: }
