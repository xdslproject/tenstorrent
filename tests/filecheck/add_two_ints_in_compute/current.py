# RUN : python3.11 %s

from tenstorrent.frontend import tt


@tt.compute
def add_two_ints_vectorised():
    cb0 = 0
    cb1 = 1
    cb_out = 16

    tt.binary_op_init_common(cb0, cb1, cb_out)
    tt.add_init(cb0, cb1, False)

    # wait for a single block of tiles in each input CB
    tt.cb_wait_front(cb0, 1)
    tt.cb_wait_front(cb1, 1)

    # acquire 8 tile registers
    tt.regs_acquire()

    # add the first tiles in cb0 and cb1, storing the result tile
    tt.add(cb0, cb1, 0, 0, 0)

    # commit the result, signals the packer
    tt.regs_commit()

    tt.regs_wait()  # the packer core will wait here
    tt.pack_tile(False, 0, cb_out, 0)
    tt.regs_release()  # and release here

    tt.cb_pop_front(cb0, 1)
    tt.cb_pop_front(cb1, 1)
    tt.cb_push_back(cb0, 1)


# CHECK:      builtin.module {
# CHECK-NEXT:   builtin.module attributes  {kernel_type = compute} {
# CHECK-NEXT:     func.func @add_two_ints_vectorised() {
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
# CHECK-NEXT:       %6 = builtin.unrealized_conversion_cast %3 : i32 to ui32
# CHECK-NEXT:       %7 = builtin.unrealized_conversion_cast %4 : i32 to ui32
# CHECK-NEXT:       %8 = builtin.unrealized_conversion_cast %5 : i32 to ui32
# CHECK-NEXT:       comp.binary_op_init_common(%6, %7, %8) : (ui32, ui32, ui32) -> ()
# CHECK-NEXT:       %9 = memref.load %cb0[] : memref<i32>
# CHECK-NEXT:       %10 = memref.load %cb1[] : memref<i32>
# CHECK-NEXT:       %11 = arith.constant false
# CHECK-NEXT:       %12 = builtin.unrealized_conversion_cast %9 : i32 to ui32
# CHECK-NEXT:       %13 = builtin.unrealized_conversion_cast %10 : i32 to ui32
# CHECK-NEXT:       comp.add_tiles_init(%12, %13, %11) : (ui32, ui32, i1) -> ()
# CHECK-NEXT:       %14 = memref.load %cb0[] : memref<i32>
# CHECK-NEXT:       %15 = arith.constant 1 : i32
# CHECK-NEXT:       cb.wait_front(%14, %15) : (i32, i32) -> ()
# CHECK-NEXT:       %16 = memref.load %cb1[] : memref<i32>
# CHECK-NEXT:       %17 = arith.constant 1 : i32
# CHECK-NEXT:       cb.wait_front(%16, %17) : (i32, i32) -> ()
# CHECK-NEXT:       comp.tile_regs_acquire() : () -> ()
# CHECK-NEXT:       %18 = memref.load %cb0[] : memref<i32>
# CHECK-NEXT:       %19 = memref.load %cb1[] : memref<i32>
# CHECK-NEXT:       %20 = arith.constant 0 : i32
# CHECK-NEXT:       %21 = arith.constant 0 : i32
# CHECK-NEXT:       %22 = arith.constant 0 : i32
# CHECK-NEXT:       %23 = builtin.unrealized_conversion_cast %18 : i32 to ui32
# CHECK-NEXT:       %24 = builtin.unrealized_conversion_cast %19 : i32 to ui32
# CHECK-NEXT:       %25 = builtin.unrealized_conversion_cast %20 : i32 to ui32
# CHECK-NEXT:       %26 = builtin.unrealized_conversion_cast %21 : i32 to ui32
# CHECK-NEXT:       %27 = builtin.unrealized_conversion_cast %22 : i32 to ui32
# CHECK-NEXT:       comp.add_tiles(%23, %24, %25, %26, %27) : (ui32, ui32, ui32, ui32, ui32) -> ()
# CHECK-NEXT:       comp.tile_regs_commit() : () -> ()
# CHECK-NEXT:       comp.tile_regs_wait() : () -> ()
# CHECK-NEXT:       %28 = arith.constant 0 : i32
# CHECK-NEXT:       %29 = memref.load %cb_out[] : memref<i32>
# CHECK-NEXT:       %30 = arith.constant 0 : i32
# CHECK-NEXT:       %31 = builtin.unrealized_conversion_cast %28 : i32 to ui32
# CHECK-NEXT:       %32 = builtin.unrealized_conversion_cast %29 : i32 to ui32
# CHECK-NEXT:       %33 = builtin.unrealized_conversion_cast %30 : i32 to ui32
# CHECK-NEXT:       comp.pack_tile(%31, %32, %33) <{out_of_order_output = false}> : (ui32, ui32, ui32) -> ()
# CHECK-NEXT:       comp.tile_regs_release() : () -> ()
# CHECK-NEXT:       %34 = memref.load %cb0[] : memref<i32>
# CHECK-NEXT:       %35 = arith.constant 1 : i32
# CHECK-NEXT:       cb.pop_front(%34, %35) : (i32, i32) -> ()
# CHECK-NEXT:       %36 = memref.load %cb1[] : memref<i32>
# CHECK-NEXT:       %37 = arith.constant 1 : i32
# CHECK-NEXT:       cb.pop_front(%36, %37) : (i32, i32) -> ()
# CHECK-NEXT:       %38 = memref.load %cb0[] : memref<i32>
# CHECK-NEXT:       %39 = arith.constant 1 : i32
# CHECK-NEXT:       cb.push_back(%38, %39) : (i32, i32) -> ()
# CHECK-NEXT:       func.return
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT: }
