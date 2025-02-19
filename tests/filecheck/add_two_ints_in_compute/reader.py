# RUN: python3.12 %s | filecheck %s
from tenstorrent.frontend import tt


@tt.data_in
def kernel_main(src0_dram: uint, src1_dram: uint, src0_dram_id: uint, src1_dram_id: uint):
    src0_noc_addr = tt.get_noc_addr_from_bank_id(True, src0_dram_id, src0_dram)
    src1_noc_addr = tt.get_noc_addr_from_bank_id(True, src1_dram_id, src1_dram)

    cb_in0 = 0
    cb_in1 = 1

    l1_write_addr_in0 = tt.cb_get_write_ptr(cb_in0)
    l1_write_addr_in1 = tt.cb_get_write_ptr(cb_in1)

    # read 1 block of memory from DRAM into circular buffers, then push to compute core unpacker
    int_size_bytes = 4
    block_size = int_size_bytes * 100

    tt.cb_reserve_back(cb_in0, 1)
    tt.noc_async_read(src0_noc_addr, l1_write_addr_in0, block_size)
    tt.noc_async_read_barrier()
    tt.cb_push_back(cb_in0, 1)

    tt.cb_reserve_back(cb_in1, 1)
    tt.noc_async_read(src1_noc_addr, l1_write_addr_in1, block_size)
    tt.noc_async_read_barrier()
    tt.cb_push_back(cb_in1, 1)

# CHECK:      builtin.module {
# CHECK-NEXT:   builtin.module attributes {kernel_type = "data_in"} {
# CHECK-NEXT:     func.func @kernel_main(%0 : ui32, %1 : ui32, %2 : ui32, %3 : ui32) {
# CHECK-NEXT:       %4 = "dm.get_noc_addr_from_bank_id"(%2, %0) <{dram = true}> : (ui32, ui32) -> ui64
# CHECK-NEXT:       %src0_noc_addr = memref.alloc() : memref<ui64>
# CHECK-NEXT:       memref.store %4, %src0_noc_addr[] : memref<ui64>
# CHECK-NEXT:       %5 = "dm.get_noc_addr_from_bank_id"(%3, %1) <{dram = true}> : (ui32, ui32) -> ui64
# CHECK-NEXT:       %src1_noc_addr = memref.alloc() : memref<ui64>
# CHECK-NEXT:       memref.store %5, %src1_noc_addr[] : memref<ui64>
# CHECK-NEXT:       %6 = arith.constant 0 : i32
# CHECK-NEXT:       %cb_in0 = memref.alloc() : memref<i32>
# CHECK-NEXT:       memref.store %6, %cb_in0[] : memref<i32>
# CHECK-NEXT:       %7 = arith.constant 1 : i32
# CHECK-NEXT:       %cb_in1 = memref.alloc() : memref<i32>
# CHECK-NEXT:       memref.store %7, %cb_in1[] : memref<i32>
# CHECK-NEXT:       %8 = memref.load %cb_in0[] : memref<i32>
# CHECK-NEXT:       %9 = "cb.get_write_pointer"(%8) : (i32) -> ui32
# CHECK-NEXT:       %l1_write_addr_in0 = memref.alloc() : memref<ui32>
# CHECK-NEXT:       memref.store %9, %l1_write_addr_in0[] : memref<ui32>
# CHECK-NEXT:       %10 = memref.load %cb_in1[] : memref<i32>
# CHECK-NEXT:       %11 = "cb.get_write_pointer"(%10) : (i32) -> ui32
# CHECK-NEXT:       %l1_write_addr_in1 = memref.alloc() : memref<ui32>
# CHECK-NEXT:       memref.store %11, %l1_write_addr_in1[] : memref<ui32>
# CHECK-NEXT:       %12 = arith.constant 4 : i32
# CHECK-NEXT:       %int_size_bytes = memref.alloc() : memref<i32>
# CHECK-NEXT:       memref.store %12, %int_size_bytes[] : memref<i32>
# CHECK-NEXT:       %13 = memref.load %int_size_bytes[] : memref<i32>
# CHECK-NEXT:       %14 = arith.constant 100 : i32
# CHECK-NEXT:       %15 = arith.muli %13, %14 : i32
# CHECK-NEXT:       %block_size = memref.alloc() : memref<i32>
# CHECK-NEXT:       memref.store %15, %block_size[] : memref<i32>
# CHECK-NEXT:       %16 = memref.load %cb_in0[] : memref<i32>
# CHECK-NEXT:       %17 = arith.constant 1 : i32
# CHECK-NEXT:       "cb.reserve_back"(%16, %17) : (i32, i32) -> ()
# CHECK-NEXT:       %18 = memref.load %src0_noc_addr[] : memref<ui64>
# CHECK-NEXT:       %19 = memref.load %l1_write_addr_in0[] : memref<ui32>
# CHECK-NEXT:       %20 = memref.load %block_size[] : memref<i32>
# CHECK-NEXT:       %21 = builtin.unrealized_conversion_cast %20 : i32 to ui32
# CHECK-NEXT:       "dm.noc_async_read"(%18, %19, %21) : (ui64, ui32, ui32) -> ()
# CHECK-NEXT:       "dm.noc_async_read_barrier"() : () -> ()
# CHECK-NEXT:       %22 = memref.load %cb_in0[] : memref<i32>
# CHECK-NEXT:       %23 = arith.constant 1 : i32
# CHECK-NEXT:       "cb.push_back"(%22, %23) : (i32, i32) -> ()
# CHECK-NEXT:       %24 = memref.load %cb_in1[] : memref<i32>
# CHECK-NEXT:       %25 = arith.constant 1 : i32
# CHECK-NEXT:       "cb.reserve_back"(%24, %25) : (i32, i32) -> ()
# CHECK-NEXT:       %26 = memref.load %src1_noc_addr[] : memref<ui64>
# CHECK-NEXT:       %27 = memref.load %l1_write_addr_in1[] : memref<ui32>
# CHECK-NEXT:       %28 = memref.load %block_size[] : memref<i32>
# CHECK-NEXT:       %29 = builtin.unrealized_conversion_cast %28 : i32 to ui32
# CHECK-NEXT:       "dm.noc_async_read"(%26, %27, %29) : (ui64, ui32, ui32) -> ()
# CHECK-NEXT:       "dm.noc_async_read_barrier"() : () -> ()
# CHECK-NEXT:       %30 = memref.load %cb_in1[] : memref<i32>
# CHECK-NEXT:       %31 = arith.constant 1 : i32
# CHECK-NEXT:       "cb.push_back"(%30, %31) : (i32, i32) -> ()
# CHECK-NEXT:       func.return
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT: }