# RUN: python3.12 %s | filecheck %s
from tests import run_from_examples

run_from_examples(__file__)

# CHECK:      builtin.module {
# CHECK-NEXT:   builtin.module attributes {kernel_type = "data_out"} {
# CHECK-NEXT:     func.func @writer_kernel(%0 : ui32, %1 : ui32) {
# CHECK-NEXT:       %2 = "dm.get_noc_addr_from_bank_id"(%1, %0) <{dram = true}> : (ui32, ui32) -> ui64
# CHECK-NEXT:       %dst_noc_addr = memref.alloc() : memref<ui64>
# CHECK-NEXT:       memref.store %2, %dst_noc_addr[] : memref<ui64>
# CHECK-NEXT:       %3 = arith.constant 16 : i32
# CHECK-NEXT:       %cb_id = memref.alloc() : memref<i32>
# CHECK-NEXT:       memref.store %3, %cb_id[] : memref<i32>
# CHECK-NEXT:       %4 = memref.load %cb_id[] : memref<i32>
# CHECK-NEXT:       %5 = "cb.get_read_pointer"(%4) : (i32) -> ui32
# CHECK-NEXT:       %l1_read_addr = memref.alloc() : memref<ui32>
# CHECK-NEXT:       memref.store %5, %l1_read_addr[] : memref<ui32>
# CHECK-NEXT:       %6 = arith.constant 4 : i32
# CHECK-NEXT:       %int_size_bytes = memref.alloc() : memref<i32>
# CHECK-NEXT:       memref.store %6, %int_size_bytes[] : memref<i32>
# CHECK-NEXT:       %7 = memref.load %int_size_bytes[] : memref<i32>
# CHECK-NEXT:       %8 = arith.constant 100 : i32
# CHECK-NEXT:       %9 = arith.muli %7, %8 : i32
# CHECK-NEXT:       %block_size = memref.alloc() : memref<i32>
# CHECK-NEXT:       memref.store %9, %block_size[] : memref<i32>
# CHECK-NEXT:       %10 = memref.load %cb_id[] : memref<i32>
# CHECK-NEXT:       %11 = arith.constant 1 : i32
# CHECK-NEXT:       "cb.wait_front"(%10, %11) : (i32, i32) -> ()
# CHECK-NEXT:       %12 = memref.load %l1_read_addr[] : memref<ui32>
# CHECK-NEXT:       %13 = memref.load %dst_noc_addr[] : memref<ui64>
# CHECK-NEXT:       %14 = memref.load %block_size[] : memref<i32>
# CHECK-NEXT:       %15 = builtin.unrealized_conversion_cast %14 : i32 to ui32
# CHECK-NEXT:       "dm.noc_async_write"(%12, %13, %15) : (ui32, ui64, ui32) -> ()
# CHECK-NEXT:       "dm.noc_async_read_barrier"() : () -> ()
# CHECK-NEXT:       %16 = memref.load %cb_id[] : memref<i32>
# CHECK-NEXT:       %17 = arith.constant 1 : i32
# CHECK-NEXT:       "cb.pop_front"(%16, %17) : (i32, i32) -> ()
# CHECK-NEXT:       func.return
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT: }
