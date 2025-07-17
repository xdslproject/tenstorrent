# RUN: python3.13 %s | filecheck %s
from tests import run_from_examples

run_from_examples(__file__)


# CHECK:      builtin.module {
# CHECK-NEXT:   builtin.module attributes {kernel_type = "data_in"} {
# CHECK-NEXT:     func.func @kernel_main(%0 : i32, %1 : i32, %2 : i32, %3 : i32) {
# CHECK-NEXT:       %4 = "dm.get_noc_addr_from_bank_id"(%2, %0) <{dram = true}> : (i32, i32) -> i64
# CHECK-NEXT:       %src0_noc_addr = memref.alloc() : memref<i64>
# CHECK-NEXT:       memref.store %4, %src0_noc_addr[] : memref<i64>
# CHECK-NEXT:       %5 = "dm.get_noc_addr_from_bank_id"(%3, %1) <{dram = true}> : (i32, i32) -> i64
# CHECK-NEXT:       %src1_noc_addr = memref.alloc() : memref<i64>
# CHECK-NEXT:       memref.store %5, %src1_noc_addr[] : memref<i64>
# CHECK-NEXT:       %6 = arith.constant 0 : i32
# CHECK-NEXT:       %cb_in0 = memref.alloc() : memref<i32>
# CHECK-NEXT:       memref.store %6, %cb_in0[] : memref<i32>
# CHECK-NEXT:       %7 = arith.constant 1 : i32
# CHECK-NEXT:       %cb_in1 = memref.alloc() : memref<i32>
# CHECK-NEXT:       memref.store %7, %cb_in1[] : memref<i32>
# CHECK-NEXT:       %8 = memref.load %cb_in0[] : memref<i32>
# CHECK-NEXT:       %9 = "cb.get_write_pointer"(%8) : (i32) -> i32
# CHECK-NEXT:       %l1_write_addr_in0 = memref.alloc() : memref<i32>
# CHECK-NEXT:       memref.store %9, %l1_write_addr_in0[] : memref<i32>
# CHECK-NEXT:       %10 = memref.load %cb_in1[] : memref<i32>
# CHECK-NEXT:       %11 = "cb.get_write_pointer"(%10) : (i32) -> i32
# CHECK-NEXT:       %l1_write_addr_in1 = memref.alloc() : memref<i32>
# CHECK-NEXT:       memref.store %11, %l1_write_addr_in1[] : memref<i32>
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
# CHECK-NEXT:       %18 = memref.load %src0_noc_addr[] : memref<i64>
# CHECK-NEXT:       %19 = memref.load %l1_write_addr_in0[] : memref<i32>
# CHECK-NEXT:       %20 = memref.load %block_size[] : memref<i32>
# CHECK-NEXT:       "dm.noc_async_read"(%18, %19, %20) : (i64, i32, i32) -> ()
# CHECK-NEXT:       "dm.noc_async_read_barrier"() : () -> ()
# CHECK-NEXT:       %21 = memref.load %cb_in0[] : memref<i32>
# CHECK-NEXT:       %22 = arith.constant 1 : i32
# CHECK-NEXT:       "cb.push_back"(%21, %22) : (i32, i32) -> ()
# CHECK-NEXT:       %23 = memref.load %cb_in1[] : memref<i32>
# CHECK-NEXT:       %24 = arith.constant 1 : i32
# CHECK-NEXT:       "cb.reserve_back"(%23, %24) : (i32, i32) -> ()
# CHECK-NEXT:       %25 = memref.load %src1_noc_addr[] : memref<i64>
# CHECK-NEXT:       %26 = memref.load %l1_write_addr_in1[] : memref<i32>
# CHECK-NEXT:       %27 = memref.load %block_size[] : memref<i32>
# CHECK-NEXT:       "dm.noc_async_read"(%25, %26, %27) : (i64, i32, i32) -> ()
# CHECK-NEXT:       "dm.noc_async_read_barrier"() : () -> ()
# CHECK-NEXT:       %28 = memref.load %cb_in1[] : memref<i32>
# CHECK-NEXT:       %29 = arith.constant 1 : i32
# CHECK-NEXT:       "cb.push_back"(%28, %29) : (i32, i32) -> ()
# CHECK-NEXT:       func.return
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT: }
