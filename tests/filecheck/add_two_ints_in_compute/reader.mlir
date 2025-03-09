// RUN: python3.13 tenstorrent/tools/tt-opt %s -t tt-metalium | filecheck %s

builtin.module {
  builtin.module attributes {kernel_type = "data_in"} {
    func.func @kernel_main(%0 : ui32, %1 : ui32, %2 : ui32, %3 : ui32) {
      %4 = "dm.get_noc_addr_from_bank_id"(%2, %0) <{dram = true}> : (ui32, ui32) -> ui64
      %src0_noc_addr = memref.alloc() : memref<ui64>
      memref.store %4, %src0_noc_addr[] : memref<ui64>
      %5 = "dm.get_noc_addr_from_bank_id"(%3, %1) <{dram = true}> : (ui32, ui32) -> ui64
      %src1_noc_addr = memref.alloc() : memref<ui64>
      memref.store %5, %src1_noc_addr[] : memref<ui64>
      %6 = arith.constant 0 : i32
      %cb_in0 = memref.alloc() : memref<i32>
      memref.store %6, %cb_in0[] : memref<i32>
      %7 = arith.constant 1 : i32
      %cb_in1 = memref.alloc() : memref<i32>
      memref.store %7, %cb_in1[] : memref<i32>
      %8 = memref.load %cb_in0[] : memref<i32>
      %9 = "cb.get_write_pointer"(%8) : (i32) -> ui32
      %l1_write_addr_in0 = memref.alloc() : memref<ui32>
      memref.store %9, %l1_write_addr_in0[] : memref<ui32>
      %10 = memref.load %cb_in1[] : memref<i32>
      %11 = "cb.get_write_pointer"(%10) : (i32) -> ui32
      %l1_write_addr_in1 = memref.alloc() : memref<ui32>
      memref.store %11, %l1_write_addr_in1[] : memref<ui32>
      %12 = arith.constant 4 : i32
      %int_size_bytes = memref.alloc() : memref<i32>
      memref.store %12, %int_size_bytes[] : memref<i32>
      %13 = memref.load %int_size_bytes[] : memref<i32>
      %14 = arith.constant 100 : i32
      %15 = arith.muli %13, %14 : i32
      %block_size = memref.alloc() : memref<i32>
      memref.store %15, %block_size[] : memref<i32>
      %16 = memref.load %cb_in0[] : memref<i32>
      %17 = arith.constant 1 : i32
      "cb.reserve_back"(%16, %17) : (i32, i32) -> ()
      %18 = memref.load %src0_noc_addr[] : memref<ui64>
      %19 = memref.load %l1_write_addr_in0[] : memref<ui32>
      %20 = memref.load %block_size[] : memref<i32>
      %21 = builtin.unrealized_conversion_cast %20 : i32 to ui32
      "dm.noc_async_read"(%18, %19, %21) : (ui64, ui32, ui32) -> ()
      "dm.noc_async_read_barrier"() : () -> ()
      %22 = memref.load %cb_in0[] : memref<i32>
      %23 = arith.constant 1 : i32
      "cb.push_back"(%22, %23) : (i32, i32) -> ()
      %24 = memref.load %cb_in1[] : memref<i32>
      %25 = arith.constant 1 : i32
      "cb.reserve_back"(%24, %25) : (i32, i32) -> ()
      %26 = memref.load %src1_noc_addr[] : memref<ui64>
      %27 = memref.load %l1_write_addr_in1[] : memref<ui32>
      %28 = memref.load %block_size[] : memref<i32>
      %29 = builtin.unrealized_conversion_cast %28 : i32 to ui32
      "dm.noc_async_read"(%26, %27, %29) : (ui64, ui32, ui32) -> ()
      "dm.noc_async_read_barrier"() : () -> ()
      %30 = memref.load %cb_in1[] : memref<i32>
      %31 = arith.constant 1 : i32
      "cb.push_back"(%30, %31) : (i32, i32) -> ()
      func.return
    }
  }
}

// CHECK:      #include <stdint.h>
// CHECK-NEXT: #include "dataflow_api.h"
// CHECK:      void kernel_main() {
// CHECK-NEXT:     uint32_t fn_arg_0 = get_arg_val<uint32_t>(0);
// CHECK-NEXT:     uint32_t fn_arg_1 = get_arg_val<uint32_t>(1);
// CHECK-NEXT:     uint32_t fn_arg_2 = get_arg_val<uint32_t>(2);
// CHECK-NEXT:     uint32_t fn_arg_3 = get_arg_val<uint32_t>(3);
// CHECK-NEXT:     uint64_t src0_noc_addr = get_noc_addr_from_bank_id<true>(fn_arg_2, fn_arg_0);
// CHECK-NEXT:     uint64_t src1_noc_addr = get_noc_addr_from_bank_id<true>(fn_arg_3, fn_arg_1);
// CHECK-NEXT:     std::int32_t cb_in0 = 0;
// CHECK-NEXT:     std::int32_t cb_in1 = 1;
// CHECK-NEXT:     uint32_t l1_write_addr_in0 = get_write_ptr(cb_in0);
// CHECK-NEXT:     uint32_t l1_write_addr_in1 = get_write_ptr(cb_in1);
// CHECK-NEXT:     std::int32_t int_size_bytes = 4;
// CHECK-NEXT:     std::int32_t block_size = int_size_bytes * 100;
// CHECK-NEXT:     cb_reserve_back(cb_in0, 1);
// CHECK-NEXT:     noc_async_read(src0_noc_addr, l1_write_addr_in0, static_cast<std::uint32_t>(block_size));
// CHECK-NEXT:     noc_async_read_barrier();
// CHECK-NEXT:     cb_push_back(cb_in0, 1);
// CHECK-NEXT:     cb_reserve_back(cb_in1, 1);
// CHECK-NEXT:     noc_async_read(src1_noc_addr, l1_write_addr_in1, static_cast<std::uint32_t>(block_size));
// CHECK-NEXT:     noc_async_read_barrier();
// CHECK-NEXT:     cb_push_back(cb_in1, 1);
// CHECK-NEXT: }