// RUN: python3.13 tenstorrent/tools/tt-opt %s -t tt-metalium | filecheck %s

builtin.module {
  builtin.module attributes {kernel_type = "data_out"} {
    func.func @kernel_main(%0 : ui32, %1 : ui32) {
      %2 = "dm.get_noc_addr_from_bank_id"(%1, %0) <{dram = true}> : (ui32, ui32) -> ui64
      %dst_noc_addr = memref.alloc() : memref<ui64>
      memref.store %2, %dst_noc_addr[] : memref<ui64>
      %3 = arith.constant 16 : i32
      %cb_id = memref.alloc() : memref<i32>
      memref.store %3, %cb_id[] : memref<i32>
      %4 = memref.load %cb_id[] : memref<i32>
      %5 = "cb.get_read_pointer"(%4) : (i32) -> ui32
      %l1_read_addr = memref.alloc() : memref<ui32>
      memref.store %5, %l1_read_addr[] : memref<ui32>
      %6 = arith.constant 4 : i32
      %int_size_bytes = memref.alloc() : memref<i32>
      memref.store %6, %int_size_bytes[] : memref<i32>
      %7 = memref.load %int_size_bytes[] : memref<i32>
      %8 = arith.constant 100 : i32
      %9 = arith.muli %7, %8 : i32
      %block_size = memref.alloc() : memref<i32>
      memref.store %9, %block_size[] : memref<i32>
      %10 = memref.load %cb_id[] : memref<i32>
      %11 = arith.constant 1 : i32
      "cb.wait_front"(%10, %11) : (i32, i32) -> ()
      %12 = memref.load %l1_read_addr[] : memref<ui32>
      %13 = memref.load %dst_noc_addr[] : memref<ui64>
      %14 = memref.load %block_size[] : memref<i32>
      %15 = builtin.unrealized_conversion_cast %14 : i32 to ui32
      "dm.noc_async_write"(%12, %13, %15) : (ui32, ui64, ui32) -> ()
      "dm.noc_async_write_barrier"() : () -> ()
      %16 = memref.load %cb_id[] : memref<i32>
      %17 = arith.constant 1 : i32
      "cb.pop_front"(%16, %17) : (i32, i32) -> ()
      func.return
    }
  }
}

// CHECK:      #include "dataflow_api.h"
// CHECK:      void kernel_main() {
// CHECK-NEXT:     uint32_t fn_arg_0 = get_arg_val<uint32_t>(0);
// CHECK-NEXT:     uint32_t fn_arg_1 = get_arg_val<uint32_t>(1);
// CHECK-NEXT:     uint64_t dst_noc_addr = get_noc_addr_from_bank_id<true>(fn_arg_1, fn_arg_0);
// CHECK-NEXT:     std::int32_t cb_id = 16;
// CHECK-NEXT:     uint32_t l1_read_addr = get_read_ptr(cb_id);
// CHECK-NEXT:     std::int32_t int_size_bytes = 4;
// CHECK-NEXT:     std::int32_t block_size = int_size_bytes * 100;
// CHECK-NEXT:     cb_wait_front(cb_id, 1);
// CHECK-NEXT:     noc_async_write(l1_read_addr, dst_noc_addr, static_cast<std::uint32_t>(block_size));
// CHECK-NEXT:     noc_async_write_barrier();
// CHECK-NEXT:     cb_pop_front(cb_id, 1);
// CHECK-NEXT: }
