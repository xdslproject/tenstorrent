// RUN: python3.13 tenstorrent/tools/tt-opt %s -t tt-metalium | filecheck %s

builtin.module {
  builtin.module attributes {kernel_type = "data_in"} {
    func.func @kernel_main(%0 : i32, %1 : i32, %2 : i32, %3 : i32) {
      %4 = "dm.get_noc_addr_from_bank_id"(%2, %0) <{dram = true}> : (i32, i32) -> i64
      %src0_noc_addr = memref.alloc() : memref<i64>
      memref.store %4, %src0_noc_addr[] : memref<i64>
      %5 = "dm.get_noc_addr_from_bank_id"(%3, %1) <{dram = true}> : (i32, i32) -> i64
      %src1_noc_addr = memref.alloc() : memref<i64>
      memref.store %5, %src1_noc_addr[] : memref<i64>
      %6 = arith.constant 0 : i32
      %cb_in0 = memref.alloc() : memref<i32>
      memref.store %6, %cb_in0[] : memref<i32>
      %7 = arith.constant 1 : i32
      %cb_in1 = memref.alloc() : memref<i32>
      memref.store %7, %cb_in1[] : memref<i32>
      %8 = memref.load %cb_in0[] : memref<i32>
      %9 = "cb.get_write_pointer"(%8) : (i32) -> i32
      %l1_write_addr_in0 = memref.alloc() : memref<i32>
      memref.store %9, %l1_write_addr_in0[] : memref<i32>
      %10 = memref.load %cb_in1[] : memref<i32>
      %11 = "cb.get_write_pointer"(%10) : (i32) -> i32
      %l1_write_addr_in1 = memref.alloc() : memref<i32>
      memref.store %11, %l1_write_addr_in1[] : memref<i32>
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
      %18 = memref.load %src0_noc_addr[] : memref<i64>
      %19 = memref.load %l1_write_addr_in0[] : memref<i32>
      %20 = memref.load %block_size[] : memref<i32>
      "dm.noc_async_read"(%18, %19, %20) : (i64, i32, i32) -> ()
      "dm.noc_async_read_barrier"() : () -> ()
      %21 = memref.load %cb_in0[] : memref<i32>
      %22 = arith.constant 1 : i32
      "cb.push_back"(%21, %22) : (i32, i32) -> ()
      %23 = memref.load %cb_in1[] : memref<i32>
      %24 = arith.constant 1 : i32
      "cb.reserve_back"(%23, %24) : (i32, i32) -> ()
      %25 = memref.load %src1_noc_addr[] : memref<i64>
      %26 = memref.load %l1_write_addr_in1[] : memref<i32>
      %27 = memref.load %block_size[] : memref<i32>
      "dm.noc_async_read"(%25, %26, %27) : (i64, i32, i32) -> ()
      "dm.noc_async_read_barrier"() : () -> ()
      %28 = memref.load %cb_in1[] : memref<i32>
      %29 = arith.constant 1 : i32
      "cb.push_back"(%28, %29) : (i32, i32) -> ()
      func.return
    }
  }
}

// CHECK:     void kernel_main() {
// CHECK-NEXT:     std::int32_t fn_arg_0 = get_arg_val<std::int32_t>(0);
// CHECK-NEXT:     std::int32_t fn_arg_1 = get_arg_val<std::int32_t>(1);
// CHECK-NEXT:     std::int32_t fn_arg_2 = get_arg_val<std::int32_t>(2);
// CHECK-NEXT:     std::int32_t fn_arg_3 = get_arg_val<std::int32_t>(3);
// CHECK-NEXT:     std::int64_t noc_addr_0 = get_noc_addr_from_bank_id<true>(fn_arg_2, fn_arg_0);
// CHECK-NEXT:     std::int64_t src0_noc_addr = noc_addr_0;
// CHECK-NEXT:     std::int64_t noc_addr_1 = get_noc_addr_from_bank_id<true>(fn_arg_3, fn_arg_1);
// CHECK-NEXT:     std::int64_t src1_noc_addr = noc_addr_1;
// CHECK-NEXT:     std::int32_t cb_in0 = 0;
// CHECK-NEXT:     std::int32_t cb_in1 = 1;
// CHECK-NEXT:     std::int32_t write_ptr_2 = get_write_ptr(cb_in0);
// CHECK-NEXT:     std::int32_t l1_write_addr_in0 = write_ptr_2;
// CHECK-NEXT:     std::int32_t write_ptr_3 = get_write_ptr(cb_in1);
// CHECK-NEXT:     std::int32_t l1_write_addr_in1 = write_ptr_3;
// CHECK-NEXT:     std::int32_t int_size_bytes = 4;
// CHECK-NEXT:     std::int32_t block_size = int_size_bytes * 100;
// CHECK-NEXT:     cb_reserve_back(cb_in0, 1);
// CHECK-NEXT:     noc_async_read(src0_noc_addr, l1_write_addr_in0, block_size);
// CHECK-NEXT:     noc_async_read_barrier();
// CHECK-NEXT:     cb_push_back(cb_in0, 1);
// CHECK-NEXT:     cb_reserve_back(cb_in1, 1);
// CHECK-NEXT:     noc_async_read(src1_noc_addr, l1_write_addr_in1, block_size);
// CHECK-NEXT:     noc_async_read_barrier();
// CHECK-NEXT:     cb_push_back(cb_in1, 1);
// CHECK-NEXT: }