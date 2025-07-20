// RUN: python3.13 tenstorrent/tools/tt-opt %s -p linalg-to-tt | filecheck %s

builtin.module {
  func.func @"example_matmul"() {
    %0 = "memref.alloca"() <{"operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<32x32xf32>
    %1 = "memref.alloca"() <{"operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<32x32xf32>
    %2 = "memref.alloca"() <{"operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<32x32xf32>
    %3 = "memref.alloca"() <{"operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<32x32xf32>
    linalg.matmul ins(%0, %1 : memref<32x32xf32>, memref<32x32xf32>) outs(%3 : memref<32x32xf32>) -> ()
    "memref.copy"(%3, %2) : (memref<32x32xf32>, memref<32x32xf32>) -> ()
    func.return
  }
  func.func @main() {
    "memref.alloca_scope"() ({
      func.call @example_matmul() : () -> ()
      "memref.alloca_scope.return"() : () -> ()
    }) : () -> ()
    func.return
  }
}

// CHECK:      builtin.module {
// CHECK-NEXT:   builtin.module {
// CHECK-NEXT:     func.func @example_matmul() {
// CHECK-NEXT:       %0 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x32xf32>
// CHECK-NEXT:       %1 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x32xf32>
// CHECK-NEXT:       %2 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x32xf32>
// CHECK-NEXT:       %3 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x32xf32>
// CHECK-NEXT:       func.call @host_entry(%0, %1, %3) : (memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) -> ()
// CHECK-NEXT:       "memref.copy"(%3, %2) : (memref<32x32xf32>, memref<32x32xf32>) -> ()
// CHECK-NEXT:       func.return
// CHECK-NEXT:     }
// CHECK-NEXT:     func.func @main() {
// CHECK-NEXT:       "memref.alloca_scope"() ({
// CHECK-NEXT:         func.call @example_matmul() : () -> ()
// CHECK-NEXT:         "memref.alloca_scope.return"() : () -> ()
// CHECK-NEXT:       }) : () -> ()
// CHECK-NEXT:       func.return
// CHECK-NEXT:     }
// CHECK-NEXT:     func.func private @host_entry(memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) -> ()
// CHECK-NEXT:   }
// CHECK-NEXT:   builtin.module attributes {kernel_type = "host", vis = "external"} {
// CHECK-NEXT:     func.func @host_entry(%0 : memref<32x32xf32>, %1 : memref<32x32xf32>, %2 : memref<32x32xf32>) {
// CHECK-NEXT:       %zero = arith.constant 0 : i32
// CHECK-NEXT:       %3 = arith.constant 1 : i32
// CHECK-NEXT:       %4 = arith.constant 0 : i32
// CHECK-NEXT:       %5 = arith.constant 1 : i32
// CHECK-NEXT:       %6 = arith.constant 2 : i32
// CHECK-NEXT:       %prog = "tthost.create_program"() : () -> !tthost.program
// CHECK-NEXT:       %7 = "tthost.create_device"(%zero) : (i32) -> !tthost.device
// CHECK-NEXT:       %8 = "tthost.core"(%zero, %zero) : (i32, i32) -> !tthost.corecoord
// CHECK-NEXT:       %9 = "tthost.get_command_queue"(%7) : (!tthost.device) -> !tthost.command_queue
// CHECK-NEXT:       %size = arith.constant 4096 : i32
// CHECK-NEXT:       %10 = arith.constant 4096 : i32
// CHECK-NEXT:       %11 = "tthost.create_dram_configuration"(%10, %size) : (i32, i32) -> !tthost.dram_buffer_config
// CHECK-NEXT:       %12 = "tthost.create_buffer"(%11) : (!tthost.dram_buffer_config) -> !tthost.buffer
// CHECK-NEXT:       %13 = arith.constant false
// CHECK-NEXT:       "tthost.enqueue_write_buffer"(%9, %12, %0, %13) : (!tthost.command_queue, !tthost.buffer, memref<32x32xf32>, i1) -> ()
// CHECK-NEXT:       %14 = arith.constant 1 : i32
// CHECK-NEXT:       %15 = "tthost.create_cb_configuration"(%14, %size, %4) <{data_type = "float"}> : (i32, i32, i32) -> !tthost.circular_buffer_config
// CHECK-NEXT:       %16 = "tthost.create_circular_buffer"(%prog, %8, %15) : (!tthost.program, !tthost.corecoord, !tthost.circular_buffer_config) -> !tthost.cb_handle
// CHECK-NEXT:       %size_1 = arith.constant 4096 : i32
// CHECK-NEXT:       %17 = arith.constant 4096 : i32
// CHECK-NEXT:       %18 = "tthost.create_dram_configuration"(%17, %size_1) : (i32, i32) -> !tthost.dram_buffer_config
// CHECK-NEXT:       %19 = "tthost.create_buffer"(%18) : (!tthost.dram_buffer_config) -> !tthost.buffer
// CHECK-NEXT:       %20 = arith.constant false
// CHECK-NEXT:       "tthost.enqueue_write_buffer"(%9, %19, %1, %20) : (!tthost.command_queue, !tthost.buffer, memref<32x32xf32>, i1) -> ()
// CHECK-NEXT:       %21 = arith.constant 1 : i32
// CHECK-NEXT:       %22 = "tthost.create_cb_configuration"(%21, %size_1, %5) <{data_type = "float"}> : (i32, i32, i32) -> !tthost.circular_buffer_config
// CHECK-NEXT:       %23 = "tthost.create_circular_buffer"(%prog, %8, %22) : (!tthost.program, !tthost.corecoord, !tthost.circular_buffer_config) -> !tthost.cb_handle
// CHECK-NEXT:       %size_2 = arith.constant 4096 : i32
// CHECK-NEXT:       %24 = arith.constant 4096 : i32
// CHECK-NEXT:       %25 = "tthost.create_dram_configuration"(%24, %size_2) : (i32, i32) -> !tthost.dram_buffer_config
// CHECK-NEXT:       %26 = "tthost.create_buffer"(%25) : (!tthost.dram_buffer_config) -> !tthost.buffer
// CHECK-NEXT:       %27 = arith.constant false
// CHECK-NEXT:       "tthost.enqueue_write_buffer"(%9, %26, %2, %27) : (!tthost.command_queue, !tthost.buffer, memref<32x32xf32>, i1) -> ()
// CHECK-NEXT:       %28 = arith.constant 1 : i32
// CHECK-NEXT:       %29 = "tthost.create_cb_configuration"(%28, %size_2, %6) <{data_type = "float"}> : (i32, i32, i32) -> !tthost.circular_buffer_config
// CHECK-NEXT:       %30 = "tthost.create_circular_buffer"(%prog, %8, %29) : (!tthost.program, !tthost.corecoord, !tthost.circular_buffer_config) -> !tthost.cb_handle
// CHECK-NEXT:       %dram_addr = "tthost.get_memory_address"(%12) : (!tthost.buffer) -> index
// CHECK-NEXT:       %dram_addr_1 = "tthost.get_memory_address"(%19) : (!tthost.buffer) -> index
// CHECK-NEXT:       %dram_addr_2 = "tthost.get_memory_address"(%26) : (!tthost.buffer) -> index
// CHECK-NEXT:       %reader_kernel = "tthost.create_kernel"(%prog, %8) <{kernel_name = "reader.cpp", riscv_core = #tthost.riscv_core<datamovement_0>, noc_id = #builtin.int<0>}> : (!tthost.program, !tthost.corecoord) -> !tthost.kernel
// CHECK-NEXT:       %writer_kernel = "tthost.create_kernel"(%prog, %8) <{kernel_name = "writer.cpp", riscv_core = #tthost.riscv_core<datamovement_1>, noc_id = #builtin.int<1>}> : (!tthost.program, !tthost.corecoord) -> !tthost.kernel
// CHECK-NEXT:       %compute_kernel = "tthost.create_compute_kernel"(%prog, %8) <{kernel_name = "compute.cpp", riscv_core = #tthost.riscv_core<compute>, math_fidelity = #tthost.math_fidelity<LoFi>, fp32_dest_acc_en = false, math_approx_mode = false}> : (!tthost.program, !tthost.corecoord) -> !tthost.kernel
// CHECK-NEXT:       "tthost.set_runtime_args"(%prog, %compute_kernel, %8) {operandSegmentSizes = array<i32: 1, 1, 1, 0>} : (!tthost.program, !tthost.kernel, !tthost.corecoord) -> ()
// CHECK-NEXT:       "tthost.set_runtime_args"(%prog, %reader_kernel, %8, %zero, %zero, %dram_addr, %dram_addr_1, %size, %size_1) {operandSegmentSizes = array<i32: 1, 1, 1, 6>} : (!tthost.program, !tthost.kernel, !tthost.corecoord, i32, i32, index, index, i32, i32) -> ()
// CHECK-NEXT:       "tthost.set_runtime_args"(%prog, %writer_kernel, %8, %zero, %dram_addr_2, %size_2) {operandSegmentSizes = array<i32: 1, 1, 1, 3>} : (!tthost.program, !tthost.kernel, !tthost.corecoord, i32, index, i32) -> ()
// CHECK-NEXT:       %31 = arith.constant false
// CHECK-NEXT:       "tthost.enqueue_program"(%9, %prog, %31) : (!tthost.command_queue, !tthost.program, i1) -> ()
// CHECK-NEXT:       "tthost.finish"(%9) : (!tthost.command_queue) -> ()
// CHECK-NEXT:       "tthost.enqueue_read_buffer"(%9, %26, %2, %31) : (!tthost.command_queue, !tthost.buffer, memref<32x32xf32>, i1) -> ()
// CHECK-NEXT:       "tthost.close_device"(%7) : (!tthost.device) -> ()
// CHECK-NEXT:       func.return
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   builtin.module attributes {kernel_type = "data_in"} {
// CHECK-NEXT:     func.func @kernel_main(%bank_id : i32, %bank_id_1 : i32, %mem_addr : i32, %mem_addr_1 : i32, %size_bytes : i32, %size_bytes_1 : i32) {
// CHECK-NEXT:       %0 = arith.constant 1 : i32
// CHECK-NEXT:       %1 = arith.constant 0 : i8
// CHECK-NEXT:       %2 = arith.constant 0 : i32
// CHECK-NEXT:       %3 = "dm.get_noc_addr_from_bank_id"(%bank_id, %mem_addr, %1) <{dram = true}> : (i32, i32, i8) -> i64
// CHECK-NEXT:       %4 = "cb.get_write_pointer"(%2) : (i32) -> i32
// CHECK-NEXT:       "cb.reserve_back"(%2, %0) : (i32, i32) -> ()
// CHECK-NEXT:       "dm.noc_async_read"(%3, %4, %size_bytes) : (i64, i32, i32) -> ()
// CHECK-NEXT:       "dm.noc_async_read_barrier"() : () -> ()
// CHECK-NEXT:       "cb.push_back"(%2, %0) : (i32, i32) -> ()
// CHECK-NEXT:       %5 = arith.constant 1 : i32
// CHECK-NEXT:       %6 = arith.constant 0 : i8
// CHECK-NEXT:       %7 = arith.constant 1 : i32
// CHECK-NEXT:       %8 = "dm.get_noc_addr_from_bank_id"(%bank_id_1, %mem_addr_1, %6) <{dram = true}> : (i32, i32, i8) -> i64
// CHECK-NEXT:       %9 = "cb.get_write_pointer"(%7) : (i32) -> i32
// CHECK-NEXT:       "cb.reserve_back"(%7, %5) : (i32, i32) -> ()
// CHECK-NEXT:       "dm.noc_async_read"(%8, %9, %size_bytes_1) : (i64, i32, i32) -> ()
// CHECK-NEXT:       "dm.noc_async_read_barrier"() : () -> ()
// CHECK-NEXT:       "cb.push_back"(%7, %5) : (i32, i32) -> ()
// CHECK-NEXT:       func.return
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   builtin.module attributes {kernel_type = "compute"} {
// CHECK-NEXT:     func.func @MAIN() {
// CHECK-NEXT:       %0 = arith.constant 0 : i32
// CHECK-NEXT:       %1 = arith.constant 1 : i32
// CHECK-NEXT:       %2 = arith.constant 0 : i32
// CHECK-NEXT:       %3 = arith.constant 1 : i32
// CHECK-NEXT:       %4 = arith.constant 2 : i32
// CHECK-NEXT:       %5 = arith.constant true
// CHECK-NEXT:       %6 = arith.constant false
// CHECK-NEXT:       "comp.binary_op_init_common"(%2, %3, %4) : (i32, i32, i32) -> ()
// CHECK-NEXT:       "comp.mm_init"(%2, %3, %4, %0) : (i32, i32, i32, i32) -> ()
// CHECK-NEXT:       "cb.wait_front"(%3, %1) : (i32, i32) -> ()
// CHECK-NEXT:       "cb.wait_front"(%2, %1) : (i32, i32) -> ()
// CHECK-NEXT:       "comp.tile_regs_acquire"() : () -> ()
// CHECK-NEXT:       "comp.matmul_tiles"(%2, %3, %0, %0, %0, %0) : (i32, i32, i32, i32, i32, i32) -> ()
// CHECK-NEXT:       "comp.tile_regs_commit"() : () -> ()
// CHECK-NEXT:       "comp.tile_regs_wait"() : () -> ()
// CHECK-NEXT:       "comp.pack_tile"(%0, %4, %0) <{out_of_order_output = false}> : (i32, i32, i32) -> ()
// CHECK-NEXT:       "comp.tile_regs_release"() : () -> ()
// CHECK-NEXT:       "cb.pop_front"(%2, %1) : (i32, i32) -> ()
// CHECK-NEXT:       "cb.pop_front"(%3, %1) : (i32, i32) -> ()
// CHECK-NEXT:       "cb.push_back"(%4, %1) : (i32, i32) -> ()
// CHECK-NEXT:       func.return
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   builtin.module attributes {kernel_type = "data_out"} {
// CHECK-NEXT:     func.func @kernel_main(%bank_id : i32, %mem_addr : i32, %size_bytes : i32) {
// CHECK-NEXT:       %0 = "dm.get_noc_addr_from_bank_id"(%bank_id, %mem_addr) <{dram = true}> : (i32, i32) -> i64
// CHECK-NEXT:       %1 = arith.constant 1 : i32
// CHECK-NEXT:       %2 = arith.constant 2 : i32
// CHECK-NEXT:       %3 = "cb.get_read_pointer"(%2) : (i32) -> i32
// CHECK-NEXT:       "cb.wait_front"(%2, %1) : (i32, i32) -> ()
// CHECK-NEXT:       "dm.noc_async_write"(%3, %0, %size_bytes) : (i32, i64, i32) -> ()
// CHECK-NEXT:       "dm.noc_async_write_barrier"() : () -> ()
// CHECK-NEXT:       "cb.pop_front"(%2, %1) : (i32, i32) -> ()
// CHECK-NEXT:       func.return
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
