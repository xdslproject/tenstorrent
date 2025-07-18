// RUN: python3.13 tenstorrent/tools/tt-opt %s -t tt-metalium | filecheck %s

builtin.module {
  builtin.module attributes {kernel_type = "host", vis = "external"} {
    func.func @host_entry(%0 : memref<32x32xf32>, %1 : memref<32x32xf32>, %2 : memref<32x32xf32>) {
      %zero = arith.constant 0 : i32
      %3 = arith.constant 1 : i32
      %4 = arith.constant 0 : i32
      %5 = arith.constant 1 : i32
      %6 = arith.constant 2 : i32
      %prog = "tthost.create_program"() : () -> !tthost.program
      %7 = "tthost.create_device"(%zero) : (i32) -> !tthost.device
      %8 = "tthost.core"(%zero, %zero) : (i32, i32) -> !tthost.corecoord
      %9 = "tthost.get_command_queue"(%7) : (!tthost.device) -> !tthost.command_queue
      %size = arith.constant 4096 : i32
      %10 = "tthost.create_dram_configuration"(%size, %size) : (i32, i32) -> !tthost.dram_buffer_config
      %11 = "tthost.create_buffer"(%10) : (!tthost.dram_buffer_config) -> !tthost.buffer
      %12 = arith.constant false
      "tthost.enqueue_write_buffer"(%9, %11, %0, %12) : (!tthost.command_queue, !tthost.buffer, memref<32x32xf32>, i1) -> ()
      %13 = arith.constant 1 : i32
      %14 = "tthost.create_cb_configuration"(%13, %size, %4) <{data_type = "float"}> : (i32, i32, i32) -> !tthost.circular_buffer_config
      %15 = "tthost.create_circular_buffer"(%prog, %8, %14) : (!tthost.program, !tthost.corecoord, !tthost.circular_buffer_config) -> !tthost.cb_handle
      %size_1 = arith.constant 4096 : i32
      %16 = "tthost.create_dram_configuration"(%size_1, %size_1) : (i32, i32) -> !tthost.dram_buffer_config
      %17 = "tthost.create_buffer"(%16) : (!tthost.dram_buffer_config) -> !tthost.buffer
      %18 = arith.constant false
      "tthost.enqueue_write_buffer"(%9, %17, %1, %18) : (!tthost.command_queue, !tthost.buffer, memref<32x32xf32>, i1) -> ()
      %19 = arith.constant 1 : i32
      %20 = "tthost.create_cb_configuration"(%19, %size_1, %5) <{data_type = "float"}> : (i32, i32, i32) -> !tthost.circular_buffer_config
      %21 = "tthost.create_circular_buffer"(%prog, %8, %20) : (!tthost.program, !tthost.corecoord, !tthost.circular_buffer_config) -> !tthost.cb_handle
      %size_2 = arith.constant 4096 : i32
      %22 = "tthost.create_dram_configuration"(%size_2, %size_2) : (i32, i32) -> !tthost.dram_buffer_config
      %23 = "tthost.create_buffer"(%22) : (!tthost.dram_buffer_config) -> !tthost.buffer
      %24 = arith.constant false
      "tthost.enqueue_write_buffer"(%9, %23, %2, %24) : (!tthost.command_queue, !tthost.buffer, memref<32x32xf32>, i1) -> ()
      %25 = arith.constant 1 : i32
      %26 = "tthost.create_cb_configuration"(%25, %size_2, %6) <{data_type = "float"}> : (i32, i32, i32) -> !tthost.circular_buffer_config
      %27 = "tthost.create_circular_buffer"(%prog, %8, %26) : (!tthost.program, !tthost.corecoord, !tthost.circular_buffer_config) -> !tthost.cb_handle
      %dram_addr = "tthost.get_memory_address"(%11) : (!tthost.buffer) -> index
      %dram_addr_1 = "tthost.get_memory_address"(%17) : (!tthost.buffer) -> index
      %dram_addr_2 = "tthost.get_memory_address"(%23) : (!tthost.buffer) -> index
      %reader_kernel = "tthost.create_kernel"(%prog, %8) <{kernel_name = "reader.cpp", riscv_core = #tthost.riscv_core<datamovement_0>, noc_id = #builtin.int<0>}> : (!tthost.program, !tthost.corecoord) -> !tthost.kernel
      %writer_kernel = "tthost.create_kernel"(%prog, %8) <{kernel_name = "writer.cpp", riscv_core = #tthost.riscv_core<datamovement_1>, noc_id = #builtin.int<1>}> : (!tthost.program, !tthost.corecoord) -> !tthost.kernel
      %compute_kernel = "tthost.create_compute_kernel"(%prog, %8) <{kernel_name = "compute.cpp", riscv_core = #tthost.riscv_core<compute>, math_fidelity = #tthost.math_fidelity<LoFi>, fp32_dest_acc_en = false, math_approx_mode = false}> : (!tthost.program, !tthost.corecoord) -> !tthost.kernel
      "tthost.set_runtime_args"(%prog, %compute_kernel, %8) {operandSegmentSizes = array<i32: 1, 1, 1, 0>} : (!tthost.program, !tthost.kernel, !tthost.corecoord) -> ()
      "tthost.set_runtime_args"(%prog, %reader_kernel, %8, %zero, %zero, %dram_addr, %dram_addr_1, %size, %size_1) {operandSegmentSizes = array<i32: 1, 1, 1, 6>} : (!tthost.program, !tthost.kernel, !tthost.corecoord, i32, i32, index, index, i32, i32) -> ()
      "tthost.set_runtime_args"(%prog, %writer_kernel, %8, %zero, %dram_addr_2, %size_2) {operandSegmentSizes = array<i32: 1, 1, 1, 3>} : (!tthost.program, !tthost.kernel, !tthost.corecoord, i32, index, i32) -> ()
      %28 = arith.constant false
      "tthost.enqueue_program"(%9, %prog, %28) : (!tthost.command_queue, !tthost.program, i1) -> ()
      "tthost.finish"(%9) : (!tthost.command_queue) -> ()
      "tthost.enqueue_read_buffer"(%9, %23, %2, %28) : (!tthost.command_queue, !tthost.buffer, memref<32x32xf32>, i1) -> ()
      "tthost.close_device"(%7) : (!tthost.device) -> ()
      func.return
    }
  }
  builtin.module attributes {kernel_type = "data_in"} {
    func.func @kernel_main(%bank_id : i32, %bank_id_1 : i32, %mem_addr : i32, %mem_addr_1 : i32, %size_bytes : i32, %size_bytes_1 : i32) {
      %0 = arith.constant 1 : i32
      %1 = arith.constant 0 : i8
      %2 = arith.constant 0 : i32
      %3 = "dm.get_noc_addr_from_bank_id"(%bank_id, %mem_addr, %1) <{dram = true}> : (i32, i32, i8) -> i64
      %4 = "cb.get_write_pointer"(%2) : (i32) -> i32
      "cb.reserve_back"(%2, %0) : (i32, i32) -> ()
      "dm.noc_async_read"(%3, %4, %size_bytes) : (i64, i32, i32) -> ()
      "dm.noc_async_read_barrier"() : () -> ()
      "cb.push_back"(%2, %0) : (i32, i32) -> ()
      %5 = arith.constant 1 : i32
      %6 = arith.constant 0 : i8
      %7 = arith.constant 1 : i32
      %8 = "dm.get_noc_addr_from_bank_id"(%bank_id_1, %mem_addr_1, %6) <{dram = true}> : (i32, i32, i8) -> i64
      %9 = "cb.get_write_pointer"(%7) : (i32) -> i32
      "cb.reserve_back"(%7, %5) : (i32, i32) -> ()
      "dm.noc_async_read"(%8, %9, %size_bytes_1) : (i64, i32, i32) -> ()
      "dm.noc_async_read_barrier"() : () -> ()
      "cb.push_back"(%7, %5) : (i32, i32) -> ()
      func.return
    }
  }
  builtin.module attributes {kernel_type = "compute"} {
    func.func @MAIN() {
      %0 = arith.constant 0 : i32
      %1 = arith.constant 1 : i32
      %2 = arith.constant 0 : i32
      %3 = arith.constant 1 : i32
      %4 = arith.constant 2 : i32
      %5 = arith.constant true
      %6 = arith.constant false
      "comp.binary_op_init_common"(%2, %3, %4) : (i32, i32, i32) -> ()
      "comp.add_tiles_init"(%2, %3, %6) : (i32, i32, i1) -> ()
      "cb.wait_front"(%3, %1) : (i32, i32) -> ()
      "cb.wait_front"(%2, %1) : (i32, i32) -> ()
      "comp.tile_regs_acquire"() : () -> ()
      "comp.add_tiles"(%2, %3, %0, %0, %0) : (i32, i32, i32, i32, i32) -> ()
      "comp.tile_regs_commit"() : () -> ()
      "comp.tile_regs_wait"() : () -> ()
      "comp.pack_tile"(%0, %4, %0) <{out_of_order_output = false}> : (i32, i32, i32) -> ()
      "comp.tile_regs_release"() : () -> ()
      "cb.pop_front"(%2, %1) : (i32, i32) -> ()
      "cb.pop_front"(%3, %1) : (i32, i32) -> ()
      "cb.push_back"(%4, %1) : (i32, i32) -> ()
      func.return
    }
  }
  builtin.module attributes {kernel_type = "data_out"} {
    func.func @kernel_main(%bank_id : i32, %mem_addr : i32, %size_bytes : i32) {
      %0 = "dm.get_noc_addr_from_bank_id"(%bank_id, %mem_addr) <{dram = true}> : (i32, i32) -> i64
      %1 = arith.constant 1 : i32
      %2 = arith.constant 2 : i32
      %3 = "cb.get_read_pointer"(%2) : (i32) -> i32
      "cb.wait_front"(%2, %1) : (i32, i32) -> ()
      "dm.noc_async_write"(%3, %0, %size_bytes) : (i32, i64, i32) -> ()
      "dm.noc_async_write_barrier"() : () -> ()
      "cb.pop_front"(%2, %1) : (i32, i32) -> ()
      func.return
    }
  }
}

// CHECK: binary_op_init_common(0, 1, 2);
// CHECK: add_tiles_init(0, 1, false);
// CHECK: cb_wait_front(1, 1);
// CHECK: cb_wait_front(0, 1);
// CHECK: tile_regs_acquire();
// CHECK: add_tiles(0, 1, 0, 0, 0);
