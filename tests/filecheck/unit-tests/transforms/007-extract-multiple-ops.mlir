// RUN: python3.13 tenstorrent/tools/tt-opt %s -p extract-metalium | filecheck %s

builtin.module {
  builtin.module {
    func.func @example_independent() {
      %0 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x32xi32>
      %1 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x32xi32>
      %2 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x32xi32>
      %3 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x32xi32>
      func.call @host_entry(%0, %1, %3) : (memref<32x32xi32>, memref<32x32xi32>, memref<32x32xi32>) -> ()
      "memref.copy"(%3, %2) : (memref<32x32xi32>, memref<32x32xi32>) -> ()
      %4 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x32xi32>
      %5 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x32xi32>
      %6 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x32xi32>
      %7 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x32xi32>
      func.call @host_entry_1(%4, %5, %7) : (memref<32x32xi32>, memref<32x32xi32>, memref<32x32xi32>) -> ()
      "memref.copy"(%7, %6) : (memref<32x32xi32>, memref<32x32xi32>) -> ()
      func.return
    }
    func.func @main() {
      "memref.alloca_scope"() ({
        func.call @example_independent() : () -> ()
        "memref.alloca_scope.return"() : () -> ()
      }) : () -> ()
      func.return
    }
    func.func private @host_entry(memref<32x32xi32>, memref<32x32xi32>, memref<32x32xi32>) -> ()
    func.func private @host_entry_1(memref<32x32xi32>, memref<32x32xi32>, memref<32x32xi32>) -> ()
  }
  builtin.module attributes {kernel_type = "host", vis = "external"} {
    func.func @host_entry(%0 : memref<32x32xi32>, %1 : memref<32x32xi32>, %2 : memref<32x32xi32>) {
      %zero = arith.constant 0 : i32
      %3 = arith.constant 1 : i32
      %4 = arith.constant 16 : i32
      %prog = "tthost.create_program"() : () -> !tthost.program
      %5 = "tthost.create_device"(%zero) : (i32) -> !tthost.device
      %6 = "tthost.core"(%zero, %zero) : (i32, i32) -> !tthost.corecoord
      %7 = "tthost.get_command_queue"(%5) : (!tthost.device) -> !tthost.command_queue
      %size = arith.constant 4096 : i32
      %8 = "tthost.create_dram_configuration"(%size, %size) : (i32, i32) -> !tthost.dram_buffer_config
      %9 = "tthost.create_buffer"(%8) : (!tthost.dram_buffer_config) -> !tthost.buffer
      %10 = arith.constant false
      "tthost.enqueue_write_buffer"(%7, %9, %0, %10) : (!tthost.command_queue, !tthost.buffer, memref<32x32xi32>, i1) -> ()
      %11 = arith.constant 1 : i32
      %12 = "tthost.create_cb_configuration"(%11, %size, %zero) <{data_type = "int"}> : (i32, i32, i32) -> !tthost.circular_buffer_config
      %13 = "tthost.create_circular_buffer"(%prog, %6, %12) : (!tthost.program, !tthost.corecoord, !tthost.circular_buffer_config) -> !tthost.cb_handle
      %size_1 = arith.constant 4096 : i32
      %14 = "tthost.create_dram_configuration"(%size_1, %size_1) : (i32, i32) -> !tthost.dram_buffer_config
      %15 = "tthost.create_buffer"(%14) : (!tthost.dram_buffer_config) -> !tthost.buffer
      %16 = arith.constant false
      "tthost.enqueue_write_buffer"(%7, %15, %1, %16) : (!tthost.command_queue, !tthost.buffer, memref<32x32xi32>, i1) -> ()
      %17 = arith.constant 1 : i32
      %18 = "tthost.create_cb_configuration"(%17, %size_1, %3) <{data_type = "int"}> : (i32, i32, i32) -> !tthost.circular_buffer_config
      %19 = "tthost.create_circular_buffer"(%prog, %6, %18) : (!tthost.program, !tthost.corecoord, !tthost.circular_buffer_config) -> !tthost.cb_handle
      %size_2 = arith.constant 4096 : i32
      %20 = "tthost.create_dram_configuration"(%size_2, %size_2) : (i32, i32) -> !tthost.dram_buffer_config
      %21 = "tthost.create_buffer"(%20) : (!tthost.dram_buffer_config) -> !tthost.buffer
      %22 = arith.constant false
      "tthost.enqueue_write_buffer"(%7, %21, %2, %22) : (!tthost.command_queue, !tthost.buffer, memref<32x32xi32>, i1) -> ()
      %23 = arith.constant 1 : i32
      %24 = "tthost.create_cb_configuration"(%23, %size_2, %4) <{data_type = "int"}> : (i32, i32, i32) -> !tthost.circular_buffer_config
      %25 = "tthost.create_circular_buffer"(%prog, %6, %24) : (!tthost.program, !tthost.corecoord, !tthost.circular_buffer_config) -> !tthost.cb_handle
      %dram_addr = "tthost.get_memory_address"(%9) : (!tthost.buffer) -> index
      %dram_addr_1 = "tthost.get_memory_address"(%15) : (!tthost.buffer) -> index
      %dram_addr_2 = "tthost.get_memory_address"(%21) : (!tthost.buffer) -> index
      %reader_kernel = "tthost.create_kernel"(%prog, %6) <{kernel_name = "reader.cpp", riscv_core = #tthost.riscv_core<datamovement_0>, noc_id = #builtin.int<0>}> : (!tthost.program, !tthost.corecoord) -> !tthost.kernel
      %writer_kernel = "tthost.create_kernel"(%prog, %6) <{kernel_name = "writer.cpp", riscv_core = #tthost.riscv_core<datamovement_1>, noc_id = #builtin.int<1>}> : (!tthost.program, !tthost.corecoord) -> !tthost.kernel
      %compute_kernel = "tthost.create_compute_kernel"(%prog, %6) <{kernel_name = "compute.cpp", riscv_core = #tthost.riscv_core<compute>, math_fidelity = #tthost.math_fidelity<LoFi>, fp32_dest_acc_en = false, math_approx_mode = false}> : (!tthost.program, !tthost.corecoord) -> !tthost.kernel
      "tthost.set_runtime_args"(%prog, %compute_kernel, %6) {operandSegmentSizes = array<i32: 1, 1, 1, 0>} : (!tthost.program, !tthost.kernel, !tthost.corecoord) -> ()
      "tthost.set_runtime_args"(%prog, %reader_kernel, %6, %zero, %zero, %dram_addr, %dram_addr_1, %size, %size_1) {operandSegmentSizes = array<i32: 1, 1, 1, 6>} : (!tthost.program, !tthost.kernel, !tthost.corecoord, i32, i32, index, index, i32, i32) -> ()
      "tthost.set_runtime_args"(%prog, %writer_kernel, %6, %zero, %dram_addr_2, %size_2) {operandSegmentSizes = array<i32: 1, 1, 1, 3>} : (!tthost.program, !tthost.kernel, !tthost.corecoord, i32, index, i32) -> ()
      %26 = arith.constant false
      "tthost.enqueue_program"(%7, %prog, %26) : (!tthost.command_queue, !tthost.program, i1) -> ()
      "tthost.finish"(%7) : (!tthost.command_queue) -> ()
      "tthost.enqueue_read_buffer"(%7, %21, %2, %26) : (!tthost.command_queue, !tthost.buffer, memref<32x32xi32>, i1) -> ()
      "tthost.close_device"(%5) : (!tthost.device) -> ()
      func.return
    }
  }
  builtin.module attributes {kernel_type = "data_in"} {
    func.func @kernel_main(%bank_id : ui32, %bank_id_1 : ui32, %mem_addr : ui32, %mem_addr_1 : ui32, %size_bytes : ui32, %size_bytes_1 : ui32) {
      %0 = arith.constant 1 : i32
      %1 = arith.constant 0 : i8
      %2 = builtin.unrealized_conversion_cast %1 : i8 to ui8
      %3 = arith.constant 0 : i32
      %4 = "dm.get_noc_addr_from_bank_id"(%bank_id, %mem_addr, %2) <{dram = true}> : (ui32, ui32, ui8) -> ui64
      %5 = "cb.get_write_pointer"(%3) : (i32) -> ui32
      "cb.reserve_back"(%3, %0) : (i32, i32) -> ()
      "dm.noc_async_read"(%4, %5, %size_bytes) : (ui64, ui32, ui32) -> ()
      "dm.noc_async_read_barrier"() : () -> ()
      "cb.push_back"(%3, %0) : (i32, i32) -> ()
      %6 = arith.constant 1 : i32
      %7 = arith.constant 0 : i8
      %8 = builtin.unrealized_conversion_cast %7 : i8 to ui8
      %9 = arith.constant 1 : i32
      %10 = "dm.get_noc_addr_from_bank_id"(%bank_id_1, %mem_addr_1, %8) <{dram = true}> : (ui32, ui32, ui8) -> ui64
      %11 = "cb.get_write_pointer"(%9) : (i32) -> ui32
      "cb.reserve_back"(%9, %6) : (i32, i32) -> ()
      "dm.noc_async_read"(%10, %11, %size_bytes_1) : (ui64, ui32, ui32) -> ()
      "dm.noc_async_read_barrier"() : () -> ()
      "cb.push_back"(%9, %6) : (i32, i32) -> ()
      func.return
    }
  }
  builtin.module attributes {kernel_type = "compute"} {
    func.func @MAIN() {
      %0 = arith.constant 0 : i32
      %1 = arith.constant 1 : i32
      %2 = arith.constant 16 : i32
      %3 = builtin.unrealized_conversion_cast %0 : i32 to ui32
      %4 = builtin.unrealized_conversion_cast %1 : i32 to ui32
      %5 = builtin.unrealized_conversion_cast %2 : i32 to ui32
      %6 = arith.constant true
      %7 = arith.constant false
      "comp.binary_op_init_common"(%3, %4, %5) : (ui32, ui32, ui32) -> ()
      "comp.add_tiles_init"(%3, %4, %7) : (ui32, ui32, i1) -> ()
      "cb.wait_front"(%1, %1) : (i32, i32) -> ()
      "cb.wait_front"(%0, %1) : (i32, i32) -> ()
      "comp.tile_regs_acquire"() : () -> ()
      "comp.add_tiles"(%3, %4, %3, %3, %3) : (ui32, ui32, ui32, ui32, ui32) -> ()
      "comp.tile_regs_commit"() : () -> ()
      "comp.tile_regs_wait"() : () -> ()
      "comp.pack_tile"(%3, %5, %3) <{out_of_order_output = false}> : (ui32, ui32, ui32) -> ()
      "comp.tile_regs_release"() : () -> ()
      "cb.pop_front"(%0, %1) : (i32, i32) -> ()
      "cb.pop_front"(%1, %1) : (i32, i32) -> ()
      "cb.push_back"(%2, %1) : (i32, i32) -> ()
      func.return
    }
  }
  builtin.module attributes {kernel_type = "data_out"} {
    func.func @kernel_main(%bank_id : ui32, %mem_addr : ui32, %size_bytes : ui32) {
      %0 = "dm.get_noc_addr_from_bank_id"(%bank_id, %mem_addr) <{dram = true}> : (ui32, ui32) -> ui64
      %1 = arith.constant 1 : i32
      %2 = arith.constant 16 : i32
      %3 = "cb.get_read_pointer"(%2) : (i32) -> ui32
      "cb.wait_front"(%2, %1) : (i32, i32) -> ()
      "dm.noc_async_write"(%3, %0, %size_bytes) : (ui32, ui64, ui32) -> ()
      "dm.noc_async_write_barrier"() : () -> ()
      "cb.pop_front"(%2, %1) : (i32, i32) -> ()
      func.return
    }
  }
  builtin.module attributes {kernel_type = "host", vis = "external"} {
    func.func @host_entry_1(%0 : memref<32x32xi32>, %1 : memref<32x32xi32>, %2 : memref<32x32xi32>) {
      %zero = arith.constant 0 : i32
      %3 = arith.constant 1 : i32
      %4 = arith.constant 16 : i32
      %prog = "tthost.create_program"() : () -> !tthost.program
      %5 = "tthost.create_device"(%zero) : (i32) -> !tthost.device
      %6 = "tthost.core"(%zero, %zero) : (i32, i32) -> !tthost.corecoord
      %7 = "tthost.get_command_queue"(%5) : (!tthost.device) -> !tthost.command_queue
      %size = arith.constant 4096 : i32
      %8 = "tthost.create_dram_configuration"(%size, %size) : (i32, i32) -> !tthost.dram_buffer_config
      %9 = "tthost.create_buffer"(%8) : (!tthost.dram_buffer_config) -> !tthost.buffer
      %10 = arith.constant false
      "tthost.enqueue_write_buffer"(%7, %9, %0, %10) : (!tthost.command_queue, !tthost.buffer, memref<32x32xi32>, i1) -> ()
      %11 = arith.constant 1 : i32
      %12 = "tthost.create_cb_configuration"(%11, %size, %zero) <{data_type = "int"}> : (i32, i32, i32) -> !tthost.circular_buffer_config
      %13 = "tthost.create_circular_buffer"(%prog, %6, %12) : (!tthost.program, !tthost.corecoord, !tthost.circular_buffer_config) -> !tthost.cb_handle
      %size_1 = arith.constant 4096 : i32
      %14 = "tthost.create_dram_configuration"(%size_1, %size_1) : (i32, i32) -> !tthost.dram_buffer_config
      %15 = "tthost.create_buffer"(%14) : (!tthost.dram_buffer_config) -> !tthost.buffer
      %16 = arith.constant false
      "tthost.enqueue_write_buffer"(%7, %15, %1, %16) : (!tthost.command_queue, !tthost.buffer, memref<32x32xi32>, i1) -> ()
      %17 = arith.constant 1 : i32
      %18 = "tthost.create_cb_configuration"(%17, %size_1, %3) <{data_type = "int"}> : (i32, i32, i32) -> !tthost.circular_buffer_config
      %19 = "tthost.create_circular_buffer"(%prog, %6, %18) : (!tthost.program, !tthost.corecoord, !tthost.circular_buffer_config) -> !tthost.cb_handle
      %size_2 = arith.constant 4096 : i32
      %20 = "tthost.create_dram_configuration"(%size_2, %size_2) : (i32, i32) -> !tthost.dram_buffer_config
      %21 = "tthost.create_buffer"(%20) : (!tthost.dram_buffer_config) -> !tthost.buffer
      %22 = arith.constant false
      "tthost.enqueue_write_buffer"(%7, %21, %2, %22) : (!tthost.command_queue, !tthost.buffer, memref<32x32xi32>, i1) -> ()
      %23 = arith.constant 1 : i32
      %24 = "tthost.create_cb_configuration"(%23, %size_2, %4) <{data_type = "int"}> : (i32, i32, i32) -> !tthost.circular_buffer_config
      %25 = "tthost.create_circular_buffer"(%prog, %6, %24) : (!tthost.program, !tthost.corecoord, !tthost.circular_buffer_config) -> !tthost.cb_handle
      %dram_addr = "tthost.get_memory_address"(%9) : (!tthost.buffer) -> index
      %dram_addr_1 = "tthost.get_memory_address"(%15) : (!tthost.buffer) -> index
      %dram_addr_2 = "tthost.get_memory_address"(%21) : (!tthost.buffer) -> index
      %reader_kernel = "tthost.create_kernel"(%prog, %6) <{kernel_name = "reader_1.cpp", riscv_core = #tthost.riscv_core<datamovement_0>, noc_id = #builtin.int<0>}> : (!tthost.program, !tthost.corecoord) -> !tthost.kernel
      %writer_kernel = "tthost.create_kernel"(%prog, %6) <{kernel_name = "writer_1.cpp", riscv_core = #tthost.riscv_core<datamovement_1>, noc_id = #builtin.int<1>}> : (!tthost.program, !tthost.corecoord) -> !tthost.kernel
      %compute_kernel = "tthost.create_compute_kernel"(%prog, %6) <{kernel_name = "compute_1.cpp", riscv_core = #tthost.riscv_core<compute>, math_fidelity = #tthost.math_fidelity<LoFi>, fp32_dest_acc_en = false, math_approx_mode = false}> : (!tthost.program, !tthost.corecoord) -> !tthost.kernel
      "tthost.set_runtime_args"(%prog, %compute_kernel, %6) {operandSegmentSizes = array<i32: 1, 1, 1, 0>} : (!tthost.program, !tthost.kernel, !tthost.corecoord) -> ()
      "tthost.set_runtime_args"(%prog, %reader_kernel, %6, %zero, %zero, %dram_addr, %dram_addr_1, %size, %size_1) {operandSegmentSizes = array<i32: 1, 1, 1, 6>} : (!tthost.program, !tthost.kernel, !tthost.corecoord, i32, i32, index, index, i32, i32) -> ()
      "tthost.set_runtime_args"(%prog, %writer_kernel, %6, %zero, %dram_addr_2, %size_2) {operandSegmentSizes = array<i32: 1, 1, 1, 3>} : (!tthost.program, !tthost.kernel, !tthost.corecoord, i32, index, i32) -> ()
      %26 = arith.constant false
      "tthost.enqueue_program"(%7, %prog, %26) : (!tthost.command_queue, !tthost.program, i1) -> ()
      "tthost.finish"(%7) : (!tthost.command_queue) -> ()
      "tthost.enqueue_read_buffer"(%7, %21, %2, %26) : (!tthost.command_queue, !tthost.buffer, memref<32x32xi32>, i1) -> ()
      "tthost.close_device"(%5) : (!tthost.device) -> ()
      func.return
    }
  }
  builtin.module attributes {kernel_type = "data_in"} {
    func.func @kernel_main(%bank_id : ui32, %bank_id_1 : ui32, %mem_addr : ui32, %mem_addr_1 : ui32, %size_bytes : ui32, %size_bytes_1 : ui32) {
      %0 = arith.constant 1 : i32
      %1 = arith.constant 0 : i8
      %2 = builtin.unrealized_conversion_cast %1 : i8 to ui8
      %3 = arith.constant 0 : i32
      %4 = "dm.get_noc_addr_from_bank_id"(%bank_id, %mem_addr, %2) <{dram = true}> : (ui32, ui32, ui8) -> ui64
      %5 = "cb.get_write_pointer"(%3) : (i32) -> ui32
      "cb.reserve_back"(%3, %0) : (i32, i32) -> ()
      "dm.noc_async_read"(%4, %5, %size_bytes) : (ui64, ui32, ui32) -> ()
      "dm.noc_async_read_barrier"() : () -> ()
      "cb.push_back"(%3, %0) : (i32, i32) -> ()
      %6 = arith.constant 1 : i32
      %7 = arith.constant 0 : i8
      %8 = builtin.unrealized_conversion_cast %7 : i8 to ui8
      %9 = arith.constant 1 : i32
      %10 = "dm.get_noc_addr_from_bank_id"(%bank_id_1, %mem_addr_1, %8) <{dram = true}> : (ui32, ui32, ui8) -> ui64
      %11 = "cb.get_write_pointer"(%9) : (i32) -> ui32
      "cb.reserve_back"(%9, %6) : (i32, i32) -> ()
      "dm.noc_async_read"(%10, %11, %size_bytes_1) : (ui64, ui32, ui32) -> ()
      "dm.noc_async_read_barrier"() : () -> ()
      "cb.push_back"(%9, %6) : (i32, i32) -> ()
      func.return
    }
  }
  builtin.module attributes {kernel_type = "compute"} {
    func.func @MAIN() {
      %0 = arith.constant 0 : i32
      %1 = arith.constant 1 : i32
      %2 = arith.constant 16 : i32
      %3 = builtin.unrealized_conversion_cast %0 : i32 to ui32
      %4 = builtin.unrealized_conversion_cast %1 : i32 to ui32
      %5 = builtin.unrealized_conversion_cast %2 : i32 to ui32
      %6 = arith.constant true
      %7 = arith.constant false
      "comp.binary_op_init_common"(%3, %4, %5) : (ui32, ui32, ui32) -> ()
      "comp.mm_init"(%3, %4, %5, %3) : (ui32, ui32, ui32, ui32) -> ()
      "cb.wait_front"(%1, %1) : (i32, i32) -> ()
      "cb.wait_front"(%0, %1) : (i32, i32) -> ()
      "comp.tile_regs_acquire"() : () -> ()
      "comp.matmul_tiles"(%3, %4, %3, %3, %3, %3) : (ui32, ui32, ui32, ui32, ui32, ui32) -> ()
      "comp.tile_regs_commit"() : () -> ()
      "comp.tile_regs_wait"() : () -> ()
      "comp.pack_tile"(%3, %5, %3) <{out_of_order_output = false}> : (ui32, ui32, ui32) -> ()
      "comp.tile_regs_release"() : () -> ()
      "cb.pop_front"(%0, %1) : (i32, i32) -> ()
      "cb.pop_front"(%1, %1) : (i32, i32) -> ()
      "cb.push_back"(%2, %1) : (i32, i32) -> ()
      func.return
    }
  }
  builtin.module attributes {kernel_type = "data_out"} {
    func.func @kernel_main(%bank_id : ui32, %mem_addr : ui32, %size_bytes : ui32) {
      %0 = "dm.get_noc_addr_from_bank_id"(%bank_id, %mem_addr) <{dram = true}> : (ui32, ui32) -> ui64
      %1 = arith.constant 1 : i32
      %2 = arith.constant 16 : i32
      %3 = "cb.get_read_pointer"(%2) : (i32) -> ui32
      "cb.wait_front"(%2, %1) : (i32, i32) -> ()
      "dm.noc_async_write"(%3, %0, %size_bytes) : (ui32, ui64, ui32) -> ()
      "dm.noc_async_write_barrier"() : () -> ()
      "cb.pop_front"(%2, %1) : (i32, i32) -> ()
      func.return
    }
  }
}

// CHECK:      builtin.module {
// CHECK-NEXT:   builtin.module attributes {kernel_type = "host", vis = "external"} {
// CHECK-NEXT:     func.func @host_entry(%0 : memref<32x32xi32>, %1 : memref<32x32xi32>, %2 : memref<32x32xi32>) {
// CHECK-NEXT:       %zero = arith.constant 0 : i32
// CHECK-NEXT:       %3 = arith.constant 1 : i32
// CHECK-NEXT:       %4 = arith.constant 16 : i32
// CHECK-NEXT:       %prog = "tthost.create_program"() : () -> !tthost.program
// CHECK-NEXT:       %5 = "tthost.create_device"(%zero) : (i32) -> !tthost.device
// CHECK-NEXT:       %6 = "tthost.core"(%zero, %zero) : (i32, i32) -> !tthost.corecoord
// CHECK-NEXT:       %7 = "tthost.get_command_queue"(%5) : (!tthost.device) -> !tthost.command_queue
// CHECK-NEXT:       %size = arith.constant 4096 : i32
// CHECK-NEXT:       %8 = "tthost.create_dram_configuration"(%size, %size) : (i32, i32) -> !tthost.dram_buffer_config
// CHECK-NEXT:       %9 = "tthost.create_buffer"(%8) : (!tthost.dram_buffer_config) -> !tthost.buffer
// CHECK-NEXT:       %10 = arith.constant false
// CHECK-NEXT:       "tthost.enqueue_write_buffer"(%7, %9, %0, %10) : (!tthost.command_queue, !tthost.buffer, memref<32x32xi32>, i1) -> ()
// CHECK-NEXT:       %11 = arith.constant 1 : i32
// CHECK-NEXT:       %12 = "tthost.create_cb_configuration"(%11, %size, %zero) <{data_type = "int"}> : (i32, i32, i32) -> !tthost.circular_buffer_config
// CHECK-NEXT:       %13 = "tthost.create_circular_buffer"(%prog, %6, %12) : (!tthost.program, !tthost.corecoord, !tthost.circular_buffer_config) -> !tthost.cb_handle
// CHECK-NEXT:       %size_1 = arith.constant 4096 : i32
// CHECK-NEXT:       %14 = "tthost.create_dram_configuration"(%size_1, %size_1) : (i32, i32) -> !tthost.dram_buffer_config
// CHECK-NEXT:       %15 = "tthost.create_buffer"(%14) : (!tthost.dram_buffer_config) -> !tthost.buffer
// CHECK-NEXT:       %16 = arith.constant false
// CHECK-NEXT:       "tthost.enqueue_write_buffer"(%7, %15, %1, %16) : (!tthost.command_queue, !tthost.buffer, memref<32x32xi32>, i1) -> ()
// CHECK-NEXT:       %17 = arith.constant 1 : i32
// CHECK-NEXT:       %18 = "tthost.create_cb_configuration"(%17, %size_1, %3) <{data_type = "int"}> : (i32, i32, i32) -> !tthost.circular_buffer_config
// CHECK-NEXT:       %19 = "tthost.create_circular_buffer"(%prog, %6, %18) : (!tthost.program, !tthost.corecoord, !tthost.circular_buffer_config) -> !tthost.cb_handle
// CHECK-NEXT:       %size_2 = arith.constant 4096 : i32
// CHECK-NEXT:       %20 = "tthost.create_dram_configuration"(%size_2, %size_2) : (i32, i32) -> !tthost.dram_buffer_config
// CHECK-NEXT:       %21 = "tthost.create_buffer"(%20) : (!tthost.dram_buffer_config) -> !tthost.buffer
// CHECK-NEXT:       %22 = arith.constant false
// CHECK-NEXT:       "tthost.enqueue_write_buffer"(%7, %21, %2, %22) : (!tthost.command_queue, !tthost.buffer, memref<32x32xi32>, i1) -> ()
// CHECK-NEXT:       %23 = arith.constant 1 : i32
// CHECK-NEXT:       %24 = "tthost.create_cb_configuration"(%23, %size_2, %4) <{data_type = "int"}> : (i32, i32, i32) -> !tthost.circular_buffer_config
// CHECK-NEXT:       %25 = "tthost.create_circular_buffer"(%prog, %6, %24) : (!tthost.program, !tthost.corecoord, !tthost.circular_buffer_config) -> !tthost.cb_handle
// CHECK-NEXT:       %dram_addr = "tthost.get_memory_address"(%9) : (!tthost.buffer) -> index
// CHECK-NEXT:       %dram_addr_1 = "tthost.get_memory_address"(%15) : (!tthost.buffer) -> index
// CHECK-NEXT:       %dram_addr_2 = "tthost.get_memory_address"(%21) : (!tthost.buffer) -> index
// CHECK-NEXT:       %reader_kernel = "tthost.create_kernel"(%prog, %6) <{kernel_name = "reader.cpp", riscv_core = #tthost.riscv_core<datamovement_0>, noc_id = #builtin.int<0>}> : (!tthost.program, !tthost.corecoord) -> !tthost.kernel
// CHECK-NEXT:       %writer_kernel = "tthost.create_kernel"(%prog, %6) <{kernel_name = "writer.cpp", riscv_core = #tthost.riscv_core<datamovement_1>, noc_id = #builtin.int<1>}> : (!tthost.program, !tthost.corecoord) -> !tthost.kernel
// CHECK-NEXT:       %compute_kernel = "tthost.create_compute_kernel"(%prog, %6) <{kernel_name = "compute.cpp", riscv_core = #tthost.riscv_core<compute>, math_fidelity = #tthost.math_fidelity<LoFi>, fp32_dest_acc_en = false, math_approx_mode = false}> : (!tthost.program, !tthost.corecoord) -> !tthost.kernel
// CHECK-NEXT:       "tthost.set_runtime_args"(%prog, %compute_kernel, %6) {operandSegmentSizes = array<i32: 1, 1, 1, 0>} : (!tthost.program, !tthost.kernel, !tthost.corecoord) -> ()
// CHECK-NEXT:       "tthost.set_runtime_args"(%prog, %reader_kernel, %6, %zero, %zero, %dram_addr, %dram_addr_1, %size, %size_1) {operandSegmentSizes = array<i32: 1, 1, 1, 6>} : (!tthost.program, !tthost.kernel, !tthost.corecoord, i32, i32, index, index, i32, i32) -> ()
// CHECK-NEXT:       "tthost.set_runtime_args"(%prog, %writer_kernel, %6, %zero, %dram_addr_2, %size_2) {operandSegmentSizes = array<i32: 1, 1, 1, 3>} : (!tthost.program, !tthost.kernel, !tthost.corecoord, i32, index, i32) -> ()
// CHECK-NEXT:       %26 = arith.constant false
// CHECK-NEXT:       "tthost.enqueue_program"(%7, %prog, %26) : (!tthost.command_queue, !tthost.program, i1) -> ()
// CHECK-NEXT:       "tthost.finish"(%7) : (!tthost.command_queue) -> ()
// CHECK-NEXT:       "tthost.enqueue_read_buffer"(%7, %21, %2, %26) : (!tthost.command_queue, !tthost.buffer, memref<32x32xi32>, i1) -> ()
// CHECK-NEXT:       "tthost.close_device"(%5) : (!tthost.device) -> ()
// CHECK-NEXT:       func.return
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   builtin.module attributes {kernel_type = "data_in"} {
// CHECK-NEXT:     func.func @kernel_main(%bank_id : ui32, %bank_id_1 : ui32, %mem_addr : ui32, %mem_addr_1 : ui32, %size_bytes : ui32, %size_bytes_1 : ui32) {
// CHECK-NEXT:       %0 = arith.constant 1 : i32
// CHECK-NEXT:       %1 = arith.constant 0 : i8
// CHECK-NEXT:       %2 = builtin.unrealized_conversion_cast %1 : i8 to ui8
// CHECK-NEXT:       %3 = arith.constant 0 : i32
// CHECK-NEXT:       %4 = "dm.get_noc_addr_from_bank_id"(%bank_id, %mem_addr, %2) <{dram = true}> : (ui32, ui32, ui8) -> ui64
// CHECK-NEXT:       %5 = "cb.get_write_pointer"(%3) : (i32) -> ui32
// CHECK-NEXT:       "cb.reserve_back"(%3, %0) : (i32, i32) -> ()
// CHECK-NEXT:       "dm.noc_async_read"(%4, %5, %size_bytes) : (ui64, ui32, ui32) -> ()
// CHECK-NEXT:       "dm.noc_async_read_barrier"() : () -> ()
// CHECK-NEXT:       "cb.push_back"(%3, %0) : (i32, i32) -> ()
// CHECK-NEXT:       %6 = arith.constant 1 : i32
// CHECK-NEXT:       %7 = arith.constant 0 : i8
// CHECK-NEXT:       %8 = builtin.unrealized_conversion_cast %7 : i8 to ui8
// CHECK-NEXT:       %9 = arith.constant 1 : i32
// CHECK-NEXT:       %10 = "dm.get_noc_addr_from_bank_id"(%bank_id_1, %mem_addr_1, %8) <{dram = true}> : (ui32, ui32, ui8) -> ui64
// CHECK-NEXT:       %11 = "cb.get_write_pointer"(%9) : (i32) -> ui32
// CHECK-NEXT:       "cb.reserve_back"(%9, %6) : (i32, i32) -> ()
// CHECK-NEXT:       "dm.noc_async_read"(%10, %11, %size_bytes_1) : (ui64, ui32, ui32) -> ()
// CHECK-NEXT:       "dm.noc_async_read_barrier"() : () -> ()
// CHECK-NEXT:       "cb.push_back"(%9, %6) : (i32, i32) -> ()
// CHECK-NEXT:       func.return
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   builtin.module attributes {kernel_type = "compute"} {
// CHECK-NEXT:     func.func @MAIN() {
// CHECK-NEXT:       %0 = arith.constant 0 : i32
// CHECK-NEXT:       %1 = arith.constant 1 : i32
// CHECK-NEXT:       %2 = arith.constant 16 : i32
// CHECK-NEXT:       %3 = builtin.unrealized_conversion_cast %0 : i32 to ui32
// CHECK-NEXT:       %4 = builtin.unrealized_conversion_cast %1 : i32 to ui32
// CHECK-NEXT:       %5 = builtin.unrealized_conversion_cast %2 : i32 to ui32
// CHECK-NEXT:       %6 = arith.constant true
// CHECK-NEXT:       %7 = arith.constant false
// CHECK-NEXT:       "comp.binary_op_init_common"(%3, %4, %5) : (ui32, ui32, ui32) -> ()
// CHECK-NEXT:       "comp.add_tiles_init"(%3, %4, %7) : (ui32, ui32, i1) -> ()
// CHECK-NEXT:       "cb.wait_front"(%1, %1) : (i32, i32) -> ()
// CHECK-NEXT:       "cb.wait_front"(%0, %1) : (i32, i32) -> ()
// CHECK-NEXT:       "comp.tile_regs_acquire"() : () -> ()
// CHECK-NEXT:       "comp.add_tiles"(%3, %4, %3, %3, %3) : (ui32, ui32, ui32, ui32, ui32) -> ()
// CHECK-NEXT:       "comp.tile_regs_commit"() : () -> ()
// CHECK-NEXT:       "comp.tile_regs_wait"() : () -> ()
// CHECK-NEXT:       "comp.pack_tile"(%3, %5, %3) <{out_of_order_output = false}> : (ui32, ui32, ui32) -> ()
// CHECK-NEXT:       "comp.tile_regs_release"() : () -> ()
// CHECK-NEXT:       "cb.pop_front"(%0, %1) : (i32, i32) -> ()
// CHECK-NEXT:       "cb.pop_front"(%1, %1) : (i32, i32) -> ()
// CHECK-NEXT:       "cb.push_back"(%2, %1) : (i32, i32) -> ()
// CHECK-NEXT:       func.return
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   builtin.module attributes {kernel_type = "data_out"} {
// CHECK-NEXT:     func.func @kernel_main(%bank_id : ui32, %mem_addr : ui32, %size_bytes : ui32) {
// CHECK-NEXT:       %0 = "dm.get_noc_addr_from_bank_id"(%bank_id, %mem_addr) <{dram = true}> : (ui32, ui32) -> ui64
// CHECK-NEXT:       %1 = arith.constant 1 : i32
// CHECK-NEXT:       %2 = arith.constant 16 : i32
// CHECK-NEXT:       %3 = "cb.get_read_pointer"(%2) : (i32) -> ui32
// CHECK-NEXT:       "cb.wait_front"(%2, %1) : (i32, i32) -> ()
// CHECK-NEXT:       "dm.noc_async_write"(%3, %0, %size_bytes) : (ui32, ui64, ui32) -> ()
// CHECK-NEXT:       "dm.noc_async_write_barrier"() : () -> ()
// CHECK-NEXT:       "cb.pop_front"(%2, %1) : (i32, i32) -> ()
// CHECK-NEXT:       func.return
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   builtin.module attributes {kernel_type = "host", vis = "external"} {
// CHECK-NEXT:     func.func @host_entry_1(%0 : memref<32x32xi32>, %1 : memref<32x32xi32>, %2 : memref<32x32xi32>) {
// CHECK-NEXT:       %zero = arith.constant 0 : i32
// CHECK-NEXT:       %3 = arith.constant 1 : i32
// CHECK-NEXT:       %4 = arith.constant 16 : i32
// CHECK-NEXT:       %prog = "tthost.create_program"() : () -> !tthost.program
// CHECK-NEXT:       %5 = "tthost.create_device"(%zero) : (i32) -> !tthost.device
// CHECK-NEXT:       %6 = "tthost.core"(%zero, %zero) : (i32, i32) -> !tthost.corecoord
// CHECK-NEXT:       %7 = "tthost.get_command_queue"(%5) : (!tthost.device) -> !tthost.command_queue
// CHECK-NEXT:       %size = arith.constant 4096 : i32
// CHECK-NEXT:       %8 = "tthost.create_dram_configuration"(%size, %size) : (i32, i32) -> !tthost.dram_buffer_config
// CHECK-NEXT:       %9 = "tthost.create_buffer"(%8) : (!tthost.dram_buffer_config) -> !tthost.buffer
// CHECK-NEXT:       %10 = arith.constant false
// CHECK-NEXT:       "tthost.enqueue_write_buffer"(%7, %9, %0, %10) : (!tthost.command_queue, !tthost.buffer, memref<32x32xi32>, i1) -> ()
// CHECK-NEXT:       %11 = arith.constant 1 : i32
// CHECK-NEXT:       %12 = "tthost.create_cb_configuration"(%11, %size, %zero) <{data_type = "int"}> : (i32, i32, i32) -> !tthost.circular_buffer_config
// CHECK-NEXT:       %13 = "tthost.create_circular_buffer"(%prog, %6, %12) : (!tthost.program, !tthost.corecoord, !tthost.circular_buffer_config) -> !tthost.cb_handle
// CHECK-NEXT:       %size_1 = arith.constant 4096 : i32
// CHECK-NEXT:       %14 = "tthost.create_dram_configuration"(%size_1, %size_1) : (i32, i32) -> !tthost.dram_buffer_config
// CHECK-NEXT:       %15 = "tthost.create_buffer"(%14) : (!tthost.dram_buffer_config) -> !tthost.buffer
// CHECK-NEXT:       %16 = arith.constant false
// CHECK-NEXT:       "tthost.enqueue_write_buffer"(%7, %15, %1, %16) : (!tthost.command_queue, !tthost.buffer, memref<32x32xi32>, i1) -> ()
// CHECK-NEXT:       %17 = arith.constant 1 : i32
// CHECK-NEXT:       %18 = "tthost.create_cb_configuration"(%17, %size_1, %3) <{data_type = "int"}> : (i32, i32, i32) -> !tthost.circular_buffer_config
// CHECK-NEXT:       %19 = "tthost.create_circular_buffer"(%prog, %6, %18) : (!tthost.program, !tthost.corecoord, !tthost.circular_buffer_config) -> !tthost.cb_handle
// CHECK-NEXT:       %size_2 = arith.constant 4096 : i32
// CHECK-NEXT:       %20 = "tthost.create_dram_configuration"(%size_2, %size_2) : (i32, i32) -> !tthost.dram_buffer_config
// CHECK-NEXT:       %21 = "tthost.create_buffer"(%20) : (!tthost.dram_buffer_config) -> !tthost.buffer
// CHECK-NEXT:       %22 = arith.constant false
// CHECK-NEXT:       "tthost.enqueue_write_buffer"(%7, %21, %2, %22) : (!tthost.command_queue, !tthost.buffer, memref<32x32xi32>, i1) -> ()
// CHECK-NEXT:       %23 = arith.constant 1 : i32
// CHECK-NEXT:       %24 = "tthost.create_cb_configuration"(%23, %size_2, %4) <{data_type = "int"}> : (i32, i32, i32) -> !tthost.circular_buffer_config
// CHECK-NEXT:       %25 = "tthost.create_circular_buffer"(%prog, %6, %24) : (!tthost.program, !tthost.corecoord, !tthost.circular_buffer_config) -> !tthost.cb_handle
// CHECK-NEXT:       %dram_addr = "tthost.get_memory_address"(%9) : (!tthost.buffer) -> index
// CHECK-NEXT:       %dram_addr_1 = "tthost.get_memory_address"(%15) : (!tthost.buffer) -> index
// CHECK-NEXT:       %dram_addr_2 = "tthost.get_memory_address"(%21) : (!tthost.buffer) -> index
// CHECK-NEXT:       %reader_kernel = "tthost.create_kernel"(%prog, %6) <{kernel_name = "reader_1.cpp", riscv_core = #tthost.riscv_core<datamovement_0>, noc_id = #builtin.int<0>}> : (!tthost.program, !tthost.corecoord) -> !tthost.kernel
// CHECK-NEXT:       %writer_kernel = "tthost.create_kernel"(%prog, %6) <{kernel_name = "writer_1.cpp", riscv_core = #tthost.riscv_core<datamovement_1>, noc_id = #builtin.int<1>}> : (!tthost.program, !tthost.corecoord) -> !tthost.kernel
// CHECK-NEXT:       %compute_kernel = "tthost.create_compute_kernel"(%prog, %6) <{kernel_name = "compute_1.cpp", riscv_core = #tthost.riscv_core<compute>, math_fidelity = #tthost.math_fidelity<LoFi>, fp32_dest_acc_en = false, math_approx_mode = false}> : (!tthost.program, !tthost.corecoord) -> !tthost.kernel
// CHECK-NEXT:       "tthost.set_runtime_args"(%prog, %compute_kernel, %6) {operandSegmentSizes = array<i32: 1, 1, 1, 0>} : (!tthost.program, !tthost.kernel, !tthost.corecoord) -> ()
// CHECK-NEXT:       "tthost.set_runtime_args"(%prog, %reader_kernel, %6, %zero, %zero, %dram_addr, %dram_addr_1, %size, %size_1) {operandSegmentSizes = array<i32: 1, 1, 1, 6>} : (!tthost.program, !tthost.kernel, !tthost.corecoord, i32, i32, index, index, i32, i32) -> ()
// CHECK-NEXT:       "tthost.set_runtime_args"(%prog, %writer_kernel, %6, %zero, %dram_addr_2, %size_2) {operandSegmentSizes = array<i32: 1, 1, 1, 3>} : (!tthost.program, !tthost.kernel, !tthost.corecoord, i32, index, i32) -> ()
// CHECK-NEXT:       %26 = arith.constant false
// CHECK-NEXT:       "tthost.enqueue_program"(%7, %prog, %26) : (!tthost.command_queue, !tthost.program, i1) -> ()
// CHECK-NEXT:       "tthost.finish"(%7) : (!tthost.command_queue) -> ()
// CHECK-NEXT:       "tthost.enqueue_read_buffer"(%7, %21, %2, %26) : (!tthost.command_queue, !tthost.buffer, memref<32x32xi32>, i1) -> ()
// CHECK-NEXT:       "tthost.close_device"(%5) : (!tthost.device) -> ()
// CHECK-NEXT:       func.return
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   builtin.module attributes {kernel_type = "data_in"} {
// CHECK-NEXT:     func.func @kernel_main(%bank_id : ui32, %bank_id_1 : ui32, %mem_addr : ui32, %mem_addr_1 : ui32, %size_bytes : ui32, %size_bytes_1 : ui32) {
// CHECK-NEXT:       %0 = arith.constant 1 : i32
// CHECK-NEXT:       %1 = arith.constant 0 : i8
// CHECK-NEXT:       %2 = builtin.unrealized_conversion_cast %1 : i8 to ui8
// CHECK-NEXT:       %3 = arith.constant 0 : i32
// CHECK-NEXT:       %4 = "dm.get_noc_addr_from_bank_id"(%bank_id, %mem_addr, %2) <{dram = true}> : (ui32, ui32, ui8) -> ui64
// CHECK-NEXT:       %5 = "cb.get_write_pointer"(%3) : (i32) -> ui32
// CHECK-NEXT:       "cb.reserve_back"(%3, %0) : (i32, i32) -> ()
// CHECK-NEXT:       "dm.noc_async_read"(%4, %5, %size_bytes) : (ui64, ui32, ui32) -> ()
// CHECK-NEXT:       "dm.noc_async_read_barrier"() : () -> ()
// CHECK-NEXT:       "cb.push_back"(%3, %0) : (i32, i32) -> ()
// CHECK-NEXT:       %6 = arith.constant 1 : i32
// CHECK-NEXT:       %7 = arith.constant 0 : i8
// CHECK-NEXT:       %8 = builtin.unrealized_conversion_cast %7 : i8 to ui8
// CHECK-NEXT:       %9 = arith.constant 1 : i32
// CHECK-NEXT:       %10 = "dm.get_noc_addr_from_bank_id"(%bank_id_1, %mem_addr_1, %8) <{dram = true}> : (ui32, ui32, ui8) -> ui64
// CHECK-NEXT:       %11 = "cb.get_write_pointer"(%9) : (i32) -> ui32
// CHECK-NEXT:       "cb.reserve_back"(%9, %6) : (i32, i32) -> ()
// CHECK-NEXT:       "dm.noc_async_read"(%10, %11, %size_bytes_1) : (ui64, ui32, ui32) -> ()
// CHECK-NEXT:       "dm.noc_async_read_barrier"() : () -> ()
// CHECK-NEXT:       "cb.push_back"(%9, %6) : (i32, i32) -> ()
// CHECK-NEXT:       func.return
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   builtin.module attributes {kernel_type = "compute"} {
// CHECK-NEXT:     func.func @MAIN() {
// CHECK-NEXT:       %0 = arith.constant 0 : i32
// CHECK-NEXT:       %1 = arith.constant 1 : i32
// CHECK-NEXT:       %2 = arith.constant 16 : i32
// CHECK-NEXT:       %3 = builtin.unrealized_conversion_cast %0 : i32 to ui32
// CHECK-NEXT:       %4 = builtin.unrealized_conversion_cast %1 : i32 to ui32
// CHECK-NEXT:       %5 = builtin.unrealized_conversion_cast %2 : i32 to ui32
// CHECK-NEXT:       %6 = arith.constant true
// CHECK-NEXT:       %7 = arith.constant false
// CHECK-NEXT:       "comp.binary_op_init_common"(%3, %4, %5) : (ui32, ui32, ui32) -> ()
// CHECK-NEXT:       "comp.mm_init"(%3, %4, %5, %3) : (ui32, ui32, ui32, ui32) -> ()
// CHECK-NEXT:       "cb.wait_front"(%1, %1) : (i32, i32) -> ()
// CHECK-NEXT:       "cb.wait_front"(%0, %1) : (i32, i32) -> ()
// CHECK-NEXT:       "comp.tile_regs_acquire"() : () -> ()
// CHECK-NEXT:       "comp.matmul_tiles"(%3, %4, %3, %3, %3, %3) : (ui32, ui32, ui32, ui32, ui32, ui32) -> ()
// CHECK-NEXT:       "comp.tile_regs_commit"() : () -> ()
// CHECK-NEXT:       "comp.tile_regs_wait"() : () -> ()
// CHECK-NEXT:       "comp.pack_tile"(%3, %5, %3) <{out_of_order_output = false}> : (ui32, ui32, ui32) -> ()
// CHECK-NEXT:       "comp.tile_regs_release"() : () -> ()
// CHECK-NEXT:       "cb.pop_front"(%0, %1) : (i32, i32) -> ()
// CHECK-NEXT:       "cb.pop_front"(%1, %1) : (i32, i32) -> ()
// CHECK-NEXT:       "cb.push_back"(%2, %1) : (i32, i32) -> ()
// CHECK-NEXT:       func.return
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   builtin.module attributes {kernel_type = "data_out"} {
// CHECK-NEXT:     func.func @kernel_main(%bank_id : ui32, %mem_addr : ui32, %size_bytes : ui32) {
// CHECK-NEXT:       %0 = "dm.get_noc_addr_from_bank_id"(%bank_id, %mem_addr) <{dram = true}> : (ui32, ui32) -> ui64
// CHECK-NEXT:       %1 = arith.constant 1 : i32
// CHECK-NEXT:       %2 = arith.constant 16 : i32
// CHECK-NEXT:       %3 = "cb.get_read_pointer"(%2) : (i32) -> ui32
// CHECK-NEXT:       "cb.wait_front"(%2, %1) : (i32, i32) -> ()
// CHECK-NEXT:       "dm.noc_async_write"(%3, %0, %size_bytes) : (ui32, ui64, ui32) -> ()
// CHECK-NEXT:       "dm.noc_async_write_barrier"() : () -> ()
// CHECK-NEXT:       "cb.pop_front"(%2, %1) : (i32, i32) -> ()
// CHECK-NEXT:       func.return
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }