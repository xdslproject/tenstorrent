// RUN: python3.13 tenstorrent/tools/tt-opt %s -p linalg-to-tt | filecheck %s

builtin.module {
  func.func @"example_matmul"() {
    %0 = "memref.alloca"() <{"operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<32x32xi32>
    %1 = "memref.alloca"() <{"operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<32x32xi32>
    %2 = "memref.alloca"() <{"operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<32x32xi32>
    %3 = "memref.alloca"() <{"operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<32x32xi32>
    linalg.matmul ins(%0, %1 : memref<32x32xi32>, memref<32x32xi32>) outs(%3 : memref<32x32xi32>) -> ()
    "memref.copy"(%3, %2) : (memref<32x32xi32>, memref<32x32xi32>) -> ()
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
// CHECK-NEXT:       %0 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x32xi32>
// CHECK-NEXT:       %1 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x32xi32>
// CHECK-NEXT:       %2 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x32xi32>
// CHECK-NEXT:       %3 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x32xi32>
// CHECK-NEXT:       func.call @host_entry(%0, %1, %3) : (memref<32x32xi32>, memref<32x32xi32>, memref<32x32xi32>) -> ()
// CHECK-NEXT:       "memref.copy"(%3, %2) : (memref<32x32xi32>, memref<32x32xi32>) -> ()
// CHECK-NEXT:       func.return
// CHECK-NEXT:     }
// CHECK-NEXT:     func.func @main() {
// CHECK-NEXT:       "memref.alloca_scope"() ({
// CHECK-NEXT:         func.call @example_matmul() : () -> ()
// CHECK-NEXT:         "memref.alloca_scope.return"() : () -> ()
// CHECK-NEXT:       }) : () -> ()
// CHECK-NEXT:       func.return
// CHECK-NEXT:     }
// CHECK-NEXT:     func.func private @host_entry(memref<32x32xi32>, memref<32x32xi32>, memref<32x32xi32>) -> ()
// CHECK-NEXT:   }
// CHECK-NEXT:   builtin.module attributes {kernel_type = "host", vis = "external"} {
// CHECK-NEXT:     func.func @host_entry(%0 : memref<32x32xi32>, %1 : memref<32x32xi32>, %2 : memref<32x32xi32>) {
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
// CHECK-NEXT:       %10 = "tthost.create_dram_configuration"(%size, %size) : (i32, i32) -> !tthost.dram_buffer_config
// CHECK-NEXT:       %11 = "tthost.create_buffer"(%10) : (!tthost.dram_buffer_config) -> !tthost.buffer
// CHECK-NEXT:       %12 = arith.constant false
// CHECK-NEXT:       "tthost.enqueue_write_buffer"(%9, %11, %0, %12) : (!tthost.command_queue, !tthost.buffer, memref<32x32xi32>, i1) -> ()
// CHECK-NEXT:       %13 = arith.constant 1 : i32
// CHECK-NEXT:       %14 = "tthost.create_cb_configuration"(%13, %size, %4) <{data_type = "int"}> : (i32, i32, i32) -> !tthost.circular_buffer_config
// CHECK-NEXT:       %15 = "tthost.create_circular_buffer"(%prog, %8, %14) : (!tthost.program, !tthost.corecoord, !tthost.circular_buffer_config) -> !tthost.cb_handle
// CHECK-NEXT:       %size_1 = arith.constant 4096 : i32
// CHECK-NEXT:       %16 = "tthost.create_dram_configuration"(%size_1, %size_1) : (i32, i32) -> !tthost.dram_buffer_config
// CHECK-NEXT:       %17 = "tthost.create_buffer"(%16) : (!tthost.dram_buffer_config) -> !tthost.buffer
// CHECK-NEXT:       %18 = arith.constant false
// CHECK-NEXT:       "tthost.enqueue_write_buffer"(%9, %17, %1, %18) : (!tthost.command_queue, !tthost.buffer, memref<32x32xi32>, i1) -> ()
// CHECK-NEXT:       %19 = arith.constant 1 : i32
// CHECK-NEXT:       %20 = "tthost.create_cb_configuration"(%19, %size_1, %5) <{data_type = "int"}> : (i32, i32, i32) -> !tthost.circular_buffer_config
// CHECK-NEXT:       %21 = "tthost.create_circular_buffer"(%prog, %8, %20) : (!tthost.program, !tthost.corecoord, !tthost.circular_buffer_config) -> !tthost.cb_handle
// CHECK-NEXT:       %size_2 = arith.constant 4096 : i32
// CHECK-NEXT:       %22 = "tthost.create_dram_configuration"(%size_2, %size_2) : (i32, i32) -> !tthost.dram_buffer_config
// CHECK-NEXT:       %23 = "tthost.create_buffer"(%22) : (!tthost.dram_buffer_config) -> !tthost.buffer
// CHECK-NEXT:       %24 = arith.constant false
// CHECK-NEXT:       "tthost.enqueue_write_buffer"(%9, %23, %2, %24) : (!tthost.command_queue, !tthost.buffer, memref<32x32xi32>, i1) -> ()
// CHECK-NEXT:       %25 = arith.constant 1 : i32
// CHECK-NEXT:       %26 = "tthost.create_cb_configuration"(%25, %size_2, %6) <{data_type = "int"}> : (i32, i32, i32) -> !tthost.circular_buffer_config
// CHECK-NEXT:       %27 = "tthost.create_circular_buffer"(%prog, %8, %26) : (!tthost.program, !tthost.corecoord, !tthost.circular_buffer_config) -> !tthost.cb_handle
// CHECK-NEXT:       %dram_addr = "tthost.get_memory_address"(%11) : (!tthost.buffer) -> index
// CHECK-NEXT:       %dram_addr_1 = "tthost.get_memory_address"(%17) : (!tthost.buffer) -> index
// CHECK-NEXT:       %dram_addr_2 = "tthost.get_memory_address"(%23) : (!tthost.buffer) -> index
// CHECK-NEXT:       %reader_kernel = "tthost.create_kernel"(%prog, %8) <{kernel_name = "reader.cpp", riscv_core = #tthost.riscv_core<datamovement_0>, noc_id = #builtin.int<0>}> : (!tthost.program, !tthost.corecoord) -> !tthost.kernel
// CHECK-NEXT:       %writer_kernel = "tthost.create_kernel"(%prog, %8) <{kernel_name = "writer.cpp", riscv_core = #tthost.riscv_core<datamovement_1>, noc_id = #builtin.int<1>}> : (!tthost.program, !tthost.corecoord) -> !tthost.kernel
// CHECK-NEXT:       %compute_kernel = "tthost.create_compute_kernel"(%prog, %8) <{kernel_name = "compute.cpp", riscv_core = #tthost.riscv_core<compute>, math_fidelity = #tthost.math_fidelity<LoFi>, fp32_dest_acc_en = false, math_approx_mode = false}> : (!tthost.program, !tthost.corecoord) -> !tthost.kernel
// CHECK-NEXT:       "tthost.set_runtime_args"(%prog, %compute_kernel, %8) {operandSegmentSizes = array<i32: 1, 1, 1, 0>} : (!tthost.program, !tthost.kernel, !tthost.corecoord) -> ()
// CHECK-NEXT:       "tthost.set_runtime_args"(%prog, %reader_kernel, %8, %zero, %zero, %dram_addr, %dram_addr_1, %size, %size_1) {operandSegmentSizes = array<i32: 1, 1, 1, 6>} : (!tthost.program, !tthost.kernel, !tthost.corecoord, i32, i32, index, index, i32, i32) -> ()
// CHECK-NEXT:       "tthost.set_runtime_args"(%prog, %writer_kernel, %8, %zero, %dram_addr_2, %size_2) {operandSegmentSizes = array<i32: 1, 1, 1, 3>} : (!tthost.program, !tthost.kernel, !tthost.corecoord, i32, index, i32) -> ()
// CHECK-NEXT:       %28 = arith.constant false
// CHECK-NEXT:       "tthost.enqueue_program"(%9, %prog, %28) : (!tthost.command_queue, !tthost.program, i1) -> ()
// CHECK-NEXT:       "tthost.finish"(%9) : (!tthost.command_queue) -> ()
// CHECK-NEXT:       "tthost.enqueue_read_buffer"(%9, %23, %2, %28) : (!tthost.command_queue, !tthost.buffer, memref<32x32xi32>, i1) -> ()
// CHECK-NEXT:       "tthost.close_device"(%7) : (!tthost.device) -> ()
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
// CHECK-NEXT:       %2 = arith.constant 0 : i32
// CHECK-NEXT:       %3 = arith.constant 1 : i32
// CHECK-NEXT:       %4 = arith.constant 2 : i32
// CHECK-NEXT:       %5 = builtin.unrealized_conversion_cast %0 : i32 to ui32
// CHECK-NEXT:       %6 = builtin.unrealized_conversion_cast %1 : i32 to ui32
// CHECK-NEXT:       %7 = builtin.unrealized_conversion_cast %2 : i32 to ui32
// CHECK-NEXT:       %8 = builtin.unrealized_conversion_cast %3 : i32 to ui32
// CHECK-NEXT:       %9 = builtin.unrealized_conversion_cast %4 : i32 to ui32
// CHECK-NEXT:       %10 = arith.constant true
// CHECK-NEXT:       %11 = arith.constant false
// CHECK-NEXT:       "comp.binary_op_init_common"(%7, %8, %9) : (ui32, ui32, ui32) -> ()
// CHECK-NEXT:       "comp.mm_init"(%7, %8, %9, %5) : (ui32, ui32, ui32, ui32) -> ()
// CHECK-NEXT:       "cb.wait_front"(%3, %1) : (i32, i32) -> ()
// CHECK-NEXT:       "cb.wait_front"(%2, %1) : (i32, i32) -> ()
// CHECK-NEXT:       "comp.tile_regs_acquire"() : () -> ()
// CHECK-NEXT:       "comp.matmul_tiles"(%7, %8, %5, %5, %5, %5) : (ui32, ui32, ui32, ui32, ui32, ui32) -> ()
// CHECK-NEXT:       "comp.tile_regs_commit"() : () -> ()
// CHECK-NEXT:       "comp.tile_regs_wait"() : () -> ()
// CHECK-NEXT:       "comp.pack_tile"(%5, %9, %5) <{out_of_order_output = false}> : (ui32, ui32, ui32) -> ()
// CHECK-NEXT:       "comp.tile_regs_release"() : () -> ()
// CHECK-NEXT:       "cb.pop_front"(%2, %1) : (i32, i32) -> ()
// CHECK-NEXT:       "cb.pop_front"(%3, %1) : (i32, i32) -> ()
// CHECK-NEXT:       "cb.push_back"(%4, %1) : (i32, i32) -> ()
// CHECK-NEXT:       func.return
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   builtin.module attributes {kernel_type = "data_out"} {
// CHECK-NEXT:     func.func @kernel_main(%bank_id : ui32, %mem_addr : ui32, %size_bytes : ui32) {
// CHECK-NEXT:       %0 = "dm.get_noc_addr_from_bank_id"(%bank_id, %mem_addr) <{dram = true}> : (ui32, ui32) -> ui64
// CHECK-NEXT:       %1 = arith.constant 1 : i32
// CHECK-NEXT:       %2 = arith.constant 2 : i32
// CHECK-NEXT:       %3 = "cb.get_read_pointer"(%2) : (i32) -> ui32
// CHECK-NEXT:       "cb.wait_front"(%2, %1) : (i32, i32) -> ()
// CHECK-NEXT:       "dm.noc_async_write"(%3, %0, %size_bytes) : (ui32, ui64, ui32) -> ()
// CHECK-NEXT:       "dm.noc_async_write_barrier"() : () -> ()
// CHECK-NEXT:       "cb.pop_front"(%2, %1) : (i32, i32) -> ()
// CHECK-NEXT:       func.return
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
