// RUN: python3.13 tenstorrent/tools/tt-opt %s -p rewrite-matmul-to-tt | filecheck %s

builtin.module {
  func.func @"_QMproblem_modPentry"() {
    %0 = "memref.alloca"() <{"operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<10x10xi32>
    %1 = "memref.alloca"() <{"operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<10x10xi32>
    %2 = "memref.alloca"() <{"operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<10x10xi32>
    %3 = "memref.alloca"() <{"operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<10x10xi32>
    linalg.matmul ins(%0, %1 : memref<10x10xi32>, memref<10x10xi32>) outs(%3 : memref<10x10xi32>) -> ()
    "memref.copy"(%3, %2) : (memref<10x10xi32>, memref<10x10xi32>) -> ()
    func.return
  }
  func.func @main() {
    "memref.alloca_scope"() ({
      func.call @_QMproblem_modPentry() : () -> ()
      "memref.alloca_scope.return"() : () -> ()
    }) : () -> ()
    func.return
  }
}

// CHECK:      builtin.module {
// CHECK-NEXT:   builtin.module {
// CHECK-NEXT:     func.func @_QMproblem_modPentry() {
// CHECK-NEXT:       %0 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<10x10xi32>
// CHECK-NEXT:       %1 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<10x10xi32>
// CHECK-NEXT:       %2 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<10x10xi32>
// CHECK-NEXT:       %3 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<10x10xi32>
// CHECK-NEXT:       func.call @host_entry(%0, %1, %3) : (memref<10x10xi32>, memref<10x10xi32>, memref<10x10xi32>) -> ()
// CHECK-NEXT:       "memref.copy"(%3, %2) : (memref<10x10xi32>, memref<10x10xi32>) -> ()
// CHECK-NEXT:       func.return
// CHECK-NEXT:     }
// CHECK-NEXT:     func.func @main() {
// CHECK-NEXT:       "memref.alloca_scope"() ({
// CHECK-NEXT:         func.call @_QMproblem_modPentry() : () -> ()
// CHECK-NEXT:         "memref.alloca_scope.return"() : () -> ()
// CHECK-NEXT:       }) : () -> ()
// CHECK-NEXT:       func.return
// CHECK-NEXT:     }
// CHECK-NEXT:     func.func private @host_entry(memref<10x10xi32>, memref<10x10xi32>, memref<10x10xi32>) -> ()
// CHECK-NEXT:   }
// CHECK-NEXT:   builtin.module attributes {kernel_type = "host", vis = "external"} {
// CHECK-NEXT:     func.func @host_entry(%0 : memref<10x10xi32>, %1 : memref<10x10xi32>, %2 : memref<10x10xi32>) {
// CHECK-NEXT:       %size0 = arith.constant 400 : i32
// CHECK-NEXT:       %size1 = arith.constant 400 : i32
// CHECK-NEXT:       %size_out = arith.constant 400 : i32
// CHECK-NEXT:       %prog = "tthost.create_program"() : () -> !tthost.program
// CHECK-NEXT:       %zero = arith.constant 0 : i32
// CHECK-NEXT:       %3 = arith.constant 1 : i32
// CHECK-NEXT:       %4 = arith.constant 16 : i32
// CHECK-NEXT:       %5 = "tthost.create_device"(%zero) : (i32) -> !tthost.device
// CHECK-NEXT:       %6 = "tthost.core"(%zero, %zero) : (i32, i32) -> !tthost.corecoord
// CHECK-NEXT:       %7 = "tthost.get_command_queue"(%5) : (!tthost.device) -> !tthost.command_queue
// CHECK-NEXT:       %8 = "tthost.create_dram_configuration"(%size0, %size0) : (i32, i32) -> !tthost.dram_buffer_config
// CHECK-NEXT:       %9 = "tthost.create_dram_configuration"(%size1, %size1) : (i32, i32) -> !tthost.dram_buffer_config
// CHECK-NEXT:       %10 = "tthost.create_dram_configuration"(%size_out, %size_out) : (i32, i32) -> !tthost.dram_buffer_config
// CHECK-NEXT:       %11 = "tthost.create_buffer"(%8) : (!tthost.dram_buffer_config) -> !tthost.buffer
// CHECK-NEXT:       %12 = "tthost.create_buffer"(%9) : (!tthost.dram_buffer_config) -> !tthost.buffer
// CHECK-NEXT:       %13 = "tthost.create_buffer"(%10) : (!tthost.dram_buffer_config) -> !tthost.buffer
// CHECK-NEXT:       %14 = arith.constant false
// CHECK-NEXT:       "tthost.enqueue_write_buffer"(%7, %11, %0, %14) : (!tthost.command_queue, !tthost.buffer, memref<10x10xi32>, i1) -> ()
// CHECK-NEXT:       "tthost.enqueue_write_buffer"(%7, %12, %1, %14) : (!tthost.command_queue, !tthost.buffer, memref<10x10xi32>, i1) -> ()
// CHECK-NEXT:       %15 = "tthost.create_cb_configuration"(%3, %size0, %zero) <{data_type = "int"}> : (i32, i32, i32) -> !tthost.circular_buffer_config
// CHECK-NEXT:       %16 = "tthost.create_cb_configuration"(%3, %size1, %3) <{data_type = "int"}> : (i32, i32, i32) -> !tthost.circular_buffer_config
// CHECK-NEXT:       %17 = "tthost.create_cb_configuration"(%3, %size_out, %4) <{data_type = "int"}> : (i32, i32, i32) -> !tthost.circular_buffer_config
// CHECK-NEXT:       %18 = "tthost.create_circular_buffer"(%prog, %6, %15) : (!tthost.program, !tthost.corecoord, !tthost.circular_buffer_config) -> !tthost.cb_handle
// CHECK-NEXT:       %19 = "tthost.create_circular_buffer"(%prog, %6, %16) : (!tthost.program, !tthost.corecoord, !tthost.circular_buffer_config) -> !tthost.cb_handle
// CHECK-NEXT:       %20 = "tthost.create_circular_buffer"(%prog, %6, %17) : (!tthost.program, !tthost.corecoord, !tthost.circular_buffer_config) -> !tthost.cb_handle
// CHECK-NEXT:       %reader_kernel = "tthost.create_kernel"(%prog, %6) <{kernel_name = "reader.cpp", riscv_core = #tthost.riscv_core<datamovement_0>, noc_id = #builtin.int<0>}> : (!tthost.program, !tthost.corecoord) -> !tthost.kernel
// CHECK-NEXT:       %writer_kernel = "tthost.create_kernel"(%prog, %6) <{kernel_name = "writer.cpp", riscv_core = #tthost.riscv_core<datamovement_1>, noc_id = #builtin.int<1>}> : (!tthost.program, !tthost.corecoord) -> !tthost.kernel
// CHECK-NEXT:       %compute_kernel = "tthost.create_compute_kernel"(%prog, %6) <{kernel_name = "compute.cpp", riscv_core = #tthost.riscv_core<compute>, math_fidelity = #tthost.math_fidelity<LoFi>, fp32_dest_acc_en = false, math_approx_mode = false}> : (!tthost.program, !tthost.corecoord) -> !tthost.kernel
// CHECK-NEXT:       %dram_in0_addr = "tthost.get_memory_address"(%11) : (!tthost.buffer) -> index
// CHECK-NEXT:       %dram_in1_addr = "tthost.get_memory_address"(%12) : (!tthost.buffer) -> index
// CHECK-NEXT:       %dram_out_addr = "tthost.get_memory_address"(%13) : (!tthost.buffer) -> index
// CHECK-NEXT:       "tthost.set_runtime_args"(%prog, %reader_kernel, %6, %zero, %zero, %dram_in0_addr, %dram_in1_addr, %size0, %size1) {operandSegmentSizes = array<i32: 1, 1, 1, 6>} : (!tthost.program, !tthost.kernel, !tthost.corecoord, i32, i32, index, index, i32, i32) -> ()
// CHECK-NEXT:       "tthost.set_runtime_args"(%prog, %compute_kernel, %6) {operandSegmentSizes = array<i32: 1, 1, 1, 0>} : (!tthost.program, !tthost.kernel, !tthost.corecoord) -> ()
// CHECK-NEXT:       "tthost.set_runtime_args"(%prog, %writer_kernel, %6, %zero, %dram_out_addr, %size_out) {operandSegmentSizes = array<i32: 1, 1, 1, 3>} : (!tthost.program, !tthost.kernel, !tthost.corecoord, i32, index, i32) -> ()
// CHECK-NEXT:       "tthost.enqueue_program"(%7, %prog, %14) : (!tthost.command_queue, !tthost.program, i1) -> ()
// CHECK-NEXT:       "tthost.finish"(%7) : (!tthost.command_queue) -> ()
// CHECK-NEXT:       "tthost.enqueue_read_buffer"(%7, %13, %2, %14) : (!tthost.command_queue, !tthost.buffer, memref<10x10xi32>, i1) -> ()
// CHECK-NEXT:       "tthost.close_device"(%5) : (!tthost.device) -> ()
// CHECK-NEXT:       func.return
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   builtin.module attributes {kernel_type = "data_in"} {
// CHECK-NEXT:     func.func @kernel_main(%bank_id0 : ui32, %bank_id1 : ui32, %mem_addr0 : ui32, %mem_addr1 : ui32, %size_bytes0 : ui32, %size_bytes1 : ui32) {
// CHECK-NEXT:       %0 = arith.constant 0 : i8
// CHECK-NEXT:       %1 = builtin.unrealized_conversion_cast %0 : i8 to ui8
// CHECK-NEXT:       %2 = arith.constant 0 : i32
// CHECK-NEXT:       %3 = arith.constant 1 : i32
// CHECK-NEXT:       %4 = "dm.get_noc_addr_from_bank_id"(%bank_id0, %mem_addr0, %1) <{dram = true}> : (ui32, ui32, ui8) -> ui64
// CHECK-NEXT:       %5 = "dm.get_noc_addr_from_bank_id"(%bank_id1, %mem_addr1, %1) <{dram = true}> : (ui32, ui32, ui8) -> ui64
// CHECK-NEXT:       %6 = "cb.get_write_pointer"(%2) : (i32) -> ui32
// CHECK-NEXT:       %7 = "cb.get_write_pointer"(%3) : (i32) -> ui32
// CHECK-NEXT:       "cb.reserve_back"(%2, %3) : (i32, i32) -> ()
// CHECK-NEXT:       "dm.noc_async_read"(%4, %6, %size_bytes0) : (ui64, ui32, ui32) -> ()
// CHECK-NEXT:       "dm.noc_async_read_barrier"() : () -> ()
// CHECK-NEXT:       "cb.push_back"(%2, %3) : (i32, i32) -> ()
// CHECK-NEXT:       "cb.reserve_back"(%3, %3) : (i32, i32) -> ()
// CHECK-NEXT:       "dm.noc_async_read"(%5, %7, %size_bytes1) : (ui64, ui32, ui32) -> ()
// CHECK-NEXT:       "dm.noc_async_read_barrier"() : () -> ()
// CHECK-NEXT:       "cb.push_back"(%3, %3) : (i32, i32) -> ()
// CHECK-NEXT:       func.return
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   builtin.module attributes {kernel_type = "compute"} {
// CHECK-NEXT:     func.func @kernel_main() {
// CHECK-NEXT:       %0 = arith.constant 0 : i32
// CHECK-NEXT:       %1 = arith.constant 1 : i32
// CHECK-NEXT:       %2 = arith.constant 16 : i32
// CHECK-NEXT:       %3 = builtin.unrealized_conversion_cast %0 : i32 to ui32
// CHECK-NEXT:       %4 = builtin.unrealized_conversion_cast %1 : i32 to ui32
// CHECK-NEXT:       %5 = builtin.unrealized_conversion_cast %2 : i32 to ui32
// CHECK-NEXT:       "comp.binary_op_init_common"(%3, %4, %5) : (ui32, ui32, ui32) -> ()
// CHECK-NEXT:       "comp.mm_init"(%3, %4, %3, %3) : (ui32, ui32, ui32, ui32) -> ()
// CHECK-NEXT:       "cb.wait_front"(%0, %1) : (i32, i32) -> ()
// CHECK-NEXT:       "cb.wait_front"(%1, %1) : (i32, i32) -> ()
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
