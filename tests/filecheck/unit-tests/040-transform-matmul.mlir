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
// CHECK-NEXT:       %3 = arith.constant 400 : i32
// CHECK-NEXT:       %4 = arith.constant 400 : i32
// CHECK-NEXT:       %5 = arith.constant 400 : i32
// CHECK-NEXT:       %6 = "tthost.create_program"() : () -> !tthost.program
// CHECK-NEXT:       %7 = arith.constant 0 : i32
// CHECK-NEXT:       %8 = arith.constant 1 : i32
// CHECK-NEXT:       %9 = arith.constant 16 : i32
// CHECK-NEXT:       %10 = "tthost.create_device"(%7) : (i32) -> !tthost.device
// CHECK-NEXT:       %11 = "tthost.core"(%7, %7) : (i32, i32) -> !tthost.corecoord
// CHECK-NEXT:       %12 = "tthost.get_command_queue"(%10) : (!tthost.device) -> !tthost.command_queue
// CHECK-NEXT:       %13 = "tthost.create_dram_configuration"(%3, %3) : (i32, i32) -> !tthost.dram_buffer_config
// CHECK-NEXT:       %14 = "tthost.create_dram_configuration"(%4, %4) : (i32, i32) -> !tthost.dram_buffer_config
// CHECK-NEXT:       %15 = "tthost.create_dram_configuration"(%5, %5) : (i32, i32) -> !tthost.dram_buffer_config
// CHECK-NEXT:       %16 = "tthost.create_buffer"(%13) : (!tthost.dram_buffer_config) -> !tthost.buffer
// CHECK-NEXT:       %17 = "tthost.create_buffer"(%14) : (!tthost.dram_buffer_config) -> !tthost.buffer
// CHECK-NEXT:       %18 = "tthost.create_buffer"(%15) : (!tthost.dram_buffer_config) -> !tthost.buffer
// CHECK-NEXT:       %19 = arith.constant false
// CHECK-NEXT:       "tthost.enqueue_write_buffer"(%12, %16, %0, %19) : (!tthost.command_queue, !tthost.buffer, memref<10x10xi32>, i1) -> ()
// CHECK-NEXT:       "tthost.enqueue_write_buffer"(%12, %17, %1, %19) : (!tthost.command_queue, !tthost.buffer, memref<10x10xi32>, i1) -> ()
// CHECK-NEXT:       %20 = "tthost.create_cb_configuration"(%8, %3, %7) <{data_type = "int"}> : (i32, i32, i32) -> !tthost.circular_buffer_config
// CHECK-NEXT:       %21 = "tthost.create_cb_configuration"(%8, %4, %8) <{data_type = "int"}> : (i32, i32, i32) -> !tthost.circular_buffer_config
// CHECK-NEXT:       %22 = "tthost.create_cb_configuration"(%8, %5, %9) <{data_type = "int"}> : (i32, i32, i32) -> !tthost.circular_buffer_config
// CHECK-NEXT:       %23 = "tthost.create_circular_buffer"(%6, %11, %20) : (!tthost.program, !tthost.corecoord, !tthost.circular_buffer_config) -> !tthost.cb_handle
// CHECK-NEXT:       %24 = "tthost.create_circular_buffer"(%6, %11, %21) : (!tthost.program, !tthost.corecoord, !tthost.circular_buffer_config) -> !tthost.cb_handle
// CHECK-NEXT:       %25 = "tthost.create_circular_buffer"(%6, %11, %22) : (!tthost.program, !tthost.corecoord, !tthost.circular_buffer_config) -> !tthost.cb_handle
// CHECK-NEXT:       %26 = "tthost.create_kernel"(%6, %11) <{kernel_name = "reader.cpp", riscv_core = #tthost.riscv_core<datamovement_0>, noc_id = #builtin.int<0>}> : (!tthost.program, !tthost.corecoord) -> !tthost.kernel
// CHECK-NEXT:       %27 = "tthost.create_kernel"(%6, %11) <{kernel_name = "writer.cpp", riscv_core = #tthost.riscv_core<datamovement_1>, noc_id = #builtin.int<1>}> : (!tthost.program, !tthost.corecoord) -> !tthost.kernel
// CHECK-NEXT:       %28 = "tthost.create_compute_kernel"(%6, %11) <{kernel_name = "compute.cpp", riscv_core = #tthost.riscv_core<compute>, math_fidelity = #tthost.math_fidelity<LoFi>, fp32_dest_acc_en = false, math_approx_mode = false}> : (!tthost.program, !tthost.corecoord) -> !tthost.kernel
// CHECK-NEXT:       %29 = "tthost.get_memory_address"(%16) : (!tthost.buffer) -> index
// CHECK-NEXT:       %30 = "tthost.get_memory_address"(%17) : (!tthost.buffer) -> index
// CHECK-NEXT:       %31 = "tthost.get_memory_address"(%18) : (!tthost.buffer) -> index
// CHECK-NEXT:       "tthost.set_runtime_args"(%6, %26, %11, %29, %30, %7, %7, %3, %4) {operandSegmentSizes = array<i32: 1, 1, 1, 6>} : (!tthost.program, !tthost.kernel, !tthost.corecoord, index, index, i32, i32, i32, i32) -> ()
// CHECK-NEXT:       "tthost.set_runtime_args"(%6, %28, %11) {operandSegmentSizes = array<i32: 1, 1, 1, 0>} : (!tthost.program, !tthost.kernel, !tthost.corecoord) -> ()
// CHECK-NEXT:       "tthost.set_runtime_args"(%6, %27, %11, %31, %7, %5) {operandSegmentSizes = array<i32: 1, 1, 1, 3>} : (!tthost.program, !tthost.kernel, !tthost.corecoord, index, i32, i32) -> ()
// CHECK-NEXT:       "tthost.enqueue_program"(%12, %6, %19) : (!tthost.command_queue, !tthost.program, i1) -> ()
// CHECK-NEXT:       "tthost.finish"(%12) : (!tthost.command_queue) -> ()
// CHECK-NEXT:       "tthost.enqueue_read_buffer"(%12, %18, %2, %19) : (!tthost.command_queue, !tthost.buffer, memref<10x10xi32>, i1) -> ()
// CHECK-NEXT:       "tthost.close_device"(%10) : (!tthost.device) -> ()
// CHECK-NEXT:       func.return
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   builtin.module attributes {kernel_type = "data_in"} {
// CHECK-NEXT:     func.func @kernel_main(%0 : ui32, %1 : ui32, %2 : ui32, %3 : ui32, %4 : ui32, %5 : ui32) {
// CHECK-NEXT:       %6 = arith.constant 0 : i8
// CHECK-NEXT:       %7 = builtin.unrealized_conversion_cast %6 : i8 to ui8
// CHECK-NEXT:       %8 = arith.constant 0 : i32
// CHECK-NEXT:       %9 = arith.constant 1 : i32
// CHECK-NEXT:       %10 = "dm.get_noc_addr_from_bank_id"(%0, %1, %7) <{dram = true}> : (ui32, ui32, ui8) -> ui64
// CHECK-NEXT:       %11 = "dm.get_noc_addr_from_bank_id"(%2, %3, %7) <{dram = true}> : (ui32, ui32, ui8) -> ui64
// CHECK-NEXT:       %12 = "cb.get_write_pointer"(%8) : (i32) -> ui32
// CHECK-NEXT:       %13 = "cb.get_write_pointer"(%9) : (i32) -> ui32
// CHECK-NEXT:       "cb.reserve_back"(%8, %9) : (i32, i32) -> ()
// CHECK-NEXT:       "dm.noc_async_read"(%10, %12, %4) : (ui64, ui32, ui32) -> ()
// CHECK-NEXT:       "dm.noc_async_read_barrier"() : () -> ()
// CHECK-NEXT:       "cb.push_back"(%8, %9) : (i32, i32) -> ()
// CHECK-NEXT:       "cb.reserve_back"(%9, %9) : (i32, i32) -> ()
// CHECK-NEXT:       "dm.noc_async_read"(%11, %13, %5) : (ui64, ui32, ui32) -> ()
// CHECK-NEXT:       "dm.noc_async_read_barrier"() : () -> ()
// CHECK-NEXT:       "cb.push_back"(%9, %9) : (i32, i32) -> ()
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
// CHECK-NEXT:     func.func @kernel_main(%0 : ui32, %1 : ui32, %2 : ui32) {
// CHECK-NEXT:       %3 = "dm.get_noc_addr_from_bank_id"(%0, %1) <{dram = true}> : (ui32, ui32) -> ui64
// CHECK-NEXT:       %4 = arith.constant 1 : i32
// CHECK-NEXT:       %5 = arith.constant 16 : i32
// CHECK-NEXT:       %6 = "cb.get_read_pointer"(%5) : (i32) -> ui32
// CHECK-NEXT:       "cb.wait_front"(%5, %4) : (i32, i32) -> ()
// CHECK-NEXT:       "dm.noc_async_write"(%6, %3, %2) : (ui32, ui64, ui32) -> ()
// CHECK-NEXT:       "dm.noc_async_write_barrier"() : () -> ()
// CHECK-NEXT:       "cb.pop_front"(%5, %4) : (i32, i32) -> ()
// CHECK-NEXT:       func.return
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
