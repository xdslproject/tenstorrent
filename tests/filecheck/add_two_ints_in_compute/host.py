# RUN: python3.13 %s | filecheck %s
from tests import run_from_examples

run_from_examples(__file__)

# CHECK:      builtin.module {
# CHECK-NEXT:   builtin.module attributes {kernel_type = "host"} {
# CHECK-NEXT:     func.func @main() -> i32 {
# CHECK-NEXT:       %0 = arith.constant 0 : i32
# CHECK-NEXT:       %1 = "tthost.create_device"(%0) : (i32) -> !tthost.device
# CHECK-NEXT:       %device = memref.alloc() : memref<!tthost.device>
# CHECK-NEXT:       memref.store %1, %device[] : memref<!tthost.device>
# CHECK-NEXT:       %2 = memref.load %device[] : memref<!tthost.device>
# CHECK-NEXT:       %3 = "tthost.get_command_queue"(%2) : (!tthost.device) -> !tthost.command_queue
# CHECK-NEXT:       %cq = memref.alloc() : memref<!tthost.command_queue>
# CHECK-NEXT:       memref.store %3, %cq[] : memref<!tthost.command_queue>
# CHECK-NEXT:       %4 = "tthost.create_program"() : () -> !tthost.program
# CHECK-NEXT:       %program = memref.alloc() : memref<!tthost.program>
# CHECK-NEXT:       memref.store %4, %program[] : memref<!tthost.program>
# CHECK-NEXT:       %5 = arith.constant 0 : i32
# CHECK-NEXT:       %6 = arith.constant 0 : i32
# CHECK-NEXT:       %7 = "tthost.core"(%5, %6) : (i32, i32) -> !tthost.corecoord
# CHECK-NEXT:       %core = memref.alloc() : memref<!tthost.corecoord>
# CHECK-NEXT:       memref.store %7, %core[] : memref<!tthost.corecoord>
# CHECK-NEXT:       %8 = arith.constant 4 : i32
# CHECK-NEXT:       %dt_size = memref.alloc() : memref<i32>
# CHECK-NEXT:       memref.store %8, %dt_size[] : memref<i32>
# CHECK-NEXT:       %9 = memref.load %dt_size[] : memref<i32>
# CHECK-NEXT:       %10 = arith.constant 100 : i32
# CHECK-NEXT:       %11 = arith.muli %9, %10 : i32
# CHECK-NEXT:       %single_tile_size = memref.alloc() : memref<i32>
# CHECK-NEXT:       memref.store %11, %single_tile_size[] : memref<i32>
# CHECK-NEXT:       %12 = memref.load %single_tile_size[] : memref<i32>
# CHECK-NEXT:       %13 = memref.load %single_tile_size[] : memref<i32>
# CHECK-NEXT:       %14 = "tthost.create_dram_configuration"(%12, %13) : (i32, i32) -> !tthost.dram_buffer_config
# CHECK-NEXT:       %dram_config = memref.alloc() : memref<!tthost.dram_buffer_config>
# CHECK-NEXT:       memref.store %14, %dram_config[] : memref<!tthost.dram_buffer_config>
# CHECK-NEXT:       %15 = memref.load %dram_config[] : memref<!tthost.dram_buffer_config>
# CHECK-NEXT:       %16 = "tthost.create_buffer"(%15) : (!tthost.dram_buffer_config) -> !tthost.buffer
# CHECK-NEXT:       %src0_dram_buffer = memref.alloc() : memref<!tthost.buffer>
# CHECK-NEXT:       memref.store %16, %src0_dram_buffer[] : memref<!tthost.buffer>
# CHECK-NEXT:       %17 = memref.load %dram_config[] : memref<!tthost.dram_buffer_config>
# CHECK-NEXT:       %18 = "tthost.create_buffer"(%17) : (!tthost.dram_buffer_config) -> !tthost.buffer
# CHECK-NEXT:       %src1_dram_buffer = memref.alloc() : memref<!tthost.buffer>
# CHECK-NEXT:       memref.store %18, %src1_dram_buffer[] : memref<!tthost.buffer>
# CHECK-NEXT:       %19 = memref.load %dram_config[] : memref<!tthost.dram_buffer_config>
# CHECK-NEXT:       %20 = "tthost.create_buffer"(%19) : (!tthost.dram_buffer_config) -> !tthost.buffer
# CHECK-NEXT:       %dest_dram_buffer = memref.alloc() : memref<!tthost.buffer>
# CHECK-NEXT:       memref.store %20, %dest_dram_buffer[] : memref<!tthost.buffer>
# CHECK-NEXT:       %21 = arith.constant 1 : i32
# CHECK-NEXT:       %22 = arith.constant 400 : i32
# CHECK-NEXT:       %23 = arith.constant 0 : i32
# CHECK-NEXT:       %24 = "tthost.create_cb_configuration"(%21, %22, %23) <{data_type = "int"}> : (i32, i32, i32) -> !tthost.circular_buffer_config
# CHECK-NEXT:       %cb0_config = memref.alloc() : memref<!tthost.circular_buffer_config>
# CHECK-NEXT:       memref.store %24, %cb0_config[] : memref<!tthost.circular_buffer_config>
# CHECK-NEXT:       %25 = arith.constant 1 : i32
# CHECK-NEXT:       %26 = arith.constant 400 : i32
# CHECK-NEXT:       %27 = arith.constant 1 : i32
# CHECK-NEXT:       %28 = "tthost.create_cb_configuration"(%25, %26, %27) <{data_type = "int"}> : (i32, i32, i32) -> !tthost.circular_buffer_config
# CHECK-NEXT:       %cb1_config = memref.alloc() : memref<!tthost.circular_buffer_config>
# CHECK-NEXT:       memref.store %28, %cb1_config[] : memref<!tthost.circular_buffer_config>
# CHECK-NEXT:       %29 = arith.constant 1 : i32
# CHECK-NEXT:       %30 = arith.constant 400 : i32
# CHECK-NEXT:       %31 = arith.constant 2 : i32
# CHECK-NEXT:       %32 = "tthost.create_cb_configuration"(%29, %30, %31) <{data_type = "int"}> : (i32, i32, i32) -> !tthost.circular_buffer_config
# CHECK-NEXT:       %cb2_config = memref.alloc() : memref<!tthost.circular_buffer_config>
# CHECK-NEXT:       memref.store %32, %cb2_config[] : memref<!tthost.circular_buffer_config>
# CHECK-NEXT:       %33 = memref.load %program[] : memref<!tthost.program>
# CHECK-NEXT:       %34 = memref.load %core[] : memref<!tthost.corecoord>
# CHECK-NEXT:       %35 = memref.load %cb0_config[] : memref<!tthost.circular_buffer_config>
# CHECK-NEXT:       %36 = "tthost.create_circular_buffer"(%33, %34, %35) : (!tthost.program, !tthost.corecoord, !tthost.circular_buffer_config) -> !tthost.cb_handle
# CHECK-NEXT:       %cb0 = memref.alloc() : memref<!tthost.cb_handle>
# CHECK-NEXT:       memref.store %36, %cb0[] : memref<!tthost.cb_handle>
# CHECK-NEXT:       %37 = memref.load %program[] : memref<!tthost.program>
# CHECK-NEXT:       %38 = memref.load %core[] : memref<!tthost.corecoord>
# CHECK-NEXT:       %39 = memref.load %cb1_config[] : memref<!tthost.circular_buffer_config>
# CHECK-NEXT:       %40 = "tthost.create_circular_buffer"(%37, %38, %39) : (!tthost.program, !tthost.corecoord, !tthost.circular_buffer_config) -> !tthost.cb_handle
# CHECK-NEXT:       %cb1 = memref.alloc() : memref<!tthost.cb_handle>
# CHECK-NEXT:       memref.store %40, %cb1[] : memref<!tthost.cb_handle>
# CHECK-NEXT:       %41 = memref.load %program[] : memref<!tthost.program>
# CHECK-NEXT:       %42 = memref.load %core[] : memref<!tthost.corecoord>
# CHECK-NEXT:       %43 = memref.load %cb2_config[] : memref<!tthost.circular_buffer_config>
# CHECK-NEXT:       %44 = "tthost.create_circular_buffer"(%41, %42, %43) : (!tthost.program, !tthost.corecoord, !tthost.circular_buffer_config) -> !tthost.cb_handle
# CHECK-NEXT:       %cb2 = memref.alloc() : memref<!tthost.cb_handle>
# CHECK-NEXT:       memref.store %44, %cb2[] : memref<!tthost.cb_handle>
# CHECK-NEXT:       %45 = arith.constant 100 : i32
# CHECK-NEXT:       %host_src0 = memref.alloc() : memref<100xi32>
# CHECK-NEXT:       %46 = arith.constant 100 : i32
# CHECK-NEXT:       %host_src1 = memref.alloc() : memref<100xi32>
# CHECK-NEXT:       %47 = arith.constant 100 : i32
# CHECK-NEXT:       %host_dst = memref.alloc() : memref<100xi32>
# CHECK-NEXT:       %48 = arith.constant 0 : i32
# CHECK-NEXT:       %49 = arith.constant 100 : i32
# CHECK-NEXT:       %50 = arith.constant 1 : i32
# CHECK-NEXT:       %i = memref.alloc() : memref<i32>
# CHECK-NEXT:       scf.for %51 = %48 to %49 step %50  : i32 {
# CHECK-NEXT:         memref.store %51, %i[] : memref<i32>
# CHECK-NEXT:         %52 = memref.load %i[] : memref<i32>
# CHECK-NEXT:         %53 = memref.load %i[] : memref<i32>
# CHECK-NEXT:         %54 = arith.index_cast %53 : i32 to index
# CHECK-NEXT:         memref.store %52, %host_src0[%54] : memref<100xi32>
# CHECK-NEXT:         %55 = arith.constant 100 : i32
# CHECK-NEXT:         %56 = memref.load %i[] : memref<i32>
# CHECK-NEXT:         %57 = arith.subi %55, %56 : i32
# CHECK-NEXT:         %58 = memref.load %i[] : memref<i32>
# CHECK-NEXT:         %59 = arith.index_cast %58 : i32 to index
# CHECK-NEXT:         memref.store %57, %host_src1[%59] : memref<100xi32>
# CHECK-NEXT:       }
# CHECK-NEXT:       %60 = memref.load %program[] : memref<!tthost.program>
# CHECK-NEXT:       %61 = memref.load %core[] : memref<!tthost.corecoord>
# CHECK-NEXT:       %62 = "tthost.create_kernel"(%60, %61) <{kernel_name = "reader_kernel_kernel.cpp", riscv_core = #tthost.riscv_core<datamovement_0>, noc_id = #builtin.int<0>}> : (!tthost.program, !tthost.corecoord) -> !tthost.kernel
# CHECK-NEXT:       %kernel_din = memref.alloc() : memref<!tthost.kernel>
# CHECK-NEXT:       memref.store %62, %kernel_din[] : memref<!tthost.kernel>
# CHECK-NEXT:       %63 = memref.load %program[] : memref<!tthost.program>
# CHECK-NEXT:       %64 = memref.load %core[] : memref<!tthost.corecoord>
# CHECK-NEXT:       %65 = "tthost.create_kernel"(%63, %64) <{kernel_name = "writer_kernel_kernel.cpp", riscv_core = #tthost.riscv_core<datamovement_1>, noc_id = #builtin.int<1>}> : (!tthost.program, !tthost.corecoord) -> !tthost.kernel
# CHECK-NEXT:       %kernel_dout = memref.alloc() : memref<!tthost.kernel>
# CHECK-NEXT:       memref.store %65, %kernel_dout[] : memref<!tthost.kernel>
# CHECK-NEXT:       %66 = memref.load %program[] : memref<!tthost.program>
# CHECK-NEXT:       %67 = memref.load %core[] : memref<!tthost.corecoord>
# CHECK-NEXT:       %68 = "tthost.create_compute_kernel"(%66, %67) <{kernel_name = "add_two_int_tiles_kernel.cpp", riscv_core = #tthost.riscv_core<compute>, math_fidelity = #tthost.math_fidelity<HiFi4>, fp32_dest_acc_en = false, math_approx_mode = false}> : (!tthost.program, !tthost.corecoord) -> !tthost.kernel
# CHECK-NEXT:       %kernel_comp = memref.alloc() : memref<!tthost.kernel>
# CHECK-NEXT:       memref.store %68, %kernel_comp[] : memref<!tthost.kernel>
# CHECK-NEXT:       %69 = memref.load %cq[] : memref<!tthost.command_queue>
# CHECK-NEXT:       %70 = memref.load %src0_dram_buffer[] : memref<!tthost.buffer>
# CHECK-NEXT:       %71 = arith.constant false
# CHECK-NEXT:       "tthost.enqueue_write_buffer"(%69, %70, %host_src0, %71) : (!tthost.command_queue, !tthost.buffer, memref<100xi32>, i1) -> ()
# CHECK-NEXT:       %72 = memref.load %cq[] : memref<!tthost.command_queue>
# CHECK-NEXT:       %73 = memref.load %src1_dram_buffer[] : memref<!tthost.buffer>
# CHECK-NEXT:       %74 = arith.constant false
# CHECK-NEXT:       "tthost.enqueue_write_buffer"(%72, %73, %host_src1, %74) : (!tthost.command_queue, !tthost.buffer, memref<100xi32>, i1) -> ()
# CHECK-NEXT:       %75 = memref.load %program[] : memref<!tthost.program>
# CHECK-NEXT:       %76 = memref.load %kernel_din[] : memref<!tthost.kernel>
# CHECK-NEXT:       %77 = memref.load %core[] : memref<!tthost.corecoord>
# CHECK-NEXT:       %78 = memref.load %src0_dram_buffer[] : memref<!tthost.buffer>
# CHECK-NEXT:       %79 = "tthost.get_memory_address"(%78) : (!tthost.buffer) -> index
# CHECK-NEXT:       %80 = memref.load %src1_dram_buffer[] : memref<!tthost.buffer>
# CHECK-NEXT:       %81 = "tthost.get_memory_address"(%80) : (!tthost.buffer) -> index
# CHECK-NEXT:       %82 = arith.constant 0 : i32
# CHECK-NEXT:       %83 = arith.constant 0 : i32
# CHECK-NEXT:       "tthost.set_runtime_args"(%75, %76, %77, %79, %81, %82, %83) {operandSegmentSizes = array<i32: 1, 1, 1, 4>} : (!tthost.program, !tthost.kernel, !tthost.corecoord, index, index, i32, i32) -> ()
# CHECK-NEXT:       %84 = memref.load %program[] : memref<!tthost.program>
# CHECK-NEXT:       %85 = memref.load %kernel_comp[] : memref<!tthost.kernel>
# CHECK-NEXT:       %86 = memref.load %core[] : memref<!tthost.corecoord>
# CHECK-NEXT:       "tthost.set_runtime_args"(%84, %85, %86) {operandSegmentSizes = array<i32: 1, 1, 1, 0>} : (!tthost.program, !tthost.kernel, !tthost.corecoord) -> ()
# CHECK-NEXT:       %87 = memref.load %program[] : memref<!tthost.program>
# CHECK-NEXT:       %88 = memref.load %kernel_dout[] : memref<!tthost.kernel>
# CHECK-NEXT:       %89 = memref.load %core[] : memref<!tthost.corecoord>
# CHECK-NEXT:       %90 = memref.load %dest_dram_buffer[] : memref<!tthost.buffer>
# CHECK-NEXT:       %91 = "tthost.get_memory_address"(%90) : (!tthost.buffer) -> index
# CHECK-NEXT:       %92 = arith.constant 0 : i32
# CHECK-NEXT:       "tthost.set_runtime_args"(%87, %88, %89, %91, %92) {operandSegmentSizes = array<i32: 1, 1, 1, 2>} : (!tthost.program, !tthost.kernel, !tthost.corecoord, index, i32) -> ()
# CHECK-NEXT:       %93 = memref.load %cq[] : memref<!tthost.command_queue>
# CHECK-NEXT:       %94 = memref.load %program[] : memref<!tthost.program>
# CHECK-NEXT:       %95 = arith.constant false
# CHECK-NEXT:       "tthost.enqueue_program"(%93, %94, %95) : (!tthost.command_queue, !tthost.program, i1) -> ()
# CHECK-NEXT:       %96 = memref.load %cq[] : memref<!tthost.command_queue>
# CHECK-NEXT:       "tthost.finish"(%96) : (!tthost.command_queue) -> ()
# CHECK-NEXT:       %97 = memref.load %cq[] : memref<!tthost.command_queue>
# CHECK-NEXT:       %98 = memref.load %dest_dram_buffer[] : memref<!tthost.buffer>
# CHECK-NEXT:       %99 = arith.constant true
# CHECK-NEXT:       "tthost.enqueue_read_buffer"(%97, %98, %host_dst, %99) : (!tthost.command_queue, !tthost.buffer, memref<100xi32>, i1) -> ()
# CHECK-NEXT:       %100 = memref.load %device[] : memref<!tthost.device>
# CHECK-NEXT:       "tthost.close_device"(%100) : (!tthost.device) -> ()
# CHECK-NEXT:       %101 = arith.constant 0 : i32
# CHECK-NEXT:       func.return %101 : i32
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT: }
