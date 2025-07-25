// RUN: python3.13 tenstorrent/tools/tt-opt %s -t tt-metalium | filecheck %s

builtin.module {
  builtin.module attributes {kernel_type = "host"} {
    func.func @main() -> i32 {
      %0 = arith.constant 0 : i32
      %1 = "tthost.create_device"(%0) : (i32) -> !tthost.device
      %device = memref.alloc() : memref<!tthost.device>
      memref.store %1, %device[] : memref<!tthost.device>
      %2 = memref.load %device[] : memref<!tthost.device>
      %3 = "tthost.get_command_queue"(%2) : (!tthost.device) -> !tthost.command_queue
      %cq = memref.alloc() : memref<!tthost.command_queue>
      memref.store %3, %cq[] : memref<!tthost.command_queue>
      %4 = "tthost.create_program"() : () -> !tthost.program
      %program = memref.alloc() : memref<!tthost.program>
      memref.store %4, %program[] : memref<!tthost.program>
      %5 = arith.constant 0 : i32
      %6 = arith.constant 0 : i32
      %7 = "tthost.core"(%5, %6) : (i32, i32) -> !tthost.corecoord
      %core = memref.alloc() : memref<!tthost.corecoord>
      memref.store %7, %core[] : memref<!tthost.corecoord>
      %8 = arith.constant 4 : i32
      %dt_size = memref.alloc() : memref<i32>
      memref.store %8, %dt_size[] : memref<i32>
      %9 = memref.load %dt_size[] : memref<i32>
      %10 = arith.constant 100 : i32
      %11 = arith.muli %9, %10 : i32
      %single_tile_size = memref.alloc() : memref<i32>
      memref.store %11, %single_tile_size[] : memref<i32>
      %12 = memref.load %single_tile_size[] : memref<i32>
      %13 = memref.load %single_tile_size[] : memref<i32>
      %14 = "tthost.create_dram_configuration"(%12, %13) : (i32, i32) -> !tthost.dram_buffer_config
      %dram_config = memref.alloc() : memref<!tthost.dram_buffer_config>
      memref.store %14, %dram_config[] : memref<!tthost.dram_buffer_config>
      %15 = memref.load %dram_config[] : memref<!tthost.dram_buffer_config>
      %16 = "tthost.create_buffer"(%15) : (!tthost.dram_buffer_config) -> !tthost.buffer
      %src0_dram_buffer = memref.alloc() : memref<!tthost.buffer>
      memref.store %16, %src0_dram_buffer[] : memref<!tthost.buffer>
      %17 = memref.load %dram_config[] : memref<!tthost.dram_buffer_config>
      %18 = "tthost.create_buffer"(%17) : (!tthost.dram_buffer_config) -> !tthost.buffer
      %src1_dram_buffer = memref.alloc() : memref<!tthost.buffer>
      memref.store %18, %src1_dram_buffer[] : memref<!tthost.buffer>
      %19 = memref.load %dram_config[] : memref<!tthost.dram_buffer_config>
      %20 = "tthost.create_buffer"(%19) : (!tthost.dram_buffer_config) -> !tthost.buffer
      %dest_dram_buffer = memref.alloc() : memref<!tthost.buffer>
      memref.store %20, %dest_dram_buffer[] : memref<!tthost.buffer>
      %21 = arith.constant 1 : i32
      %22 = arith.constant 400 : i32
      %23 = arith.constant 0 : i32
      %24 = "tthost.create_cb_configuration"(%21, %22, %23) <{data_type = "int"}> : (i32, i32, i32) -> !tthost.circular_buffer_config
      %cb0_config = memref.alloc() : memref<!tthost.circular_buffer_config>
      memref.store %24, %cb0_config[] : memref<!tthost.circular_buffer_config>
      %25 = arith.constant 1 : i32
      %26 = arith.constant 400 : i32
      %27 = arith.constant 1 : i32
      %28 = "tthost.create_cb_configuration"(%25, %26, %27) <{data_type = "int"}> : (i32, i32, i32) -> !tthost.circular_buffer_config
      %cb1_config = memref.alloc() : memref<!tthost.circular_buffer_config>
      memref.store %28, %cb1_config[] : memref<!tthost.circular_buffer_config>
      %29 = arith.constant 1 : i32
      %30 = arith.constant 400 : i32
      %31 = arith.constant 2 : i32
      %32 = "tthost.create_cb_configuration"(%29, %30, %31) <{data_type = "int"}> : (i32, i32, i32) -> !tthost.circular_buffer_config
      %cb2_config = memref.alloc() : memref<!tthost.circular_buffer_config>
      memref.store %32, %cb2_config[] : memref<!tthost.circular_buffer_config>
      %33 = memref.load %program[] : memref<!tthost.program>
      %34 = memref.load %core[] : memref<!tthost.corecoord>
      %35 = memref.load %cb0_config[] : memref<!tthost.circular_buffer_config>
      %36 = "tthost.create_circular_buffer"(%33, %34, %35) : (!tthost.program, !tthost.corecoord, !tthost.circular_buffer_config) -> !tthost.cb_handle
      %cb0 = memref.alloc() : memref<!tthost.cb_handle>
      memref.store %36, %cb0[] : memref<!tthost.cb_handle>
      %37 = memref.load %program[] : memref<!tthost.program>
      %38 = memref.load %core[] : memref<!tthost.corecoord>
      %39 = memref.load %cb1_config[] : memref<!tthost.circular_buffer_config>
      %40 = "tthost.create_circular_buffer"(%37, %38, %39) : (!tthost.program, !tthost.corecoord, !tthost.circular_buffer_config) -> !tthost.cb_handle
      %cb1 = memref.alloc() : memref<!tthost.cb_handle>
      memref.store %40, %cb1[] : memref<!tthost.cb_handle>
      %41 = memref.load %program[] : memref<!tthost.program>
      %42 = memref.load %core[] : memref<!tthost.corecoord>
      %43 = memref.load %cb2_config[] : memref<!tthost.circular_buffer_config>
      %44 = "tthost.create_circular_buffer"(%41, %42, %43) : (!tthost.program, !tthost.corecoord, !tthost.circular_buffer_config) -> !tthost.cb_handle
      %cb2 = memref.alloc() : memref<!tthost.cb_handle>
      memref.store %44, %cb2[] : memref<!tthost.cb_handle>
      %45 = arith.constant 100 : i32
      %host_src0 = memref.alloc() : memref<100xi32>
      %46 = arith.constant 100 : i32
      %host_src1 = memref.alloc() : memref<100xi32>
      %47 = arith.constant 100 : i32
      %host_dst = memref.alloc() : memref<100xi32>
      %48 = arith.constant 0 : i32
      %49 = arith.constant 100 : i32
      %50 = arith.constant 1 : i32
      %i = memref.alloc() : memref<i32>
      scf.for %51 = %48 to %49 step %50  : i32 {
        memref.store %51, %i[] : memref<i32>
        %52 = memref.load %i[] : memref<i32>
        %53 = memref.load %i[] : memref<i32>
        %54 = arith.index_cast %53 : i32 to index
        memref.store %52, %host_src0[%54] : memref<100xi32>
        %55 = arith.constant 100 : i32
        %56 = memref.load %i[] : memref<i32>
        %57 = arith.subi %55, %56 : i32
        %58 = memref.load %i[] : memref<i32>
        %59 = arith.index_cast %58 : i32 to index
        memref.store %57, %host_src1[%59] : memref<100xi32>
      }
      %60 = memref.load %program[] : memref<!tthost.program>
      %61 = memref.load %core[] : memref<!tthost.corecoord>
      %62 = "tthost.create_kernel"(%60, %61) <{kernel_name = "reader.cpp", riscv_core = #tthost.riscv_core<datamovement_0>, noc_id = #builtin.int<0>}> : (!tthost.program, !tthost.corecoord) -> !tthost.kernel
      %kernel_din = memref.alloc() : memref<!tthost.kernel>
      memref.store %62, %kernel_din[] : memref<!tthost.kernel>
      %63 = memref.load %program[] : memref<!tthost.program>
      %64 = memref.load %core[] : memref<!tthost.corecoord>
      %65 = "tthost.create_kernel"(%63, %64) <{kernel_name = "writer.cpp", riscv_core = #tthost.riscv_core<datamovement_1>, noc_id = #builtin.int<1>}> : (!tthost.program, !tthost.corecoord) -> !tthost.kernel
      %kernel_dout = memref.alloc() : memref<!tthost.kernel>
      memref.store %65, %kernel_dout[] : memref<!tthost.kernel>
      %66 = memref.load %program[] : memref<!tthost.program>
      %67 = memref.load %core[] : memref<!tthost.corecoord>
      %68 = "tthost.create_compute_kernel"(%66, %67) <{kernel_name = "compute.cpp", riscv_core = #tthost.riscv_core<compute>, math_fidelity = #tthost.math_fidelity<HiFi4>, fp32_dest_acc_en = false, math_approx_mode = false}> : (!tthost.program, !tthost.corecoord) -> !tthost.kernel
      %kernel_comp = memref.alloc() : memref<!tthost.kernel>
      memref.store %68, %kernel_comp[] : memref<!tthost.kernel>
      %69 = memref.load %cq[] : memref<!tthost.command_queue>
      %70 = memref.load %src0_dram_buffer[] : memref<!tthost.buffer>
      %71 = arith.constant false
      "tthost.enqueue_write_buffer"(%69, %70, %host_src0, %71) : (!tthost.command_queue, !tthost.buffer, memref<100xi32>, i1) -> ()
      %72 = memref.load %cq[] : memref<!tthost.command_queue>
      %73 = memref.load %src1_dram_buffer[] : memref<!tthost.buffer>
      %74 = arith.constant false
      "tthost.enqueue_write_buffer"(%72, %73, %host_src1, %74) : (!tthost.command_queue, !tthost.buffer, memref<100xi32>, i1) -> ()
      %75 = memref.load %program[] : memref<!tthost.program>
      %76 = memref.load %kernel_din[] : memref<!tthost.kernel>
      %77 = memref.load %core[] : memref<!tthost.corecoord>
      %78 = memref.load %src0_dram_buffer[] : memref<!tthost.buffer>
      %79 = "tthost.get_memory_address"(%78) : (!tthost.buffer) -> index
      %80 = memref.load %src1_dram_buffer[] : memref<!tthost.buffer>
      %81 = "tthost.get_memory_address"(%80) : (!tthost.buffer) -> index
      %82 = arith.constant 0 : i32
      %83 = arith.constant 0 : i32
      "tthost.set_runtime_args"(%75, %76, %77, %79, %81, %82, %83) {operandSegmentSizes = array<i32: 1, 1, 1, 4>} : (!tthost.program, !tthost.kernel, !tthost.corecoord, index, index, i32, i32) -> ()
      %84 = memref.load %program[] : memref<!tthost.program>
      %85 = memref.load %kernel_comp[] : memref<!tthost.kernel>
      %86 = memref.load %core[] : memref<!tthost.corecoord>
      "tthost.set_runtime_args"(%84, %85, %86) {operandSegmentSizes = array<i32: 1, 1, 1, 0>} : (!tthost.program, !tthost.kernel, !tthost.corecoord) -> ()
      %87 = memref.load %program[] : memref<!tthost.program>
      %88 = memref.load %kernel_dout[] : memref<!tthost.kernel>
      %89 = memref.load %core[] : memref<!tthost.corecoord>
      %90 = memref.load %dest_dram_buffer[] : memref<!tthost.buffer>
      %91 = "tthost.get_memory_address"(%90) : (!tthost.buffer) -> index
      %92 = arith.constant 0 : i32
      "tthost.set_runtime_args"(%87, %88, %89, %91, %92) {operandSegmentSizes = array<i32: 1, 1, 1, 2>} : (!tthost.program, !tthost.kernel, !tthost.corecoord, index, i32) -> ()
      %93 = memref.load %cq[] : memref<!tthost.command_queue>
      %94 = memref.load %program[] : memref<!tthost.program>
      %95 = arith.constant false
      "tthost.enqueue_program"(%93, %94, %95) : (!tthost.command_queue, !tthost.program, i1) -> ()
      %96 = memref.load %cq[] : memref<!tthost.command_queue>
      "tthost.finish"(%96) : (!tthost.command_queue) -> ()
      %97 = memref.load %cq[] : memref<!tthost.command_queue>
      %98 = memref.load %dest_dram_buffer[] : memref<!tthost.buffer>
      %99 = arith.constant true
      "tthost.enqueue_read_buffer"(%97, %98, %host_dst, %99) : (!tthost.command_queue, !tthost.buffer, memref<100xi32>, i1) -> ()
      %100 = memref.load %device[] : memref<!tthost.device>
      "tthost.close_device"(%100) : (!tthost.device) -> ()
      %101 = arith.constant 0 : i32
      func.return %101 : i32
    }
  }
}


// CHECK:      std::int32_t main() {
// CHECK-NEXT:     IDevice* device_0 = CreateDevice(0);
// CHECK-NEXT:     IDevice* device = device_0;
// CHECK-NEXT:     CommandQueue & cq = device->command_queue();
// CHECK-NEXT:     Program program_1 = CreateProgram();
// CHECK-NEXT:     Program& program = program_1;
// CHECK-NEXT:     CoreCoord core = CoreCoord{0, 0};
// CHECK-NEXT:     std::int32_t dt_size = 4;
// CHECK-NEXT:     std::int32_t single_tile_size = dt_size * 100;
// CHECK-NEXT:     InterleavedBufferConfig dram_config {.device=device, .size=single_tile_size, .page_size=single_tile_size, .buffer_type = BufferType::DRAM};
// CHECK-NEXT:     std::shared_ptr<Buffer> buffer_2 = CreateBuffer(dram_config);
// CHECK-NEXT:     std::shared_ptr<Buffer> src0_dram_buffer = buffer_2;
// CHECK-NEXT:     std::shared_ptr<Buffer> buffer_3 = CreateBuffer(dram_config);
// CHECK-NEXT:     std::shared_ptr<Buffer> src1_dram_buffer = buffer_3;
// CHECK-NEXT:     std::shared_ptr<Buffer> buffer_4 = CreateBuffer(dram_config);
// CHECK-NEXT:     std::shared_ptr<Buffer> dest_dram_buffer = buffer_4;
// CHECK-NEXT:     CircularBufferConfig cb_config_5 = CircularBufferConfig(1 * 400, {{[{][{]}}0, tt::DataFormat::Int32{{[}][}]}}).set_page_size(0, 400);
// CHECK-NEXT:     CircularBufferConfig cb0_config = cb_config_5;
// CHECK-NEXT:     CircularBufferConfig cb_config_6 = CircularBufferConfig(1 * 400, {{[{][{]}}1, tt::DataFormat::Int32{{[}][}]}}).set_page_size(1, 400);
// CHECK-NEXT:     CircularBufferConfig cb1_config = cb_config_6;
// CHECK-NEXT:     CircularBufferConfig cb_config_7 = CircularBufferConfig(1 * 400, {{[{][{]}}2, tt::DataFormat::Int32{{[}][}]}}).set_page_size(2, 400);
// CHECK-NEXT:     CircularBufferConfig cb2_config = cb_config_7;
// CHECK-NEXT:     CBHandle cb_8 = tt_metal::CreateCircularBuffer(program, core, cb0_config);
// CHECK-NEXT:     CBHandle cb0 = cb_8;
// CHECK-NEXT:     CBHandle cb_9 = tt_metal::CreateCircularBuffer(program, core, cb1_config);
// CHECK-NEXT:     CBHandle cb1 = cb_9;
// CHECK-NEXT:     CBHandle cb_10 = tt_metal::CreateCircularBuffer(program, core, cb2_config);
// CHECK-NEXT:     CBHandle cb2 = cb_10;
// CHECK-NEXT:     std::int32_t * host_src0 = (std::int32_t*) malloc(sizeof(std::int32_t)*100);
// CHECK-NEXT:     std::int32_t * host_src1 = (std::int32_t*) malloc(sizeof(std::int32_t)*100);
// CHECK-NEXT:     std::int32_t * host_dst = (std::int32_t*) malloc(sizeof(std::int32_t)*100);
// CHECK-NEXT:     std::int32_t i;
// CHECK-NEXT:     for (i = 0; i < 100; i += 1) {
// CHECK-NEXT:         host_src0[i] = i;
// CHECK-NEXT:         host_src1[i] = 100 - i;
// CHECK-NEXT:     }
// CHECK-NEXT:     KernelHandle kernel_11 = CreateKernel(program, "reader.cpp", core, DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc=NOC::RISCV_0_default});
// CHECK-NEXT:     KernelHandle kernel_din = kernel_11;
// CHECK-NEXT:     KernelHandle kernel_12 = CreateKernel(program, "writer.cpp", core, DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc=NOC::RISCV_1_default});
// CHECK-NEXT:     KernelHandle kernel_dout = kernel_12;
// CHECK-NEXT:     KernelHandle kernel_13 = CreateKernel(program, "compute.cpp", core, ComputeConfig {.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = false, .math_approx_mode = false, .compile_args = {}});
// CHECK-NEXT:     KernelHandle kernel_comp = kernel_13;
// CHECK-NEXT:     EnqueueWriteBuffer(cq, src0_dram_buffer, host_src0, false);
// CHECK-NEXT:     EnqueueWriteBuffer(cq, src1_dram_buffer, host_src1, false);
// CHECK-NEXT:     SetRuntimeArgs(program, kernel_din, core, {src0_dram_buffer->address(), src1_dram_buffer->address(), 0, 0});
// CHECK-NEXT:     SetRuntimeArgs(program, kernel_comp, core, {});
// CHECK-NEXT:     SetRuntimeArgs(program, kernel_dout, core, {dest_dram_buffer->address(), 0});
// CHECK-NEXT:     EnqueueProgram(cq, program, false);
// CHECK-NEXT:     Finish(cq);
// CHECK-NEXT:     EnqueueReadBuffer(cq, dest_dram_buffer, host_dst, true);
// CHECK-NEXT:     CloseDevice(device);
// CHECK-NEXT:     free(host_src0);
// CHECK-NEXT:     free(host_src1);
// CHECK-NEXT:     free(host_dst);
// CHECK-NEXT:     return 0;
// CHECK-NEXT: }
