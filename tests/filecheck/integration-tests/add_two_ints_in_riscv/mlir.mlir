// RUN: python3.13 tenstorrent/tools/tt-opt %s -t tt-metalium | filecheck %s

builtin.module {
  builtin.module attributes {kernel_type = "data_in"} {
    func.func @kernel_main(%0 : i32, %1 : i32, %2 : i32, %3 : i32, %4 : i32, %5 : i32) {
      %6 = "dm.get_noc_addr_from_bank_id"(%3, %0) <{dram = true}> : (i32, i32) -> i64
      %src0_dram_noc_addr = memref.alloc() : memref<i64>
      memref.store %6, %src0_dram_noc_addr[] : memref<i64>
      %7 = "dm.get_noc_addr_from_bank_id"(%4, %1) <{dram = true}> : (i32, i32) -> i64
      %src1_dram_noc_addr = memref.alloc() : memref<i64>
      memref.store %7, %src1_dram_noc_addr[] : memref<i64>
      %8 = "dm.get_noc_addr_from_bank_id"(%5, %2) <{dram = true}> : (i32, i32) -> i64
      %dst_dram_noc_addr = memref.alloc() : memref<i64>
      memref.store %8, %dst_dram_noc_addr[] : memref<i64>
      %9 = arith.constant 0 : i32
      %10 = "cb.get_write_pointer"(%9) : (i32) -> i32
      %l1_write_addr_in0 = memref.alloc() : memref<i32>
      memref.store %10, %l1_write_addr_in0[] : memref<i32>
      %11 = arith.constant 1 : i32
      %12 = "cb.get_write_pointer"(%11) : (i32) -> i32
      %l1_write_addr_in1 = memref.alloc() : memref<i32>
      memref.store %12, %l1_write_addr_in1[] : memref<i32>
      %13 = arith.constant 2 : i32
      %14 = "cb.get_write_pointer"(%13) : (i32) -> i32
      %l1_write_addr_in2 = memref.alloc() : memref<i32>
      memref.store %14, %l1_write_addr_in2[] : memref<i32>
      %15 = memref.load %src0_dram_noc_addr[] : memref<i64>
      %16 = memref.load %l1_write_addr_in0[] : memref<i32>
      %17 = arith.constant 400 : i32
      "dm.noc_async_read"(%15, %16, %17) : (i64, i32, i32) -> ()
      %18 = memref.load %src1_dram_noc_addr[] : memref<i64>
      %19 = memref.load %l1_write_addr_in1[] : memref<i32>
      %20 = arith.constant 400 : i32
      "dm.noc_async_read"(%18, %19, %20) : (i64, i32, i32) -> ()
      %21 = memref.load %l1_write_addr_in0[] : memref<i32>
      %src0_data = builtin.unrealized_conversion_cast %21 : i32 to memref<100xi32>
      %22 = memref.load %l1_write_addr_in1[] : memref<i32>
      %src1_data = builtin.unrealized_conversion_cast %22 : i32 to memref<100xi32>
      %23 = memref.load %l1_write_addr_in2[] : memref<i32>
      %dst_data = builtin.unrealized_conversion_cast %23 : i32 to memref<100xi32>
      "dm.noc_async_read_barrier"() : () -> ()
      %24 = arith.constant 0 : i32
      %25 = arith.constant 100 : i32
      %26 = arith.constant 1 : i32
      %x = memref.alloc() : memref<i32>
      scf.for %27 = %24 to %25 step %26  : i32 {
        memref.store %27, %x[] : memref<i32>
        %28 = memref.load %x[] : memref<i32>
        %29 = arith.index_cast %28 : i32 to index
        %30 = memref.load %src0_data[%29] : memref<100xi32>
        %31 = memref.load %x[] : memref<i32>
        %32 = arith.index_cast %31 : i32 to index
        %33 = memref.load %src1_data[%32] : memref<100xi32>
        %34 = arith.addi %30, %33 : i32
        %35 = memref.load %x[] : memref<i32>
        %36 = arith.index_cast %35 : i32 to index
        memref.store %34, %dst_data[%36] : memref<100xi32>
      }
      %37 = memref.load %l1_write_addr_in2[] : memref<i32>
      %38 = memref.load %dst_dram_noc_addr[] : memref<i64>
      %39 = arith.constant 400 : i32
      "dm.noc_async_write"(%37, %38, %39) : (i32, i64, i32) -> ()
      "dm.noc_async_write_barrier"() : () -> ()
      func.return
    }
  }
  builtin.module attributes {kernel_type = "host"} {
    func.func @main() -> i32 {
      %0 = arith.constant 0 : i32
      %1 = arith.constant 0 : i32
      %2 = "tthost.core"(%0, %1) : (i32, i32) -> !tthost.corecoord
      %core = memref.alloc() : memref<!tthost.corecoord>
      memref.store %2, %core[] : memref<!tthost.corecoord>
      %3 = arith.constant 4 : i32
      %4 = arith.constant 100 : i32
      %5 = arith.muli %3, %4 : i32
      %single_tile_size = memref.alloc() : memref<i32>
      memref.store %5, %single_tile_size[] : memref<i32>
      %6 = arith.constant 0 : i32
      %7 = "tthost.create_device"(%6) : (i32) -> !tthost.device
      %device = memref.alloc() : memref<!tthost.device>
      memref.store %7, %device[] : memref<!tthost.device>
      %8 = memref.load %device[] : memref<!tthost.device>
      %9 = "tthost.get_command_queue"(%8) : (!tthost.device) -> !tthost.command_queue
      %command_queue = memref.alloc() : memref<!tthost.command_queue>
      memref.store %9, %command_queue[] : memref<!tthost.command_queue>
      %10 = "tthost.create_program"() : () -> !tthost.program
      %program = memref.alloc() : memref<!tthost.program>
      memref.store %10, %program[] : memref<!tthost.program>
      %11 = memref.load %single_tile_size[] : memref<i32>
      %12 = memref.load %single_tile_size[] : memref<i32>
      %13 = "tthost.create_dram_configuration"(%11, %12) : (i32, i32) -> !tthost.dram_buffer_config
      %dram_configuration = memref.alloc() : memref<!tthost.dram_buffer_config>
      memref.store %13, %dram_configuration[] : memref<!tthost.dram_buffer_config>
      %14 = memref.load %dram_configuration[] : memref<!tthost.dram_buffer_config>
      %15 = "tthost.create_buffer"(%14) : (!tthost.dram_buffer_config) -> !tthost.buffer
      %src0_dram_buffer = memref.alloc() : memref<!tthost.buffer>
      memref.store %15, %src0_dram_buffer[] : memref<!tthost.buffer>
      %16 = memref.load %dram_configuration[] : memref<!tthost.dram_buffer_config>
      %17 = "tthost.create_buffer"(%16) : (!tthost.dram_buffer_config) -> !tthost.buffer
      %src1_dram_buffer = memref.alloc() : memref<!tthost.buffer>
      memref.store %17, %src1_dram_buffer[] : memref<!tthost.buffer>
      %18 = memref.load %dram_configuration[] : memref<!tthost.dram_buffer_config>
      %19 = "tthost.create_buffer"(%18) : (!tthost.dram_buffer_config) -> !tthost.buffer
      %dst_dram_buffer = memref.alloc() : memref<!tthost.buffer>
      memref.store %19, %dst_dram_buffer[] : memref<!tthost.buffer>
      %20 = arith.constant 1 : i32
      %21 = arith.constant 400 : i32
      %22 = arith.constant 0 : i32
      %23 = "tthost.create_cb_configuration"(%20, %21, %22) <{data_type = "int"}> : (i32, i32, i32) -> !tthost.circular_buffer_config
      %cb_0_config = memref.alloc() : memref<!tthost.circular_buffer_config>
      memref.store %23, %cb_0_config[] : memref<!tthost.circular_buffer_config>
      %24 = memref.load %program[] : memref<!tthost.program>
      %25 = memref.load %core[] : memref<!tthost.corecoord>
      %26 = memref.load %cb_0_config[] : memref<!tthost.circular_buffer_config>
      %27 = "tthost.create_circular_buffer"(%24, %25, %26) : (!tthost.program, !tthost.corecoord, !tthost.circular_buffer_config) -> !tthost.cb_handle
      %cb = memref.alloc() : memref<!tthost.cb_handle>
      memref.store %27, %cb[] : memref<!tthost.cb_handle>
      %28 = arith.constant 1 : i32
      %29 = arith.constant 400 : i32
      %30 = arith.constant 1 : i32
      %31 = "tthost.create_cb_configuration"(%28, %29, %30) <{data_type = "int"}> : (i32, i32, i32) -> !tthost.circular_buffer_config
      %cb_1_config = memref.alloc() : memref<!tthost.circular_buffer_config>
      memref.store %31, %cb_1_config[] : memref<!tthost.circular_buffer_config>
      %32 = memref.load %program[] : memref<!tthost.program>
      %33 = memref.load %core[] : memref<!tthost.corecoord>
      %34 = memref.load %cb_1_config[] : memref<!tthost.circular_buffer_config>
      %35 = "tthost.create_circular_buffer"(%32, %33, %34) : (!tthost.program, !tthost.corecoord, !tthost.circular_buffer_config) -> !tthost.cb_handle
      %cb_1 = memref.alloc() : memref<!tthost.cb_handle>
      memref.store %35, %cb_1[] : memref<!tthost.cb_handle>
      %36 = arith.constant 1 : i32
      %37 = arith.constant 400 : i32
      %38 = arith.constant 2 : i32
      %39 = "tthost.create_cb_configuration"(%36, %37, %38) <{data_type = "int"}> : (i32, i32, i32) -> !tthost.circular_buffer_config
      %cb_2_config = memref.alloc() : memref<!tthost.circular_buffer_config>
      memref.store %39, %cb_2_config[] : memref<!tthost.circular_buffer_config>
      %40 = memref.load %program[] : memref<!tthost.program>
      %41 = memref.load %core[] : memref<!tthost.corecoord>
      %42 = memref.load %cb_2_config[] : memref<!tthost.circular_buffer_config>
      %43 = "tthost.create_circular_buffer"(%40, %41, %42) : (!tthost.program, !tthost.corecoord, !tthost.circular_buffer_config) -> !tthost.cb_handle
      %cb_2 = memref.alloc() : memref<!tthost.cb_handle>
      memref.store %43, %cb_2[] : memref<!tthost.cb_handle>
      %44 = arith.constant 100 : i32
      %host_src0 = memref.alloc() : memref<100xi32>
      %45 = arith.constant 100 : i32
      %host_src1 = memref.alloc() : memref<100xi32>
      %46 = arith.constant 100 : i32
      %host_dst = memref.alloc() : memref<100xi32>
      %47 = arith.constant 0 : i32
      %48 = arith.constant 100 : i32
      %49 = arith.constant 1 : i32
      %i = memref.alloc() : memref<i32>
      scf.for %50 = %47 to %48 step %49  : i32 {
        memref.store %50, %i[] : memref<i32>
        %51 = memref.load %i[] : memref<i32>
        %52 = memref.load %i[] : memref<i32>
        %53 = arith.index_cast %52 : i32 to index
        memref.store %51, %host_src0[%53] : memref<100xi32>
        %54 = arith.constant 100 : i32
        %55 = memref.load %i[] : memref<i32>
        %56 = arith.subi %54, %55 : i32
        %57 = memref.load %i[] : memref<i32>
        %58 = arith.index_cast %57 : i32 to index
        memref.store %56, %host_src1[%58] : memref<100xi32>
      }
      %59 = memref.load %command_queue[] : memref<!tthost.command_queue>
      %60 = memref.load %src0_dram_buffer[] : memref<!tthost.buffer>
      %61 = arith.constant false
      "tthost.enqueue_write_buffer"(%59, %60, %host_src0, %61) : (!tthost.command_queue, !tthost.buffer, memref<100xi32>, i1) -> ()
      %62 = memref.load %command_queue[] : memref<!tthost.command_queue>
      %63 = memref.load %src1_dram_buffer[] : memref<!tthost.buffer>
      %64 = arith.constant false
      "tthost.enqueue_write_buffer"(%62, %63, %host_src1, %64) : (!tthost.command_queue, !tthost.buffer, memref<100xi32>, i1) -> ()
      %65 = memref.load %program[] : memref<!tthost.program>
      %66 = memref.load %core[] : memref<!tthost.corecoord>
      %67 = "tthost.create_kernel"(%65, %66) <{kernel_name = "reader.cpp", riscv_core = #tthost.riscv_core<datamovement_0>, noc_id = #builtin.int<0>}> : (!tthost.program, !tthost.corecoord) -> !tthost.kernel
      %kernel = memref.alloc() : memref<!tthost.kernel>
      memref.store %67, %kernel[] : memref<!tthost.kernel>
      %68 = memref.load %program[] : memref<!tthost.program>
      %69 = memref.load %kernel[] : memref<!tthost.kernel>
      %70 = memref.load %core[] : memref<!tthost.corecoord>
      %71 = memref.load %src0_dram_buffer[] : memref<!tthost.buffer>
      %72 = "tthost.get_memory_address"(%71) : (!tthost.buffer) -> index
      %73 = memref.load %src1_dram_buffer[] : memref<!tthost.buffer>
      %74 = "tthost.get_memory_address"(%73) : (!tthost.buffer) -> index
      %75 = memref.load %dst_dram_buffer[] : memref<!tthost.buffer>
      %76 = "tthost.get_memory_address"(%75) : (!tthost.buffer) -> index
      %77 = arith.constant 0 : i32
      %78 = arith.constant 0 : i32
      %79 = arith.constant 0 : i32
      "tthost.set_runtime_args"(%68, %69, %70, %72, %74, %76, %77, %78, %79) {operandSegmentSizes = array<i32: 1, 1, 1, 6>} : (!tthost.program, !tthost.kernel, !tthost.corecoord, index, index, index, i32, i32, i32) -> ()
      %80 = memref.load %command_queue[] : memref<!tthost.command_queue>
      %81 = memref.load %program[] : memref<!tthost.program>
      %82 = arith.constant false
      "tthost.enqueue_program"(%80, %81, %82) : (!tthost.command_queue, !tthost.program, i1) -> ()
      %83 = memref.load %command_queue[] : memref<!tthost.command_queue>
      "tthost.finish"(%83) : (!tthost.command_queue) -> ()
      %84 = memref.load %command_queue[] : memref<!tthost.command_queue>
      %85 = memref.load %dst_dram_buffer[] : memref<!tthost.buffer>
      %86 = arith.constant true
      "tthost.enqueue_read_buffer"(%84, %85, %host_dst, %86) : (!tthost.command_queue, !tthost.buffer, memref<100xi32>, i1) -> ()
      %87 = memref.load %device[] : memref<!tthost.device>
      "tthost.close_device"(%87) : (!tthost.device) -> ()
      %88 = arith.constant 0 : i32
      func.return %88 : i32
    }
  }
}

// CHECK:      void kernel_main() {
// CHECK-NEXT:     std::int32_t fn_arg_0 = get_arg_val<std::int32_t>(0);
// CHECK-NEXT:     std::int32_t fn_arg_1 = get_arg_val<std::int32_t>(1);
// CHECK-NEXT:     std::int32_t fn_arg_2 = get_arg_val<std::int32_t>(2);
// CHECK-NEXT:     std::int32_t fn_arg_3 = get_arg_val<std::int32_t>(3);
// CHECK-NEXT:     std::int32_t fn_arg_4 = get_arg_val<std::int32_t>(4);
// CHECK-NEXT:     std::int32_t fn_arg_5 = get_arg_val<std::int32_t>(5);
// CHECK-NEXT:     std::int64_t noc_addr_0 = get_noc_addr_from_bank_id<true>(fn_arg_3, fn_arg_0);
// CHECK-NEXT:     std::int64_t src0_dram_noc_addr = noc_addr_0;
// CHECK-NEXT:     std::int64_t noc_addr_1 = get_noc_addr_from_bank_id<true>(fn_arg_4, fn_arg_1);
// CHECK-NEXT:     std::int64_t src1_dram_noc_addr = noc_addr_1;
// CHECK-NEXT:     std::int64_t noc_addr_2 = get_noc_addr_from_bank_id<true>(fn_arg_5, fn_arg_2);
// CHECK-NEXT:     std::int64_t dst_dram_noc_addr = noc_addr_2;
// CHECK-NEXT:     std::int32_t write_ptr_3 = get_write_ptr(0);
// CHECK-NEXT:     std::int32_t l1_write_addr_in0 = write_ptr_3;
// CHECK-NEXT:     std::int32_t write_ptr_4 = get_write_ptr(1);
// CHECK-NEXT:     std::int32_t l1_write_addr_in1 = write_ptr_4;
// CHECK-NEXT:     std::int32_t write_ptr_5 = get_write_ptr(2);
// CHECK-NEXT:     std::int32_t l1_write_addr_in2 = write_ptr_5;
// CHECK-NEXT:     noc_async_read(src0_dram_noc_addr, l1_write_addr_in0, 400);
// CHECK-NEXT:     noc_async_read(src1_dram_noc_addr, l1_write_addr_in1, 400);
// CHECK-NEXT:     std::int32_t * src0_data = (std::int32_t*) l1_write_addr_in0;
// CHECK-NEXT:     std::int32_t * src1_data = (std::int32_t*) l1_write_addr_in1;
// CHECK-NEXT:     std::int32_t * dst_data = (std::int32_t*) l1_write_addr_in2;
// CHECK-NEXT:     noc_async_read_barrier();
// CHECK-NEXT:     std::int32_t x;
// CHECK-NEXT:     for (x = 0; x < 100; x += 1) {
// CHECK-NEXT:         dst_data[x] = src0_data[x] + src1_data[x];
// CHECK-NEXT:     }
// CHECK-NEXT:     noc_async_write(l1_write_addr_in2, dst_dram_noc_addr, 400);
// CHECK-NEXT:     noc_async_write_barrier();
// CHECK-NEXT: }
// CHECK:      std::int32_t main() {
// CHECK-NEXT:     CoreCoord core = CoreCoord{0, 0};
// CHECK-NEXT:     std::int32_t single_tile_size = 4 * 100;
// CHECK-NEXT:     IDevice* device_6 = CreateDevice(0);
// CHECK-NEXT:     IDevice* device = device_6;
// CHECK-NEXT:     CommandQueue & command_queue = device->command_queue();
// CHECK-NEXT:     Program program_7 = CreateProgram();
// CHECK-NEXT:     Program& program = program_7;
// CHECK-NEXT:     InterleavedBufferConfig dram_configuration {.device=device, .size=single_tile_size, .page_size=single_tile_size, .buffer_type = BufferType::DRAM};
// CHECK-NEXT:     std::shared_ptr<Buffer> buffer_8 = CreateBuffer(dram_configuration);
// CHECK-NEXT:     std::shared_ptr<Buffer> src0_dram_buffer = buffer_8;
// CHECK-NEXT:     std::shared_ptr<Buffer> buffer_9 = CreateBuffer(dram_configuration);
// CHECK-NEXT:     std::shared_ptr<Buffer> src1_dram_buffer = buffer_9;
// CHECK-NEXT:     std::shared_ptr<Buffer> buffer_10 = CreateBuffer(dram_configuration);
// CHECK-NEXT:     std::shared_ptr<Buffer> dst_dram_buffer = buffer_10;
// CHECK-NEXT:     CircularBufferConfig cb_config_11 = CircularBufferConfig(1 * 400, {{[{][{]}}0, tt::DataFormat::Int32{{[}][}]}}).set_page_size(0, 400);
// CHECK-NEXT:     CircularBufferConfig cb_0_config = cb_config_11;
// CHECK-NEXT:     CBHandle cb_12 = tt_metal::CreateCircularBuffer(program, core, cb_0_config);
// CHECK-NEXT:     CBHandle cb = cb_12;
// CHECK-NEXT:     CircularBufferConfig cb_config_13 = CircularBufferConfig(1 * 400, {{[{][{]}}1, tt::DataFormat::Int32{{[}][}]}}).set_page_size(1, 400);
// CHECK-NEXT:     CircularBufferConfig cb_1_config = cb_config_13;
// CHECK-NEXT:     CBHandle cb_14 = tt_metal::CreateCircularBuffer(program, core, cb_1_config);
// CHECK-NEXT:     CBHandle cb1 = cb_14;
// CHECK-NEXT:     CircularBufferConfig cb_config_15 = CircularBufferConfig(1 * 400, {{[{][{]}}2, tt::DataFormat::Int32{{[}][}]}}).set_page_size(2, 400);
// CHECK-NEXT:     CircularBufferConfig cb_2_config = cb_config_15;
// CHECK-NEXT:     CBHandle cb_16 = tt_metal::CreateCircularBuffer(program, core, cb_2_config);
// CHECK-NEXT:     CBHandle cb2 = cb_16;
// CHECK-NEXT:     std::int32_t * host_src0 = (std::int32_t*) malloc(sizeof(std::int32_t)*100);
// CHECK-NEXT:     std::int32_t * host_src1 = (std::int32_t*) malloc(sizeof(std::int32_t)*100);
// CHECK-NEXT:     std::int32_t * host_dst = (std::int32_t*) malloc(sizeof(std::int32_t)*100);
// CHECK-NEXT:     std::int32_t i;
// CHECK-NEXT:     for (i = 0; i < 100; i += 1) {
// CHECK-NEXT:         host_src0[i] = i;
// CHECK-NEXT:         host_src1[i] = 100 - i;
// CHECK-NEXT:     }
// CHECK-NEXT:     EnqueueWriteBuffer(command_queue, src0_dram_buffer, host_src0, false);
// CHECK-NEXT:     EnqueueWriteBuffer(command_queue, src1_dram_buffer, host_src1, false);
// CHECK-NEXT:     KernelHandle kernel_17 = CreateKernel(program, "reader.cpp", core, DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc=NOC::RISCV_0_default});
// CHECK-NEXT:     KernelHandle kernel = kernel_17;
// CHECK-NEXT:     SetRuntimeArgs(program, kernel, core, {src0_dram_buffer->address(), src1_dram_buffer->address(), dst_dram_buffer->address(), 0, 0, 0});
// CHECK-NEXT:     EnqueueProgram(command_queue, program, false);
// CHECK-NEXT:     Finish(command_queue);
// CHECK-NEXT:     EnqueueReadBuffer(command_queue, dst_dram_buffer, host_dst, true);
// CHECK-NEXT:     CloseDevice(device);
// CHECK-NEXT:     free(host_src0);
// CHECK-NEXT:     free(host_src1);
// CHECK-NEXT:     free(host_dst);
// CHECK-NEXT:     return 0;
// CHECK-NEXT: }