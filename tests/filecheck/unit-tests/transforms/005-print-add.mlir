// RUN: python3.13 tenstorrent/tools/tt-opt %s -t tt-metalium | filecheck %s

builtin.module {
  builtin.module attributes {kernel_type = "host", vis = "external"} {
    func.func @host_entry(%0 : memref<32x32xi32>, %1 : memref<32x32xi32>, %2 : memref<32x32xi32>) {
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
      "tthost.enqueue_write_buffer"(%9, %11, %0, %12) : (!tthost.command_queue, !tthost.buffer, memref<32x32xi32>, i1) -> ()
      %13 = arith.constant 1 : i32
      %14 = "tthost.create_cb_configuration"(%13, %size, %4) <{data_type = "int"}> : (i32, i32, i32) -> !tthost.circular_buffer_config
      %15 = "tthost.create_circular_buffer"(%prog, %8, %14) : (!tthost.program, !tthost.corecoord, !tthost.circular_buffer_config) -> !tthost.cb_handle
      %size_1 = arith.constant 4096 : i32
      %16 = "tthost.create_dram_configuration"(%size_1, %size_1) : (i32, i32) -> !tthost.dram_buffer_config
      %17 = "tthost.create_buffer"(%16) : (!tthost.dram_buffer_config) -> !tthost.buffer
      %18 = arith.constant false
      "tthost.enqueue_write_buffer"(%9, %17, %1, %18) : (!tthost.command_queue, !tthost.buffer, memref<32x32xi32>, i1) -> ()
      %19 = arith.constant 1 : i32
      %20 = "tthost.create_cb_configuration"(%19, %size_1, %5) <{data_type = "int"}> : (i32, i32, i32) -> !tthost.circular_buffer_config
      %21 = "tthost.create_circular_buffer"(%prog, %8, %20) : (!tthost.program, !tthost.corecoord, !tthost.circular_buffer_config) -> !tthost.cb_handle
      %size_2 = arith.constant 4096 : i32
      %22 = "tthost.create_dram_configuration"(%size_2, %size_2) : (i32, i32) -> !tthost.dram_buffer_config
      %23 = "tthost.create_buffer"(%22) : (!tthost.dram_buffer_config) -> !tthost.buffer
      %24 = arith.constant false
      "tthost.enqueue_write_buffer"(%9, %23, %2, %24) : (!tthost.command_queue, !tthost.buffer, memref<32x32xi32>, i1) -> ()
      %25 = arith.constant 1 : i32
      %26 = "tthost.create_cb_configuration"(%25, %size_2, %6) <{data_type = "int"}> : (i32, i32, i32) -> !tthost.circular_buffer_config
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
      "tthost.enqueue_read_buffer"(%9, %23, %2, %28) : (!tthost.command_queue, !tthost.buffer, memref<32x32xi32>, i1) -> ()
      "tthost.close_device"(%7) : (!tthost.device) -> ()
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
      %2 = arith.constant 0 : i32
      %3 = arith.constant 1 : i32
      %4 = arith.constant 2 : i32
      %5 = builtin.unrealized_conversion_cast %0 : i32 to ui32
      %6 = builtin.unrealized_conversion_cast %1 : i32 to ui32
      %7 = builtin.unrealized_conversion_cast %2 : i32 to ui32
      %8 = builtin.unrealized_conversion_cast %3 : i32 to ui32
      %9 = builtin.unrealized_conversion_cast %4 : i32 to ui32
      %10 = arith.constant true
      %11 = arith.constant false
      "comp.binary_op_init_common"(%7, %8, %9) : (ui32, ui32, ui32) -> ()
      "comp.add_tiles_init"(%7, %8, %11) : (ui32, ui32, i1) -> ()
      "cb.wait_front"(%3, %1) : (i32, i32) -> ()
      "cb.wait_front"(%2, %1) : (i32, i32) -> ()
      "comp.tile_regs_acquire"() : () -> ()
      "comp.add_tiles"(%7, %8, %5, %5, %5) : (ui32, ui32, ui32, ui32, ui32) -> ()
      "comp.tile_regs_commit"() : () -> ()
      "comp.tile_regs_wait"() : () -> ()
      "comp.pack_tile"(%5, %9, %5) <{out_of_order_output = false}> : (ui32, ui32, ui32) -> ()
      "comp.tile_regs_release"() : () -> ()
      "cb.pop_front"(%2, %1) : (i32, i32) -> ()
      "cb.pop_front"(%3, %1) : (i32, i32) -> ()
      "cb.push_back"(%4, %1) : (i32, i32) -> ()
      func.return
    }
  }
  builtin.module attributes {kernel_type = "data_out"} {
    func.func @kernel_main(%bank_id : ui32, %mem_addr : ui32, %size_bytes : ui32) {
      %0 = "dm.get_noc_addr_from_bank_id"(%bank_id, %mem_addr) <{dram = true}> : (ui32, ui32) -> ui64
      %1 = arith.constant 1 : i32
      %2 = arith.constant 2 : i32
      %3 = "cb.get_read_pointer"(%2) : (i32) -> ui32
      "cb.wait_front"(%2, %1) : (i32, i32) -> ()
      "dm.noc_async_write"(%3, %0, %size_bytes) : (ui32, ui64, ui32) -> ()
      "dm.noc_async_write_barrier"() : () -> ()
      "cb.pop_front"(%2, %1) : (i32, i32) -> ()
      func.return
    }
  }
}

// CHECK:      extern "C" void host_entry(std::int32_t* fn_arg_0, std::int32_t* fn_arg_1, std::int32_t* fn_arg_2) {
// CHECK-NEXT:     Program prog = CreateProgram();
// CHECK-NEXT:     IDevice* device_0 = CreateDevice(0);
// CHECK-NEXT:     std::shared_ptr<Buffer> buffer_1 = CreateBuffer({.device=device, .size=4096, .page_size=4096, .buffer_type = BufferType::DRAM});
// CHECK-NEXT:     EnqueueWriteBuffer(device_0->command_queue(), buffer_1, fn_arg_0, false);
// CHECK-NEXT:     CircularBufferConfig cb_config_2 = CircularBufferConfig(1 * 4096, {{[{][{]}}0, tt::DataFormat::Int32{{[}][}]}}).set_page_size(0, 4096);
// CHECK-NEXT:     CBHandle cb_3 = tt_metal::CreateCircularBuffer(prog, CoreCoord{0, 0}, cb_config_2);
// CHECK-NEXT:     std::shared_ptr<Buffer> buffer_4 = CreateBuffer({.device=device, .size=4096, .page_size=4096, .buffer_type = BufferType::DRAM});
// CHECK-NEXT:     EnqueueWriteBuffer(device_0->command_queue(), buffer_4, fn_arg_1, false);
// CHECK-NEXT:     CircularBufferConfig cb_config_5 = CircularBufferConfig(1 * 4096, {{[{][{]}}1, tt::DataFormat::Int32{{[}][}]}}).set_page_size(1, 4096);
// CHECK-NEXT:     CBHandle cb_6 = tt_metal::CreateCircularBuffer(prog, CoreCoord{0, 0}, cb_config_5);
// CHECK-NEXT:     std::shared_ptr<Buffer> buffer_7 = CreateBuffer({.device=device, .size=4096, .page_size=4096, .buffer_type = BufferType::DRAM});
// CHECK-NEXT:     EnqueueWriteBuffer(device_0->command_queue(), buffer_7, fn_arg_2, false);
// CHECK-NEXT:     CircularBufferConfig cb_config_8 = CircularBufferConfig(1 * 4096, {{[{][{]}}2, tt::DataFormat::Int32{{[}][}]}}).set_page_size(2, 4096);
// CHECK-NEXT:     CBHandle cb_9 = tt_metal::CreateCircularBuffer(prog, CoreCoord{0, 0}, cb_config_8);
// CHECK-NEXT:     KernelHandle reader_kernel = CreateKernel(prog, "reader.cpp", CoreCoord{0, 0}, DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc=NOC::RISCV_0_default});
// CHECK-NEXT:     KernelHandle writer_kernel = CreateKernel(prog, "writer.cpp", CoreCoord{0, 0}, DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc=NOC::RISCV_1_default});
// CHECK-NEXT:     KernelHandle compute_kernel = CreateKernel(prog, "compute.cpp", CoreCoord{0, 0}, ComputeConfig {.math_fidelity = MathFidelity::LoFi, .fp32_dest_acc_en = false, .math_approx_mode = false, .compile_args = {{{[}][}]}});
// CHECK-NEXT:     SetRuntimeArgs(prog, compute_kernel, CoreCoord{0, 0}, {});
// CHECK-NEXT:     SetRuntimeArgs(prog, reader_kernel, CoreCoord{0, 0}, {0, 0, buffer_1->address(), buffer_4->address(), 4096, 4096});
// CHECK-NEXT:     SetRuntimeArgs(prog, writer_kernel, CoreCoord{0, 0}, {0, buffer_7->address(), 4096});
// CHECK-NEXT:     EnqueueProgram(device_0->command_queue(), prog, false);
// CHECK-NEXT:     Finish(device_0->command_queue());
// CHECK-NEXT:     EnqueueReadBuffer(device_0->command_queue(), buffer_7, fn_arg_2, false);
// CHECK-NEXT:     CloseDevice(device_0);
// CHECK-NEXT: }
// CHECK:      void kernel_main() {
// CHECK-NEXT:     uint32_t fn_arg_0 = get_arg_val<uint32_t>(0);
// CHECK-NEXT:     uint32_t fn_arg_1 = get_arg_val<uint32_t>(1);
// CHECK-NEXT:     uint32_t fn_arg_2 = get_arg_val<uint32_t>(2);
// CHECK-NEXT:     uint32_t fn_arg_3 = get_arg_val<uint32_t>(3);
// CHECK-NEXT:     uint32_t fn_arg_4 = get_arg_val<uint32_t>(4);
// CHECK-NEXT:     uint32_t fn_arg_5 = get_arg_val<uint32_t>(5);
// CHECK-NEXT:     uint64_t noc_addr_10 = get_noc_addr_from_bank_id<true>(fn_arg_0, fn_arg_2, static_cast<std::uint8_t>(0));
// CHECK-NEXT:     uint32_t write_ptr_11 = get_write_ptr(0);
// CHECK-NEXT:     cb_reserve_back(0, 1);
// CHECK-NEXT:     noc_async_read(noc_addr_10, write_ptr_11, fn_arg_4);
// CHECK-NEXT:     noc_async_read_barrier();
// CHECK-NEXT:     cb_push_back(0, 1);
// CHECK-NEXT:     uint64_t noc_addr_12 = get_noc_addr_from_bank_id<true>(fn_arg_1, fn_arg_3, static_cast<std::uint8_t>(0));
// CHECK-NEXT:     uint32_t write_ptr_13 = get_write_ptr(1);
// CHECK-NEXT:     cb_reserve_back(1, 1);
// CHECK-NEXT:     noc_async_read(noc_addr_12, write_ptr_13, fn_arg_5);
// CHECK-NEXT:     noc_async_read_barrier();
// CHECK-NEXT:     cb_push_back(1, 1);
// CHECK-NEXT: }
// CHECK:      void MAIN {
// CHECK-NEXT:     binary_op_init_common(static_cast<std::uint32_t>(0), static_cast<std::uint32_t>(1), static_cast<std::uint32_t>(2));
// CHECK-NEXT:     add_tiles_init(static_cast<std::uint32_t>(0), static_cast<std::uint32_t>(1), false);
// CHECK-NEXT:     cb_wait_front(1, 1);
// CHECK-NEXT:     cb_wait_front(0, 1);
// CHECK-NEXT:     tile_regs_acquire();
// CHECK-NEXT:     add_tiles(static_cast<std::uint32_t>(0), static_cast<std::uint32_t>(1), static_cast<std::uint32_t>(0), static_cast<std::uint32_t>(0), static_cast<std::uint32_t>(0));
// CHECK-NEXT:     tile_regs_commit();
// CHECK-NEXT:     tile_regs_wait();
// CHECK-NEXT:     pack_tile<false>(static_cast<std::uint32_t>(0), static_cast<std::uint32_t>(2), static_cast<std::uint32_t>(0));
// CHECK-NEXT:     tile_regs_release();
// CHECK-NEXT:     cb_pop_front(0, 1);
// CHECK-NEXT:     cb_pop_front(1, 1);
// CHECK-NEXT:     cb_push_back(2, 1);
// CHECK-NEXT: }
// CHECK:      void kernel_main() {
// CHECK-NEXT:     uint32_t fn_arg_0 = get_arg_val<uint32_t>(0);
// CHECK-NEXT:     uint32_t fn_arg_1 = get_arg_val<uint32_t>(1);
// CHECK-NEXT:     uint32_t fn_arg_2 = get_arg_val<uint32_t>(2);
// CHECK-NEXT:     uint64_t noc_addr_14 = get_noc_addr_from_bank_id<true>(fn_arg_0, fn_arg_1);
// CHECK-NEXT:     uint32_t read_ptr_15 = get_read_ptr(2);
// CHECK-NEXT:     cb_wait_front(2, 1);
// CHECK-NEXT:     noc_async_write(read_ptr_15, noc_addr_14, fn_arg_2);
// CHECK-NEXT:     noc_async_write_barrier();
// CHECK-NEXT:     cb_pop_front(2, 1);
// CHECK-NEXT: }