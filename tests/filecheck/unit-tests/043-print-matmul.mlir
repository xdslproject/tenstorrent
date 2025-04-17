// RUN: python3.13 tenstorrent/tools/tt-opt %s -t tt-metalium | filecheck %s

builtin.module {
  builtin.module attributes {kernel_type = "host", vis = "external"} {
    func.func @host_entry(%0 : memref<10x10xi32>, %1 : memref<10x10xi32>, %2 : memref<10x10xi32>) {
      %3 = arith.constant 400 : i32
      %4 = arith.constant 400 : i32
      %5 = arith.constant 400 : i32
      %6 = "tthost.create_program"() : () -> !tthost.program
      %7 = arith.constant 0 : i32
      %8 = arith.constant 1 : i32
      %9 = arith.constant 16 : i32
      %10 = "tthost.create_device"(%7) : (i32) -> !tthost.device
      %11 = "tthost.core"(%7, %7) : (i32, i32) -> !tthost.corecoord
      %12 = "tthost.get_command_queue"(%10) : (!tthost.device) -> !tthost.command_queue
      %13 = "tthost.create_dram_configuration"(%3, %3) : (i32, i32) -> !tthost.dram_buffer_config
      %14 = "tthost.create_dram_configuration"(%4, %4) : (i32, i32) -> !tthost.dram_buffer_config
      %15 = "tthost.create_dram_configuration"(%5, %5) : (i32, i32) -> !tthost.dram_buffer_config
      %16 = "tthost.create_buffer"(%13) : (!tthost.dram_buffer_config) -> !tthost.buffer
      %17 = "tthost.create_buffer"(%14) : (!tthost.dram_buffer_config) -> !tthost.buffer
      %18 = "tthost.create_buffer"(%15) : (!tthost.dram_buffer_config) -> !tthost.buffer
      %19 = arith.constant false
      "tthost.enqueue_write_buffer"(%12, %16, %0, %19) : (!tthost.command_queue, !tthost.buffer, memref<10x10xi32>, i1) -> ()
      "tthost.enqueue_write_buffer"(%12, %17, %1, %19) : (!tthost.command_queue, !tthost.buffer, memref<10x10xi32>, i1) -> ()
      %20 = "tthost.create_cb_configuration"(%8, %3, %7) <{data_type = "int"}> : (i32, i32, i32) -> !tthost.circular_buffer_config
      %21 = "tthost.create_cb_configuration"(%8, %4, %8) <{data_type = "int"}> : (i32, i32, i32) -> !tthost.circular_buffer_config
      %22 = "tthost.create_cb_configuration"(%8, %5, %9) <{data_type = "int"}> : (i32, i32, i32) -> !tthost.circular_buffer_config
      %23 = "tthost.create_circular_buffer"(%6, %11, %20) : (!tthost.program, !tthost.corecoord, !tthost.circular_buffer_config) -> !tthost.cb_handle
      %24 = "tthost.create_circular_buffer"(%6, %11, %21) : (!tthost.program, !tthost.corecoord, !tthost.circular_buffer_config) -> !tthost.cb_handle
      %25 = "tthost.create_circular_buffer"(%6, %11, %22) : (!tthost.program, !tthost.corecoord, !tthost.circular_buffer_config) -> !tthost.cb_handle
      %26 = "tthost.create_kernel"(%6, %11) <{kernel_name = "reader.cpp", riscv_core = #tthost.riscv_core<datamovement_0>, noc_id = #builtin.int<0>}> : (!tthost.program, !tthost.corecoord) -> !tthost.kernel
      %27 = "tthost.create_kernel"(%6, %11) <{kernel_name = "writer.cpp", riscv_core = #tthost.riscv_core<datamovement_1>, noc_id = #builtin.int<1>}> : (!tthost.program, !tthost.corecoord) -> !tthost.kernel
      %28 = "tthost.create_compute_kernel"(%6, %11) <{kernel_name = "compute.cpp", riscv_core = #tthost.riscv_core<compute>, math_fidelity = #tthost.math_fidelity<LoFi>, fp32_dest_acc_en = false, math_approx_mode = false}> : (!tthost.program, !tthost.corecoord) -> !tthost.kernel
      %29 = "tthost.get_memory_address"(%16) : (!tthost.buffer) -> index
      %30 = "tthost.get_memory_address"(%17) : (!tthost.buffer) -> index
      %31 = "tthost.get_memory_address"(%18) : (!tthost.buffer) -> index
      "tthost.set_runtime_args"(%6, %26, %11, %29, %30, %7, %7, %3, %4) {operandSegmentSizes = array<i32: 1, 1, 1, 6>} : (!tthost.program, !tthost.kernel, !tthost.corecoord, index, index, i32, i32, i32, i32) -> ()
      "tthost.set_runtime_args"(%6, %28, %11) {operandSegmentSizes = array<i32: 1, 1, 1, 0>} : (!tthost.program, !tthost.kernel, !tthost.corecoord) -> ()
      "tthost.set_runtime_args"(%6, %27, %11, %31, %7, %5) {operandSegmentSizes = array<i32: 1, 1, 1, 3>} : (!tthost.program, !tthost.kernel, !tthost.corecoord, index, i32, i32) -> ()
      "tthost.enqueue_program"(%12, %6, %19) : (!tthost.command_queue, !tthost.program, i1) -> ()
      "tthost.finish"(%12) : (!tthost.command_queue) -> ()
      "tthost.enqueue_read_buffer"(%12, %18, %2, %19) : (!tthost.command_queue, !tthost.buffer, memref<10x10xi32>, i1) -> ()
      "tthost.close_device"(%10) : (!tthost.device) -> ()
      func.return
    }
  }
  builtin.module attributes {kernel_type = "data_in"} {
    func.func @kernel_main(%0 : ui32, %1 : ui32, %2 : ui32, %3 : ui32, %4 : ui32, %5 : ui32) {
      %6 = arith.constant 0 : i8
      %7 = builtin.unrealized_conversion_cast %6 : i8 to ui8
      %8 = arith.constant 0 : i32
      %9 = arith.constant 1 : i32
      %10 = "dm.get_noc_addr_from_bank_id"(%0, %1, %7) <{dram = true}> : (ui32, ui32, ui8) -> ui64
      %11 = "dm.get_noc_addr_from_bank_id"(%2, %3, %7) <{dram = true}> : (ui32, ui32, ui8) -> ui64
      %12 = "cb.get_write_pointer"(%8) : (i32) -> ui32
      %13 = "cb.get_write_pointer"(%9) : (i32) -> ui32
      "cb.reserve_back"(%8, %9) : (i32, i32) -> ()
      "dm.noc_async_read"(%10, %12, %4) : (ui64, ui32, ui32) -> ()
      "dm.noc_async_read_barrier"() : () -> ()
      "cb.push_back"(%8, %9) : (i32, i32) -> ()
      "cb.reserve_back"(%9, %9) : (i32, i32) -> ()
      "dm.noc_async_read"(%10, %12, %5) : (ui64, ui32, ui32) -> ()
      "dm.noc_async_read_barrier"() : () -> ()
      "cb.push_back"(%9, %9) : (i32, i32) -> ()
      func.return
    }
  }
  builtin.module attributes {kernel_type = "compute"} {
    func.func @kernel_main() {
      %0 = arith.constant 0 : i32
      %1 = arith.constant 1 : i32
      %2 = arith.constant 16 : i32
      %3 = builtin.unrealized_conversion_cast %0 : i32 to ui32
      %4 = builtin.unrealized_conversion_cast %1 : i32 to ui32
      %5 = builtin.unrealized_conversion_cast %2 : i32 to ui32
      "comp.binary_op_init_common"(%3, %4, %5) : (ui32, ui32, ui32) -> ()
      "comp.mm_init"(%3, %4, %3, %3) : (ui32, ui32, ui32, ui32) -> ()
      "cb.wait_front"(%0, %1) : (i32, i32) -> ()
      "cb.wait_front"(%1, %1) : (i32, i32) -> ()
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
    func.func @kernel_main(%0 : ui32, %1 : ui32, %2 : ui32) {
      %3 = "dm.get_noc_addr_from_bank_id"(%0, %1) <{dram = true}> : (ui32, ui32) -> ui64
      %4 = arith.constant 1 : i32
      %5 = arith.constant 16 : i32
      %6 = "cb.get_read_pointer"(%5) : (i32) -> ui32
      "cb.wait_front"(%5, %4) : (i32, i32) -> ()
      "dm.noc_async_write"(%6, %3, %2) : (ui32, ui64, ui32) -> ()
      "dm.noc_async_write_barrier"() : () -> ()
      "cb.pop_front"(%5, %4) : (i32, i32) -> ()
      func.return
    }
  }
}

// CHECK:      #include "host_api.hpp"
// CHECK-NEXT: #include "device_impl.hpp"
// CHECK:      using namespace tt;
// CHECK-NEXT: using namespace tt::tt_metal;
// CHECK:      extern "C" void host_entry(std::int32_t* fn_arg_0, std::int32_t* fn_arg_1, std::int32_t* fn_arg_2) {
// CHECK-NEXT:     Program program_0 = CreateProgram();
// CHECK-NEXT:     IDevice* device_1 = CreateDevice(0);
// CHECK-NEXT:     std::shared_ptr<Buffer> buffer_2 = CreateBuffer({.device=device, .size=400, .page_size=400, .buffer_type = BufferType::DRAM});
// CHECK-NEXT:     std::shared_ptr<Buffer> buffer_3 = CreateBuffer({.device=device, .size=400, .page_size=400, .buffer_type = BufferType::DRAM});
// CHECK-NEXT:     std::shared_ptr<Buffer> buffer_4 = CreateBuffer({.device=device, .size=400, .page_size=400, .buffer_type = BufferType::DRAM});
// CHECK-NEXT:     EnqueueWriteBuffer(device_1->command_queue(), buffer_2, fn_arg_0, false);
// CHECK-NEXT:     EnqueueWriteBuffer(device_1->command_queue(), buffer_3, fn_arg_1, false);
// CHECK-NEXT:     CircularBufferConfig cb_config_5 = CircularBufferConfig(1*400, {{[{][{]}}0, tt::DataFormat::Int32{{[}][}]}}).set_page_size(0, 400);
// CHECK-NEXT:     CircularBufferConfig cb_config_6 = CircularBufferConfig(1*400, {{[{][{]}}1, tt::DataFormat::Int32{{[}][}]}}).set_page_size(1, 400);
// CHECK-NEXT:     CircularBufferConfig cb_config_7 = CircularBufferConfig(1*400, {{[{][{]}}16, tt::DataFormat::Int32{{[}][}]}}).set_page_size(16, 400);
// CHECK-NEXT:     CBHandle cb_8 = tt_metal::CreateCircularBuffer(program_0, CoreCoord{0, 0}, cb_config_5);
// CHECK-NEXT:     CBHandle cb_9 = tt_metal::CreateCircularBuffer(program_0, CoreCoord{0, 0}, cb_config_6);
// CHECK-NEXT:     CBHandle cb_10 = tt_metal::CreateCircularBuffer(program_0, CoreCoord{0, 0}, cb_config_7);
// CHECK-NEXT:     KernelHandle kernel_11 = CreateKernel(program_0, "reader.cpp", CoreCoord{0, 0}, DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc=NOC::RISCV_0_default});
// CHECK-NEXT:     KernelHandle kernel_12 = CreateKernel(program_0, "writer.cpp", CoreCoord{0, 0}, DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc=NOC::RISCV_1_default});
// CHECK-NEXT:     KernelHandle kernel_13 = CreateKernel(program_0, "compute.cpp", CoreCoord{0, 0}, ComputeConfig {.math_fidelity = MathFidelity::LoFi, .fp32_dest_acc_en = false, .math_approx_mode = false, .compile_args = {}});
// CHECK-NEXT:     SetRuntimeArgs(program_0, kernel_11, CoreCoord{0, 0}, {buffer_2->address(), buffer_3->address(), 0, 0, 400, 400});
// CHECK-NEXT:     SetRuntimeArgs(program_0, kernel_13, CoreCoord{0, 0}, {});
// CHECK-NEXT:     SetRuntimeArgs(program_0, kernel_12, CoreCoord{0, 0}, {buffer_4->address(), 0, 400});
// CHECK-NEXT:     EnqueueProgram(device_1->command_queue(), program_0, false);
// CHECK-NEXT:     Finish(device_1->command_queue());
// CHECK-NEXT:     EnqueueReadBuffer(device_1->command_queue(), buffer_4, fn_arg_2, false);
// CHECK-NEXT:     CloseDevice(device_1);
// CHECK-NEXT: }
// CHECK:      #include <stdint.h>
// CHECK-NEXT: #include "dataflow_api.h"
// CHECK-NEXT: #include "debug/dprint.h"
// CHECK:      void kernel_main() {
// CHECK-NEXT:     uint32_t fn_arg_0 = get_arg_val<uint32_t>(0);
// CHECK-NEXT:     uint32_t fn_arg_1 = get_arg_val<uint32_t>(1);
// CHECK-NEXT:     uint32_t fn_arg_2 = get_arg_val<uint32_t>(2);
// CHECK-NEXT:     uint32_t fn_arg_3 = get_arg_val<uint32_t>(3);
// CHECK-NEXT:     uint32_t fn_arg_4 = get_arg_val<uint32_t>(4);
// CHECK-NEXT:     uint32_t fn_arg_5 = get_arg_val<uint32_t>(5);
// CHECK-NEXT:     uint64_t noc_addr_14 = get_noc_addr_from_bank_id<true>(fn_arg_0, fn_arg_1, static_cast<std::uint8_t>(0));
// CHECK-NEXT:     uint64_t noc_addr_15 = get_noc_addr_from_bank_id<true>(fn_arg_2, fn_arg_3, static_cast<std::uint8_t>(0));
// CHECK-NEXT:     uint32_t write_ptr_16 = get_write_ptr(0);
// CHECK-NEXT:     uint32_t write_ptr_17 = get_write_ptr(1);
// CHECK-NEXT:     cb_reserve_back(0, 1);
// CHECK-NEXT:     noc_async_read(noc_addr_14, write_ptr_16, fn_arg_4);
// CHECK-NEXT:     noc_async_read_barrier();
// CHECK-NEXT:     cb_push_back(0, 1);
// CHECK-NEXT:     cb_reserve_back(1, 1);
// CHECK-NEXT:     noc_async_read(noc_addr_14, write_ptr_16, fn_arg_5);
// CHECK-NEXT:     noc_async_read_barrier();
// CHECK-NEXT:     cb_push_back(1, 1);
// CHECK-NEXT: }
// CHECK:      #include <cstdint>
// CHECK-NEXT: #include "compute_kernel_api/matmul.h
// CHECK-NEXT: #include "compute_kernel_api/tile_move_copy.h"
// CHECK-NEXT: #include "compute_kernel_api/eltwise_binary.h"
// CHECK-NEXT: #include "debug/dprint.h"
// CHECK:      void MAIN {
// CHECK-NEXT:     binary_op_init_common(static_cast<std::uint32_t>(0), static_cast<std::uint32_t>(1), static_cast<std::uint32_t>(16));
// CHECK-NEXT:     mm_init(static_cast<std::uint32_t>(0), static_cast<std::uint32_t>(1), static_cast<std::uint32_t>(0), static_cast<std::uint32_t>(0));
// CHECK-NEXT:     cb_wait_front(0, 1);
// CHECK-NEXT:     cb_wait_front(1, 1);
// CHECK-NEXT:     tile_regs_acquire();
// CHECK-NEXT:     matmul_tiles(static_cast<std::uint32_t>(0), static_cast<std::uint32_t>(1), static_cast<std::uint32_t>(0), static_cast<std::uint32_t>(0), static_cast<std::uint32_t>(0), static_cast<std::uint32_t>(0));
// CHECK-NEXT:     tile_regs_commit();
// CHECK-NEXT:     tile_regs_wait();
// CHECK-NEXT:     pack_tile<false>(static_cast<std::uint32_t>(0), static_cast<std::uint32_t>(16), static_cast<std::uint32_t>(0));
// CHECK-NEXT:     tile_regs_release();
// CHECK-NEXT:     cb_pop_front(0, 1);
// CHECK-NEXT:     cb_pop_front(1, 1);
// CHECK-NEXT:     cb_push_back(16, 1);
// CHECK-NEXT: }
// CHECK:      #include "dataflow_api.h"
// CHECK-NEXT: #include "debug/dprint.h"
// CHECK:      void kernel_main() {
// CHECK-NEXT:     uint32_t fn_arg_0 = get_arg_val<uint32_t>(0);
// CHECK-NEXT:     uint32_t fn_arg_1 = get_arg_val<uint32_t>(1);
// CHECK-NEXT:     uint32_t fn_arg_2 = get_arg_val<uint32_t>(2);
// CHECK-NEXT:     uint64_t noc_addr_18 = get_noc_addr_from_bank_id<true>(fn_arg_0, fn_arg_1);
// CHECK-NEXT:     uint32_t read_ptr_19 = get_read_ptr(16);
// CHECK-NEXT:     cb_wait_front(16, 1);
// CHECK-NEXT:     noc_async_write(read_ptr_19, noc_addr_18, fn_arg_2);
// CHECK-NEXT:     noc_async_write_barrier();
// CHECK-NEXT:     cb_pop_front(16, 1);
// CHECK-NEXT: }
