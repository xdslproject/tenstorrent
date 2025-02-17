# RUN: python3.11 %s | filecheck %s
from tenstorrent.frontend import tt


@tt.data_in
def single_assignment(
    src0_dram: uint,
    src1_dram: uint,
    dst_dram: uint,
    src0_dram_id: uint,
    src1_dram_id: uint,
    dst_dram_id: uint,
):
    src0_dram_noc_addr = tt.get_noc_addr_from_bank_id(True, src0_dram_id, src0_dram)
    src1_dram_noc_addr = tt.get_noc_addr_from_bank_id(True, src1_dram_id, src1_dram)
    dst_dram_noc_addr = tt.get_noc_addr_from_bank_id(True, dst_dram_id, dst_dram)

    l1_write_addr_in0 = tt.cb_get_write_ptr(0)
    l1_write_addr_in1 = tt.cb_get_write_ptr(1)
    l1_write_addr_in2 = tt.cb_get_write_ptr(2)

    tt.noc_async_read(src0_dram_noc_addr, l1_write_addr_in0, 400)
    tt.noc_async_read(src1_dram_noc_addr, l1_write_addr_in1, 400)

    src0_data = tt.to_array(l1_write_addr_in0, int, 100)
    src1_data = tt.to_array(l1_write_addr_in1, int, 100)
    dst_data = tt.to_array(l1_write_addr_in2, int, 100)

    tt.noc_async_read_barrier()

    for x in range(0, 100):
        dst_data[x] = src0_data[x] + src1_data[x]

    tt.noc_async_write(l1_write_addr_in2, dst_dram_noc_addr, 400)
    tt.noc_async_write_barrier()


@tt.host
def host_code():
    core = tt.Core(0, 0)
    single_tile_size = 4 * 100

    device = tt.CreateDevice(0)
    command_queue = tt.GetCommandQueue(device)

    program = tt.CreateProgram()

    dram_configuration = tt.DRAMConfig(single_tile_size, single_tile_size)
    src0_dram_buffer = tt.CreateBuffer(dram_configuration)
    src1_dram_buffer = tt.CreateBuffer(dram_configuration)
    dst_dram_buffer = tt.CreateBuffer(dram_configuration)

    cb_0_config = tt.CBConfig(1, 400, 0, int)
    cb_0 = tt.CreateCircularBuffer(program, core, cb_0_config)

    cb_1_config = tt.CBConfig(1, 400, 1, int)
    cb_1 = tt.CreateCircularBuffer(program, core, cb_1_config)

    cb_2_config = tt.CBConfig(1, 400, 2, int)
    cb_2 = tt.CreateCircularBuffer(program, core, cb_2_config)

    host_src0 = [0] * 100
    host_src1 = [0] * 100
    host_dst = [0] * 100

    for i in range(0, 100):
        host_src0[i] = i
        host_src1[i] = 100 - i

    tt.EnqueueWriteBuffer(command_queue, src0_dram_buffer, host_src0, False)
    tt.EnqueueWriteBuffer(command_queue, src1_dram_buffer, host_src1, False)

    kernel = tt.Kernel(program, single_assignment, core, tt.DataMovement_0, 0)

    tt.SetRuntimeArgs(
        program,
        kernel,
        core,
        tt.GetMemoryAddress(src0_dram_buffer),
        tt.GetMemoryAddress(src1_dram_buffer),
        tt.GetMemoryAddress(dst_dram_buffer),
        0,
        0,
        0,
    )

    tt.EnqueueProgram(command_queue, program, False)
    tt.Finish(command_queue)

    tt.EnqueueReadBuffer(command_queue, dst_dram_buffer, host_dst, True)
    tt.CloseDevice(device)


# CHECK:      builtin.module {
# CHECK-NEXT:   builtin.module attributes {kernel_type = "data_in"} {
# CHECK-NEXT:     func.func @kernel_main(%0 : ui32, %1 : ui32, %2 : ui32, %3 : ui32, %4 : ui32, %5 : ui32) {
# CHECK-NEXT:       %6 = "dm.get_noc_addr_from_bank_id"(%3, %0) <{dram = true}> : (ui32, ui32) -> ui64
# CHECK-NEXT:       %src0_dram_noc_addr = memref.alloc() : memref<ui64>
# CHECK-NEXT:       memref.store %6, %src0_dram_noc_addr[] : memref<ui64>
# CHECK-NEXT:       %7 = "dm.get_noc_addr_from_bank_id"(%4, %1) <{dram = true}> : (ui32, ui32) -> ui64
# CHECK-NEXT:       %src1_dram_noc_addr = memref.alloc() : memref<ui64>
# CHECK-NEXT:       memref.store %7, %src1_dram_noc_addr[] : memref<ui64>
# CHECK-NEXT:       %8 = "dm.get_noc_addr_from_bank_id"(%5, %2) <{dram = true}> : (ui32, ui32) -> ui64
# CHECK-NEXT:       %dst_dram_noc_addr = memref.alloc() : memref<ui64>
# CHECK-NEXT:       memref.store %8, %dst_dram_noc_addr[] : memref<ui64>
# CHECK-NEXT:       %9 = arith.constant 0 : i32
# CHECK-NEXT:       %10 = "cb.get_write_pointer"(%9) : (i32) -> ui32
# CHECK-NEXT:       %l1_write_addr_in0 = memref.alloc() : memref<ui32>
# CHECK-NEXT:       memref.store %10, %l1_write_addr_in0[] : memref<ui32>
# CHECK-NEXT:       %11 = arith.constant 1 : i32
# CHECK-NEXT:       %12 = "cb.get_write_pointer"(%11) : (i32) -> ui32
# CHECK-NEXT:       %l1_write_addr_in1 = memref.alloc() : memref<ui32>
# CHECK-NEXT:       memref.store %12, %l1_write_addr_in1[] : memref<ui32>
# CHECK-NEXT:       %13 = arith.constant 2 : i32
# CHECK-NEXT:       %14 = "cb.get_write_pointer"(%13) : (i32) -> ui32
# CHECK-NEXT:       %l1_write_addr_in2 = memref.alloc() : memref<ui32>
# CHECK-NEXT:       memref.store %14, %l1_write_addr_in2[] : memref<ui32>
# CHECK-NEXT:       %15 = memref.load %src0_dram_noc_addr[] : memref<ui64>
# CHECK-NEXT:       %16 = memref.load %l1_write_addr_in0[] : memref<ui32>
# CHECK-NEXT:       %17 = arith.constant 400 : i32
# CHECK-NEXT:       %18 = builtin.unrealized_conversion_cast %17 : i32 to ui32
# CHECK-NEXT:       "dm.noc_async_read"(%15, %16, %18) : (ui64, ui32, ui32) -> ()
# CHECK-NEXT:       %19 = memref.load %src1_dram_noc_addr[] : memref<ui64>
# CHECK-NEXT:       %20 = memref.load %l1_write_addr_in1[] : memref<ui32>
# CHECK-NEXT:       %21 = arith.constant 400 : i32
# CHECK-NEXT:       %22 = builtin.unrealized_conversion_cast %21 : i32 to ui32
# CHECK-NEXT:       "dm.noc_async_read"(%19, %20, %22) : (ui64, ui32, ui32) -> ()
# CHECK-NEXT:       %23 = memref.load %l1_write_addr_in0[] : memref<ui32>
# CHECK-NEXT:       %src0_data = builtin.unrealized_conversion_cast %23 : ui32 to memref<100xi32>
# CHECK-NEXT:       %24 = memref.load %l1_write_addr_in1[] : memref<ui32>
# CHECK-NEXT:       %src1_data = builtin.unrealized_conversion_cast %24 : ui32 to memref<100xi32>
# CHECK-NEXT:       %25 = memref.load %l1_write_addr_in2[] : memref<ui32>
# CHECK-NEXT:       %dst_data = builtin.unrealized_conversion_cast %25 : ui32 to memref<100xi32>
# CHECK-NEXT:       "dm.noc_async_read_barrier"() : () -> ()
# CHECK-NEXT:       %26 = arith.constant 0 : i32
# CHECK-NEXT:       %27 = arith.constant 100 : i32
# CHECK-NEXT:       %28 = arith.constant 1 : i32
# CHECK-NEXT:       %x = memref.alloc() : memref<i32>
# CHECK-NEXT:       scf.for %29 = %26 to %27 step %28  : i32 {
# CHECK-NEXT:         memref.store %29, %x[] : memref<i32>
# CHECK-NEXT:         %30 = memref.load %x[] : memref<i32>
# CHECK-NEXT:         %31 = arith.index_cast %30 : i32 to index
# CHECK-NEXT:         %32 = memref.load %src0_data[%31] : memref<100xi32>
# CHECK-NEXT:         %33 = memref.load %x[] : memref<i32>
# CHECK-NEXT:         %34 = arith.index_cast %33 : i32 to index
# CHECK-NEXT:         %35 = memref.load %src1_data[%34] : memref<100xi32>
# CHECK-NEXT:         %36 = arith.addi %32, %35 : i32
# CHECK-NEXT:         %37 = memref.load %x[] : memref<i32>
# CHECK-NEXT:         %38 = arith.index_cast %37 : i32 to index
# CHECK-NEXT:         memref.store %36, %dst_data[%38] : memref<100xi32>
# CHECK-NEXT:       }
# CHECK-NEXT:       %39 = memref.load %l1_write_addr_in2[] : memref<ui32>
# CHECK-NEXT:       %40 = memref.load %dst_dram_noc_addr[] : memref<ui64>
# CHECK-NEXT:       %41 = arith.constant 400 : i32
# CHECK-NEXT:       %42 = builtin.unrealized_conversion_cast %41 : i32 to ui32
# CHECK-NEXT:       "dm.noc_async_write"(%39, %40, %42) : (ui32, ui64, ui32) -> ()
# CHECK-NEXT:       "dm.noc_async_write_barrier"() : () -> ()
# CHECK-NEXT:       func.return
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT:   builtin.module attributes {kernel_type = "host"} {
# CHECK-NEXT:     func.func @main() -> i32 {
# CHECK-NEXT:       %0 = arith.constant 0 : i32
# CHECK-NEXT:       %1 = arith.constant 0 : i32
# CHECK-NEXT:       %2 = "tthost.core"(%0, %1) : (i32, i32) -> !tthost.corecoord
# CHECK-NEXT:       %core = memref.alloc() : memref<!tthost.corecoord>
# CHECK-NEXT:       memref.store %2, %core[] : memref<!tthost.corecoord>
# CHECK-NEXT:       %3 = arith.constant 4 : i32
# CHECK-NEXT:       %4 = arith.constant 100 : i32
# CHECK-NEXT:       %5 = arith.muli %3, %4 : i32
# CHECK-NEXT:       %single_tile_size = memref.alloc() : memref<i32>
# CHECK-NEXT:       memref.store %5, %single_tile_size[] : memref<i32>
# CHECK-NEXT:       %6 = arith.constant 0 : i32
# CHECK-NEXT:       %7 = "tthost.create_device"(%6) : (i32) -> !tthost.device
# CHECK-NEXT:       %device = memref.alloc() : memref<!tthost.device>
# CHECK-NEXT:       memref.store %7, %device[] : memref<!tthost.device>
# CHECK-NEXT:       %8 = memref.load %device[] : memref<!tthost.device>
# CHECK-NEXT:       %9 = "tthost.get_command_queue"(%8) : (!tthost.device) -> !tthost.command_queue
# CHECK-NEXT:       %command_queue = memref.alloc() : memref<!tthost.command_queue>
# CHECK-NEXT:       memref.store %9, %command_queue[] : memref<!tthost.command_queue>
# CHECK-NEXT:       %10 = "tthost.create_program"() : () -> !tthost.program
# CHECK-NEXT:       %program = memref.alloc() : memref<!tthost.program>
# CHECK-NEXT:       memref.store %10, %program[] : memref<!tthost.program>
# CHECK-NEXT:       %11 = memref.load %single_tile_size[] : memref<i32>
# CHECK-NEXT:       %12 = memref.load %single_tile_size[] : memref<i32>
# CHECK-NEXT:       %13 = "tthost.create_dram_configuration"(%11, %12) : (i32, i32) -> !tthost.dram_buffer_config
# CHECK-NEXT:       %dram_configuration = memref.alloc() : memref<!tthost.dram_buffer_config>
# CHECK-NEXT:       memref.store %13, %dram_configuration[] : memref<!tthost.dram_buffer_config>
# CHECK-NEXT:       %14 = memref.load %dram_configuration[] : memref<!tthost.dram_buffer_config>
# CHECK-NEXT:       %15 = "tthost.create_buffer"(%14) : (!tthost.dram_buffer_config) -> !tthost.buffer
# CHECK-NEXT:       %src0_dram_buffer = memref.alloc() : memref<!tthost.buffer>
# CHECK-NEXT:       memref.store %15, %src0_dram_buffer[] : memref<!tthost.buffer>
# CHECK-NEXT:       %16 = memref.load %dram_configuration[] : memref<!tthost.dram_buffer_config>
# CHECK-NEXT:       %17 = "tthost.create_buffer"(%16) : (!tthost.dram_buffer_config) -> !tthost.buffer
# CHECK-NEXT:       %src1_dram_buffer = memref.alloc() : memref<!tthost.buffer>
# CHECK-NEXT:       memref.store %17, %src1_dram_buffer[] : memref<!tthost.buffer>
# CHECK-NEXT:       %18 = memref.load %dram_configuration[] : memref<!tthost.dram_buffer_config>
# CHECK-NEXT:       %19 = "tthost.create_buffer"(%18) : (!tthost.dram_buffer_config) -> !tthost.buffer
# CHECK-NEXT:       %dst_dram_buffer = memref.alloc() : memref<!tthost.buffer>
# CHECK-NEXT:       memref.store %19, %dst_dram_buffer[] : memref<!tthost.buffer>
# CHECK-NEXT:       %20 = arith.constant 1 : i32
# CHECK-NEXT:       %21 = arith.constant 400 : i32
# CHECK-NEXT:       %22 = arith.constant 0 : i32
# CHECK-NEXT:       %23 = "tthost.create_cb_configuration"(%20, %21, %22) <{data_type = "int"}> : (i32, i32, i32) -> !tthost.circular_buffer_config
# CHECK-NEXT:       %cb_0_config = memref.alloc() : memref<!tthost.circular_buffer_config>
# CHECK-NEXT:       memref.store %23, %cb_0_config[] : memref<!tthost.circular_buffer_config>
# CHECK-NEXT:       %24 = memref.load %program[] : memref<!tthost.program>
# CHECK-NEXT:       %25 = memref.load %core[] : memref<!tthost.corecoord>
# CHECK-NEXT:       %26 = memref.load %cb_0_config[] : memref<!tthost.circular_buffer_config>
# CHECK-NEXT:       %27 = "tthost.create_circular_buffer"(%24, %25, %26) : (!tthost.program, !tthost.corecoord, !tthost.circular_buffer_config) -> !tthost.cb_handle
# CHECK-NEXT:       %cb = memref.alloc() : memref<!tthost.cb_handle>
# CHECK-NEXT:       memref.store %27, %cb[] : memref<!tthost.cb_handle>
# CHECK-NEXT:       %28 = arith.constant 1 : i32
# CHECK-NEXT:       %29 = arith.constant 400 : i32
# CHECK-NEXT:       %30 = arith.constant 1 : i32
# CHECK-NEXT:       %31 = "tthost.create_cb_configuration"(%28, %29, %30) <{data_type = "int"}> : (i32, i32, i32) -> !tthost.circular_buffer_config
# CHECK-NEXT:       %cb_1_config = memref.alloc() : memref<!tthost.circular_buffer_config>
# CHECK-NEXT:       memref.store %31, %cb_1_config[] : memref<!tthost.circular_buffer_config>
# CHECK-NEXT:       %32 = memref.load %program[] : memref<!tthost.program>
# CHECK-NEXT:       %33 = memref.load %core[] : memref<!tthost.corecoord>
# CHECK-NEXT:       %34 = memref.load %cb_1_config[] : memref<!tthost.circular_buffer_config>
# CHECK-NEXT:       %35 = "tthost.create_circular_buffer"(%32, %33, %34) : (!tthost.program, !tthost.corecoord, !tthost.circular_buffer_config) -> !tthost.cb_handle
# CHECK-NEXT:       %cb_1 = memref.alloc() : memref<!tthost.cb_handle>
# CHECK-NEXT:       memref.store %35, %cb_1[] : memref<!tthost.cb_handle>
# CHECK-NEXT:       %36 = arith.constant 1 : i32
# CHECK-NEXT:       %37 = arith.constant 400 : i32
# CHECK-NEXT:       %38 = arith.constant 2 : i32
# CHECK-NEXT:       %39 = "tthost.create_cb_configuration"(%36, %37, %38) <{data_type = "int"}> : (i32, i32, i32) -> !tthost.circular_buffer_config
# CHECK-NEXT:       %cb_2_config = memref.alloc() : memref<!tthost.circular_buffer_config>
# CHECK-NEXT:       memref.store %39, %cb_2_config[] : memref<!tthost.circular_buffer_config>
# CHECK-NEXT:       %40 = memref.load %program[] : memref<!tthost.program>
# CHECK-NEXT:       %41 = memref.load %core[] : memref<!tthost.corecoord>
# CHECK-NEXT:       %42 = memref.load %cb_2_config[] : memref<!tthost.circular_buffer_config>
# CHECK-NEXT:       %43 = "tthost.create_circular_buffer"(%40, %41, %42) : (!tthost.program, !tthost.corecoord, !tthost.circular_buffer_config) -> !tthost.cb_handle
# CHECK-NEXT:       %cb_2 = memref.alloc() : memref<!tthost.cb_handle>
# CHECK-NEXT:       memref.store %43, %cb_2[] : memref<!tthost.cb_handle>
# CHECK-NEXT:       %44 = arith.constant 100 : i32
# CHECK-NEXT:       %host_src0 = memref.alloc() : memref<100xi32>
# CHECK-NEXT:       %45 = arith.constant 100 : i32
# CHECK-NEXT:       %host_src1 = memref.alloc() : memref<100xi32>
# CHECK-NEXT:       %46 = arith.constant 100 : i32
# CHECK-NEXT:       %host_dst = memref.alloc() : memref<100xi32>
# CHECK-NEXT:       %47 = arith.constant 0 : i32
# CHECK-NEXT:       %48 = arith.constant 100 : i32
# CHECK-NEXT:       %49 = arith.constant 1 : i32
# CHECK-NEXT:       %i = memref.alloc() : memref<i32>
# CHECK-NEXT:       scf.for %50 = %47 to %48 step %49  : i32 {
# CHECK-NEXT:         memref.store %50, %i[] : memref<i32>
# CHECK-NEXT:         %51 = memref.load %i[] : memref<i32>
# CHECK-NEXT:         %52 = memref.load %i[] : memref<i32>
# CHECK-NEXT:         %53 = arith.index_cast %52 : i32 to index
# CHECK-NEXT:         memref.store %51, %host_src0[%53] : memref<100xi32>
# CHECK-NEXT:         %54 = arith.constant 100 : i32
# CHECK-NEXT:         %55 = memref.load %i[] : memref<i32>
# CHECK-NEXT:         %56 = arith.subi %54, %55 : i32
# CHECK-NEXT:         %57 = memref.load %i[] : memref<i32>
# CHECK-NEXT:         %58 = arith.index_cast %57 : i32 to index
# CHECK-NEXT:         memref.store %56, %host_src1[%58] : memref<100xi32>
# CHECK-NEXT:       }
# CHECK-NEXT:       %59 = memref.load %command_queue[] : memref<!tthost.command_queue>
# CHECK-NEXT:       %60 = memref.load %src0_dram_buffer[] : memref<!tthost.buffer>
# CHECK-NEXT:       %61 = arith.constant false
# CHECK-NEXT:       "tthost.enqueue_write_buffer"(%59, %60, %host_src0, %61) : (!tthost.command_queue, !tthost.buffer, memref<100xi32>, i1) -> ()
# CHECK-NEXT:       %62 = memref.load %command_queue[] : memref<!tthost.command_queue>
# CHECK-NEXT:       %63 = memref.load %src1_dram_buffer[] : memref<!tthost.buffer>
# CHECK-NEXT:       %64 = arith.constant false
# CHECK-NEXT:       "tthost.enqueue_write_buffer"(%62, %63, %host_src1, %64) : (!tthost.command_queue, !tthost.buffer, memref<100xi32>, i1) -> ()
# CHECK-NEXT:       %65 = memref.load %program[] : memref<!tthost.program>
# CHECK-NEXT:       %66 = memref.load %core[] : memref<!tthost.corecoord>
# CHECK-NEXT:       %67 = "tthost.create_kernel"(%65, %66) <{kernel_name = "single_assignment_kernel.cpp", riscv_core = #tthost.riscv_core<datamovement_0>, noc_id = #builtin.int<0>}> : (!tthost.program, !tthost.corecoord) -> !tthost.kernel
# CHECK-NEXT:       %kernel = memref.alloc() : memref<!tthost.kernel>
# CHECK-NEXT:       memref.store %67, %kernel[] : memref<!tthost.kernel>
# CHECK-NEXT:       %68 = memref.load %program[] : memref<!tthost.program>
# CHECK-NEXT:       %69 = memref.load %kernel[] : memref<!tthost.kernel>
# CHECK-NEXT:       %70 = memref.load %core[] : memref<!tthost.corecoord>
# CHECK-NEXT:       %71 = memref.load %src0_dram_buffer[] : memref<!tthost.buffer>
# CHECK-NEXT:       %72 = "tthost.get_memory_address"(%71) : (!tthost.buffer) -> index
# CHECK-NEXT:       %73 = memref.load %src1_dram_buffer[] : memref<!tthost.buffer>
# CHECK-NEXT:       %74 = "tthost.get_memory_address"(%73) : (!tthost.buffer) -> index
# CHECK-NEXT:       %75 = memref.load %dst_dram_buffer[] : memref<!tthost.buffer>
# CHECK-NEXT:       %76 = "tthost.get_memory_address"(%75) : (!tthost.buffer) -> index
# CHECK-NEXT:       %77 = arith.constant 0 : i32
# CHECK-NEXT:       %78 = arith.constant 0 : i32
# CHECK-NEXT:       %79 = arith.constant 0 : i32
# CHECK-NEXT:       "tthost.set_runtime_args"(%68, %69, %70, %72, %74, %76, %77, %78, %79) {operandSegmentSizes = array<i32: 1, 1, 1, 6>} : (!tthost.program, !tthost.kernel, !tthost.corecoord, index, index, index, i32, i32, i32) -> ()
# CHECK-NEXT:       %80 = memref.load %command_queue[] : memref<!tthost.command_queue>
# CHECK-NEXT:       %81 = memref.load %program[] : memref<!tthost.program>
# CHECK-NEXT:       %82 = arith.constant false
# CHECK-NEXT:       "tthost.enqueue_program"(%80, %81, %82) : (!tthost.command_queue, !tthost.program, i1) -> ()
# CHECK-NEXT:       %83 = memref.load %command_queue[] : memref<!tthost.command_queue>
# CHECK-NEXT:       "tthost.finish"(%83) : (!tthost.command_queue) -> ()
# CHECK-NEXT:       %84 = memref.load %command_queue[] : memref<!tthost.command_queue>
# CHECK-NEXT:       %85 = memref.load %dst_dram_buffer[] : memref<!tthost.buffer>
# CHECK-NEXT:       %86 = arith.constant true
# CHECK-NEXT:       "tthost.enqueue_read_buffer"(%84, %85, %host_dst, %86) : (!tthost.command_queue, !tthost.buffer, memref<100xi32>, i1) -> ()
# CHECK-NEXT:       %87 = memref.load %device[] : memref<!tthost.device>
# CHECK-NEXT:       "tthost.close_device"(%87) : (!tthost.device) -> ()
# CHECK-NEXT:       %88 = arith.constant 0 : i32
# CHECK-NEXT:       func.return %88 : i32
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT: }
