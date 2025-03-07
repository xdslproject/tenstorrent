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