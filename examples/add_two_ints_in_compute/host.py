from tenstorrent.frontend import tt


@tt.host
def host_code():
    device = tt.CreateDevice(0)

    cq = tt.GetCommandQueue(device)
    program = tt.CreateProgram()
    core = tt.Core(0, 0)

    dt_size = 4
    single_tile_size = dt_size * 100
    dram_config = tt.DRAMConfig(single_tile_size, single_tile_size)
    src0_dram_buffer = tt.CreateBuffer(dram_config)
    src1_dram_buffer = tt.CreateBuffer(dram_config)
    dest_dram_buffer = tt.CreateBuffer(dram_config)

    cb0_config = tt.CBConfig(1, 400, 0, int)
    cb1_config = tt.CBConfig(1, 400, 1, int)
    cb2_config = tt.CBConfig(1, 400, 2, int)

    cb0 = tt.CreateCircularBuffer(program, core, cb0_config)
    cb1 = tt.CreateCircularBuffer(program, core, cb1_config)
    cb2 = tt.CreateCircularBuffer(program, core, cb2_config)

    host_src0 = [0] * 100
    host_src1 = [0] * 100
    host_dst = [0] * 100

    for i in range(0, 100):
        host_src0[i] = i
        host_src1[i] = 100 - i

    # create kernels for data movement
    # TODO: check DataMovement_{0 vs 1}
    kernel_din = tt.Kernel(program, reader_kernel, core, tt.DataMovement_0, 0)
    kernel_dout = tt.Kernel(program, writer_kernel, core, tt.DataMovement_1, 1)

    kernel_comp = tt.Kernel(
        program,
        add_two_int_tiles,
        core,
        tt.MathFidelity.HiFi4,
        False,
        False,
        []
    )

    tt.EnqueueWriteBuffer(cq, src0_dram_buffer, host_src0, False)
    tt.EnqueueWriteBuffer(cq, src1_dram_buffer, host_src1, False)

    tt.SetRuntimeArgs(
        program,
        kernel_din,
        core,
        tt.GetMemoryAddress(src0_dram_buffer),
        tt.GetMemoryAddress(src1_dram_buffer),
        0,
        0
    )

    tt.SetRuntimeArgs(
        program,
        kernel_comp,
        core,
    )

    tt.SetRuntimeArgs(
        program,
        kernel_dout,
        core,
        tt.GetMemoryAddress(dest_dram_buffer),
        0,
    )

    tt.EnqueueProgram(cq, program, False)
    tt.Finish(cq)

    tt.EnqueueReadBuffer(cq, dest_dram_buffer, host_dst, True)
    tt.CloseDevice(device)
