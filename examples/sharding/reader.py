from tenstorrent.frontend import tt


@tt.data_in
def kernel_main(
    src_addr: uint,
    stick_size: uint,
    shard_height: uint,
    shard_width_bytes: uint,  # unneeded variable
    padded_offset_bytes: uint,
    start_id: uint,
    current_core: uint
):
    # TODO: implement tt.get_compile_time_arg_val (device) - 
    # TODO: if RHS is a call to get_compile_time_arg_val, annotate with constexpr
    # TODO: op1. make a constexpr container type in xDSL,
    # TODO: then need to update type checker, things support type x should also
    #  support constexpr(x) -- unlikely needed on host so device dialect time?
    cb_id_in0 = tt.get_compile_time_arg_val(0)
    src_is_dram = tt.get_compile_time_arg_val(1) == 1

    # TODO: implement tt.InterleavedAddrGen operation - DM dialect
    s0 = tt.InterleavedAddrGen(src_is_dram, src_addr, stick_size)
    stick_id = start_id

    tt.cb_reserve_back(cb_id_in0, shard_height)
    l1_write_addr = tt.cb_get_write_ptr(cb_id_in0)

    print("Core (0, ", current_core, "): ")
    for h in range(0, shard_height):
        src_noc_addr = tt.get_noc_address(stick_id, s0)
        tt.noc_async_read(src_noc_addr, l1_write_addr, stick_size)


        # print both BFloat16 values that are packed into the page
        # TODO: dereference l1_write_addr
        data = tt.to_array(l1_write_addr, uint16, 2)
        print(data[0], " ", data[1], " ", end="")
        # not sure what's happening within these lines


        stick_id += 1
        l1_write_addr += padded_offset_bytes

    print()
    tt.noc_async_read_barrier()
    tt.cb_push_back(cb_id_in0, shard_height)

