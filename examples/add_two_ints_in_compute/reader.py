from tenstorrent.frontend import tt


@tt.data_in
def kernel_main(
    src0_dram: uint, src1_dram: uint, src0_dram_id: uint, src1_dram_id: uint
):
    src0_noc_addr = tt.get_noc_addr_from_bank_id(True, src0_dram_id, src0_dram)
    src1_noc_addr = tt.get_noc_addr_from_bank_id(True, src1_dram_id, src1_dram)

    cb_in0 = 0
    cb_in1 = 1

    l1_write_addr_in0 = tt.cb_get_write_ptr(cb_in0)
    l1_write_addr_in1 = tt.cb_get_write_ptr(cb_in1)

    # read 1 block of memory from DRAM into circular buffers, then push to compute core unpacker
    int_size_bytes = 4
    block_size = int_size_bytes * 100

    tt.cb_reserve_back(cb_in0, 1)
    tt.noc_async_read(src0_noc_addr, l1_write_addr_in0, block_size)
    tt.noc_async_read_barrier()
    tt.cb_push_back(cb_in0, 1)

    tt.cb_reserve_back(cb_in1, 1)
    tt.noc_async_read(src1_noc_addr, l1_write_addr_in1, block_size)
    tt.noc_async_read_barrier()
    tt.cb_push_back(cb_in1, 1)
