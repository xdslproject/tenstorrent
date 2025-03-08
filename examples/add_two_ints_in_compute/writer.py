from tenstorrent.frontend import tt


@tt.data_out
def writer_kernel(dst_addr: uint, dst_dram_id: uint):
    dst_noc_addr = tt.get_noc_addr_from_bank_id(True, dst_dram_id, dst_addr)

    cb_id = 16
    l1_read_addr = tt.cb_get_read_ptr(cb_id)

    int_size_bytes = 4
    block_size = int_size_bytes * 100

    tt.cb_wait_front(cb_id, 1)
    tt.noc_async_write(l1_read_addr, dst_noc_addr, block_size)
    tt.noc_async_read_barrier()
    tt.cb_pop_front(cb_id, 1)

