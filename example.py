import tenstorrent as tt
from tenstorrent.frontend.dummy import *


@tt.data_in
def single_assignment(src0_dram, src1_dram, dst_dram, src0_bank_id, src1_bank_id, dest_bank_id):
    src0_dram_noc_addr = get_noc_addr_from_bank_id(True, src0_bank_id, src0_dram)
    src1_dram_noc_addr = get_noc_addr_from_bank_id(True, src1_bank_id, src1_dram)
    dest_dram_noc_addr = get_noc_addr_from_bank_id(True, dest_bank_id, dst_dram)

    # create three circular buffers of 400 bytes (uint32 * 100)
    src0_data = [0] * 100
    src1_data = [0] * 100
    dest_data = [0] * 100

    # read data from DRAM into those circular buffers
    noc_async_read(src0_dram_noc_addr, src0_data, 100)
    noc_async_read(src1_dram_noc_addr, src1_data, 100)
    noc_async_read_barrier()

    for x in range(0, 100):
        dest_data[x] = src0_data[x] + src1_data[x]

    # write result back to DRAM
    noc_async_write(dest_data, dest_dram_noc_addr, 100)
    noc_async_write_barrier()


