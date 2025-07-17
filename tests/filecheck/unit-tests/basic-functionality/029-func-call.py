# RUN: python3.13 %s | filecheck %s
from tenstorrent.frontend import tt


@tt.host
def host_api():
    device = tt.CreateDevice(0)


@tt.data_in
def data_in_api():
    tt.noc_async_read_barrier()


@tt.data_out
def data_out_api():
    tt.noc_async_write_barrier()


@tt.compute
def compute():
    tt.add(1, 2, 3, 4, 5)


# CHECK: "tthost.create_device"
# CHECK: "dm.noc_async_read_barrier"
# CHECK: "dm.noc_async_write_barrier"
# CHECK: "comp.add_tiles"
