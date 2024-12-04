# Circular buffer dialect
def cb_pages_available_at_front(cb_id: int, num_pages: int) -> bool:
    pass


def cb_pages_reservable_at_back(cb_id: int, num_pages: int) -> bool:
    pass


def cb_wait_front(cb_id: int, num_tiles: int):
    pass


def cb_reserve_back(cb_id: int, num_tiles: int):
    pass


def cb_push_back(cb_id: int, num_tiles: int):
    pass


def cb_pop_front(cb_id: int, num_tiles: int):
    pass


# Data movement dialect
def noc_async_read(src_noc_addr: int, dst_local_l1_addr: int, size: int, noc: int):
    pass


def noc_async_write(src_local_l1_addr: int, dst_noc_addr: int, size: int, noc: int):
    pass


def noc_async_read_barrier(noc: int):
    pass


def noc_async_write_barrier(noc: int):
    pass


def noc_async_write_multicast(
        src_local_l1_addr: int,
        dst_noc_addr_multicast: int,
        size: int,
        num_dests: int,
        noc: int,
        linked: bool = False,
        multicast_path_reserve: bool = True,
):
    pass


def noc_semaphore_set_multicast(
        src_local_l1_addr: int,
        dst_noc_addr_multicast: int,
        num_dests: int,
        noc: int,
        linked: bool = False,
        multicast_path_reserve: bool = True,
):
    pass


# TODO: this api may benefit from high level Semaphore() class as uses pointers to addresses
def noc_semaphore_set(sem_addr: int, val: int):
    pass


# and this
def noc_semaphore_wait(sem_addr: int, val: int):
    pass


def noc_semaphore_inc(addr: int, incr: int, noc_id: int):
    pass
