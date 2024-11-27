

# Circular buffer dialect
def cb_pages_available_at_front(cb_id: int, num_pages: int) -> bool:
    pass


def cb_wait_front(cb_id: int, num_tiles: int):
    pass


def cb_pages_reservable_at_back(cb_id: int, num_pages: int) -> bool:
    pass


def cb_reserve_back(cb_id: int, num_tiles: int) -> bool:
    pass


def cb_push_back(cb_id: int, num_tiles: int):
    pass


def cb_pop_front(cb_id: int, num_tiles: int) -> bool:
    pass
