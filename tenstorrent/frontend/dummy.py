import inspect
from functools import wraps
from typing import Literal, get_type_hints
import ast


def enforce_literals(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        type_hints = get_type_hints(func)

        # Inspect the calling frame
        frame = inspect.currentframe().f_back
        code_context = inspect.getframeinfo(frame).code_context
        if not code_context:
            raise RuntimeError("Could not inspect caller's source code.")
        source_code = ''.join(code_context).strip()
        parsed = ast.parse(source_code)

        # Find the function call node
        call_node = parsed.body[0].value
        if not isinstance(call_node, ast.Call):
            raise RuntimeError("The inspected code is not a function call.")

        # validate each argument
        for arg, var in zip(call_node.args, type_hints):
            hint = type_hints[var]
            if hasattr(hint, '__origin__') and hint.__origin__ is Literal:
                if not isinstance(arg, ast.Constant):
                    raise Exception(f"Expected argument '{var}' to be a literal,"
                                    f" but found identifier: '{arg.id}'")

        return func(*args, **kwargs)

    return wrapper


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


@enforce_literals
def get_noc_addr_from_bank_id(
        dram: Literal[True, False],
        bank_id: int,
        bank_address_offset: int,
        noc: int
) -> int:
    pass


def get_noc_address(some_id: int):
    # TODO: implement this with other args
    pass
