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
        source_code = "".join(code_context).strip()
        parsed = ast.parse(source_code)

        # Find the function call node
        call_node = parsed.body[0].value
        if not isinstance(call_node, ast.Call):
            raise RuntimeError("The inspected code is not a function call.")

        # validate each argument
        for arg, var in zip(call_node.args, type_hints):
            hint = type_hints[var]
            if hasattr(hint, "__origin__") and hint.__origin__ is Literal:
                if not isinstance(arg, ast.Constant):
                    raise Exception(
                        f"Expected argument '{var}' to be a literal,"
                        f" but found identifier: '{arg.id}'"
                    )

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
    dram: Literal[True, False], bank_id: int, bank_address_offset: int, noc: int
) -> int:
    pass


def get_noc_address(some_id: int):
    # TODO: implement this with other args
    pass


# Compute dialect
def copy(cb: int, in_tile_index: int, dst_tile_index: int):
    pass


def copy_to_dst_init_short_with_dt(old_cb: int, new_cb: int, transpose: int):
    pass


def copy_to_dst_init_short(cb: int, transpose: int):
    pass


def copy_init():
    pass


def acquire_dst():
    pass


def release_dst():
    pass


def regs_acquire():
    pass


def regs_wait():
    pass


def regs_commit():
    pass


def regs_release():
    pass


def abs_init():
    pass


def abs(dst: int):
    pass


def add_init_nof():
    pass


def add_init(cb0: int, cb1: int, acc_to_dest: bool = False):
    pass


def add(cb0: int, cb1: int, tile0: int, tile1: int, dst: int):
    pass


def sub_init_nof():
    pass


def sub_init(cb0: int, cb1: int, acc_to_dest: bool):
    pass


def sub(cb0: int, cb1: int, tile0: int, tile1: int, dst: int):
    pass


def mul_init_f():
    pass


def mul_init(cb0: int, cb1: int):
    pass


def mul(cb0: int, cb1: int, tile0: int, tile1: int, dst: int):
    pass


def add_bcast_cols_init_short(cb0: int, cb1: int):
    pass


def add_bcast_rows_init_short(cb0: int, cb1: int):
    pass


def add_bcast(cb0: int, cb1: int, tile0: int, tile1: int, dst: int):
    pass


def sub_bcast_cols_init_short(cb0: int, cb1: int):
    pass


def sub_bcast(cb0: int, cb1: int, tile0: int, tile1: int, dst: int):
    pass


def mul_bcast_cols_init_short(cb0: int, cb1: int):
    pass


def mul_bcast_rows_init_short(cb0: int, cb1: int):
    pass


def mul_bcast(cb0: int, cb1: int, tile0: int, tile1: int, dst: int):
    pass


def mul_bcast_scalar_init_short(cb0: int, cb1: int):
    pass


def mul_bcast_scalar(cb0: int, cb1: int, tile0: int, tile1: int, dst: int):
    pass


def mm_init(cb0: int, cb1: int, dst: int, transpose: int):
    pass


def mm_init_short_with_dt(cb0: int, cb1: int, dst: int, transpose: int):
    pass


def mm_init_short(cb0: int, cb1: int, dst: int):
    pass


def matmul(cb0: int, cb1: int, tile0: int, tile1: int, dst: int, transpose: int):
    pass


def mm_block_init(
    cb0: int,
    cb1: int,
    dst: int,
    transpose: int,
    out_cols: int,
    out_rows: int,
    kt_dim: int,
):
    pass


def mm_block_init_short(
    cb0: int, cb1: int, transpose: int, out_cols: int, out_rows: int, kt_dim: int
):
    pass


def mm_block_init_short_with_dt(
    cb0: int, cb1: int, prev_cb1: int, out_cols: int, out_rows: int, kt_dim: int
):
    pass


def matmul_block(
    cb0: int,
    cb1: int,
    tile0: int,
    tile1: int,
    dst: int,
    transpose: bool,
    out_cols: int,
    out_rows: int,
    kt_dim: int,
):
    pass


def exp_init(fast_and_approx: Literal[True, False]):
    pass


def exp(fast_and_approx: Literal[True, False], dst: int):
    pass


def exp2init():
    pass


def exp2(dst: int):
    pass


def exp_m1init():
    pass


def exp_m1(dst: int):
    pass


def relu_init():
    pass


def relu(dst: int):
    pass


def relu_max_init():
    pass


def relu_max(dst: int, upper_limit: int):
    pass


def relu_min_init():
    pass


def relu_min(dst: int, lower_limit: int):
    pass


def leaky_relu_init(dst: int, slope: int):
    pass


def elu_init():
    pass


def elu(dst: int, slope: int):
    pass


def erf_init(fast_and_approx: Literal[True, False]):
    pass


def erf(fast_and_approx: Literal[True, False], dst: int):
    pass


def erfc_init(fast_and_approx: Literal[True, False]):
    pass


def erfc(fast_and_approx: Literal[True, False], dst: int):
    pass


def erfinv_init():
    pass


def erfinv(dst: int):
    pass


def gelu_init(fast_and_approx: Literal[True, False]):
    pass


def gelu(fast_and_approx: Literal[True, False], dst: int):
    pass


def heaviside_init():
    pass


def heaviside(param: int):
    pass


def is_inf_init():
    pass


def is_inf(dst: int):
    pass


def is_posinf_init():
    pass


def is_posinf(dst: int):
    pass


def is_neginf_init():
    pass


def is_neginf(dst: int):
    pass


def is_finite_init():
    pass


def is_finite(dst: int):
    pass


def is_na_n(dst: int):
    pass


def i0init():
    pass


def i0(dst: int):
    pass


def logical_not_unary_init():
    pass


def logical_not_unary(dst: int):
    pass


def recip_init():
    pass


def recip(dst: int):
    pass


def sign_init():
    pass


def sign(dst: int):
    pass


def sqrt_init():
    pass


def sqrt(dst: int):
    pass


def r_sqrt_init(fast_and_approx: Literal[True, False]):
    pass


def r_sqrt(fast_and_approx: Literal[True, False], dst: int):
    pass


def sigmoid_init():
    pass


def sigmoid(dst: int):
    pass


def log_init():
    pass


def log(dst: int):
    pass


def log_with_base_init():
    pass


def log_with_base(dst: int, log_base: int):
    pass


def power_init():
    pass


def power(dst: int, power_: int):
    pass


def r_sub_init():
    pass


def r_sub(dst: int, param: int):
    pass


def sign_bit_init():
    pass


def sign_bit(dst: int):
    pass


def square_init():
    pass


def square(dst: int):
    pass


def reduce(cb0: int, cb1: int, tile0: int, tile1: int, dst: int):
    pass


def transpose_wh_init(in_cb: int, out_cb: int):
    pass


def transpose_wh(cb: int, tile: int, dst: int):
    pass


def tanh_init():
    pass


def tanh(dst: int):
    pass


def tan_init():
    pass


def tan(dst: int):
    pass


def sin_init():
    pass


def sin(dst: int):
    pass


def cos_init():
    pass


def cos(dst: int):
    pass


def asin_init():
    pass


def asin(dst: int):
    pass


def atan_init():
    pass


def atan(dst: int):
    pass


def acos_init():
    pass


def acos(dst: int):
    pass


def ltz_init():
    pass


def ltz(dst: int):
    pass


def eqz_init():
    pass


def eqz(dst: int):
    pass


def lez_init():
    pass


def lez(dst: int):
    pass


def gtz_init():
    pass


def gtz(dst: int):
    pass


def gez_init():
    pass


def gez(dst: int):
    pass


def nez_init():
    pass


def nez(dst: int):
    pass


def unary_ne_init():
    pass


def unary_ne(dst: int, param: int):
    pass


def unary_gt_init():
    pass


def unary_gt(dst: int, param: int):
    pass


def unary_lt_init():
    pass


def unary_lt(dst: int, param: int):
    pass


def tilize_init(in_cb: int, block: int, out_cb: int):
    pass


def tilize_init_short(in_cb: int, block: int, out_cb: int):
    pass


def tilize_init_short_with_dt(old_in_cb: int, new_in_cb: int, block: int, out_cb: int):
    pass


def tilize_block(in_cb: int, block: int, out_cb: int):
    pass


def tilize_uninit(in_cb: int, out_cb: int):
    pass


def tilize_uninit_with_dt(old_in_cb: int, new_in_cb: int, out_cb: int):
    pass


def untilize_init(in_cb: int, out_cb: int):
    pass


def untilize_init_short(in_cb: int):
    pass


# TODO: can't enforce compile-time known int value for n
def untilize_block(n: int, in_cb: int, block: int, out_cb: int):
    pass


def untilize_uninit(in_cb: int):
    pass


def binary_op_init_common(cb0: int, cb1: int, cb_out: int = 16):
    pass


@enforce_literals
def pack_tile(
    out_of_order_output: Literal[True, False],
    from_dst: int,
    in_cb: int,
    out_tile: int = 0,
):
    pass
