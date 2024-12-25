import tenstorrent as tt
from tenstorrent.frontend.dummy import *


@tt.data_in
def single_assignment():
    a = 13  # Assign (Const)


@tt.data_in
def multiple_assignment():
    a = 1  # Assign (Const)
    b = 4  # Assign (Const)


@tt.data_in
def simple_binop():
    a = 2 + 3  # Assign (BinOp (Const Const))


@tt.data_in
def read_variable():
    a = 5       # Assign (Const)
    b = a + 1   # Assign (BinOp (Name Const))


@tt.data_in
def overwriting_binop():
    a = 5
    a = a + 1


@tt.data_in
def nested_binops():
    a = 1 + 2 * 3 + 4


@tt.data_in
def for_loop():
    for i in range(3, 5):
        a = 10
        b = 20
        c = 30
        d = 40
        a = 50


@tt.data_in
def for_loop_use_i():
    for i in range(3, 5):
        a = i


@tt.data_in
def for_loop_overwriting():
    a = 0
    for i in range(3, 5):
        a = i


@tt.data_in
def nested_for():
    for i in range(3, 5):
        for j in range(7, 9):
            a = 10


@tt.data_in
def floating_point():
    a = 27.3 + 41.2
    b = 16.2 * 13.1


@tt.data_in
def if_statement():
    if True:
        a = 5


@tt.data_in
def evaluate_bool():
    a = 5
    if a == 6:
        a = 2


@tt.data_in
def evaluate_bool_nested():
    a = 5
    if a == 6 + 3:
        a = 2


@tt.data_in
def if_elif():
    a = 5
    if a == 7:
        a = 2
    elif a == 5:
        a = 1


@tt.data_in
def if_elif_else():
    a = 5
    if a == 7:
        a = 2
    elif a == 5:
        a = 1
    else:
        a = 10


@tt.data_in
def if_elif_else_blocks():
    a = 5
    if a == 7:
        a = 2
        b = 3
    elif a == 5:
        a = 1
        b = 2
    else:
        a = 10
        b = 9



@tt.data_in
def boolean_operators():
    a = 7
    b = 3
    c = 9
    if a == 7 and b == 3 or c == 2:
        a = 15


@tt.data_in
def greater_than():
    a = 7
    if a > 3:
        a = 1


@tt.data_in
def less_than():
    a = 8
    if a < 3:
        a = 1


@tt.data_in
def less_than_or_eq():
    a = 9
    if a <= 3:
        a = 3


@tt.data_in
def greater_than_or_eq():
    a = 9
    if a >= 3:
        a = 3


@tt.data_in
def bool_not():
    a = 10
    if not a < 4:
        a = 2


@tt.data_in
def sint():
    a = -5


@tt.data_in
def subtraction():
    a = 4
    a = a - 10


@tt.data_in
def float_subtraction():
    a = 3.7
    a = a - 10.2


@tt.data_in
def implicit_cast():
    a = 7
    a = 13.4 * 3


@tt.data_in
def division():
    a = 4
    a = a / 2


@tt.data_in
def bool_assign():
    a = True
    b = False
    c = a and b

    if c:
        a = False

    d = 5
    e = d < 10 or c


@tt.data_in
def func_call():
    cb_push_back(1, 1)


@tt.data_in
def adv_arg_eval():
    a = 27 + 3

    for i in range(8, 20):
        d = cb_pages_reservable_at_back(a, i)


@tt.data_in
def call_noc_funcs():
    a = 1
    b = 2
    c = 3
    d = 4
    e = 5
    f = False
    g = True

    noc_async_write_multicast(a, b, c, d, e, f, g)
    noc_semaphore_set_multicast(a, b, c, d, f, g)
    noc_async_write(a, b, c, d)
    noc_async_read(a, b, c, d)
    noc_semaphore_set(a, b)
    noc_semaphore_wait(a, b)
    noc_semaphore_inc(a, b, c)
    noc_async_read_barrier(a)
    noc_async_write_barrier(a)
    get_noc_addr_from_bank_id(True, a, b, c)


# TODO: should be tt.compute
@tt.compute
def call_compute_funcs():
    cb = 0
    in_tile_index = 1
    dst_tile_index = 2
    old_cb = 3
    new_cb = 4
    transpose = 5
    dst = 6
    cb0 = 7
    cb1 = 8
    acc_to_dest = False
    tile0 = 9
    tile1 = 10
    out_cols = 11
    out_rows = 12
    kt_dim = 13
    prev_cb1 = 14
    transpose_bool = True
    upper_limit = 15
    lower_limit = 16
    slope = 17
    param = 18
    log_base = 19
    power_ = 20
    tile = 21
    in_cb = 22
    block = 23
    out_cb = 24
    old_in_cb = 25
    new_in_cb = 26

    copy(cb, in_tile_index, dst_tile_index)
    copy_to_dst_init_short_with_dt(old_cb, new_cb, transpose)
    copy_to_dst_init_short(cb, transpose)
    copy_init()
    acquire_dst()
    release_dst()
    regs_acquire()
    regs_wait()
    regs_commit()
    regs_release()
    abs_init()
    abs(dst)
    add_init_nof()
    add_init(cb0, cb1, acc_to_dest)
    add(cb0, cb1, tile0, tile1, dst)
    sub_init_nof()
    sub_init(cb0, cb1, acc_to_dest)
    sub(cb0, cb1, tile0, tile1, dst)
    mul_init_f()
    mul_init(cb0, cb1)
    mul(cb0, cb1, tile0, tile1, dst)
    add_bcast_cols_init_short(cb0, cb1)
    add_bcast_rows_init_short(cb0, cb1)
    add_bcast(cb0, cb1, tile0, tile1, dst)
    sub_bcast_cols_init_short(cb0, cb1)
    sub_bcast(cb0, cb1, tile0, tile1, dst)
    mul_bcast_cols_init_short(cb0, cb1)
    mul_bcast_rows_init_short(cb0, cb1)
    mul_bcast(cb0, cb1, tile0, tile1, dst)
    mul_bcast_scalar_init_short(cb0, cb1)
    mul_bcast_scalar(cb0, cb1, tile0, tile1, dst)
    mm_init(cb0, cb1, dst, transpose)
    mm_init_short_with_dt(cb0, cb1, dst, transpose)
    mm_init_short(cb0, cb1, dst)
    matmul(cb0, cb1, tile0, tile1, dst, transpose)
    mm_block_init(cb0, cb1, dst, transpose, out_cols, out_rows, kt_dim)
    mm_block_init_short(cb0, cb1, transpose, out_cols, out_rows, kt_dim)
    mm_block_init_short_with_dt(cb0, cb1, prev_cb1, out_cols, out_rows, kt_dim)
    matmul_block(cb0, cb1, tile0, tile1, dst, transpose_bool, out_cols, out_rows, kt_dim)
    exp_init(False)
    exp(True, dst)
    exp2init()
    exp2(dst)
    exp_m1init()
    exp_m1(dst)
    relu_init()
    relu(dst)
    relu_max_init()
    relu_max(dst, upper_limit)
    relu_min_init()
    relu_min(dst, lower_limit)
    leaky_relu_init(dst, slope)
    elu_init()
    elu(dst, slope)
    erf_init(True)
    erf(False, dst)
    erfc_init(True)
    erfc(False, dst)
    erfinv_init()
    erfinv(dst)
    gelu_init(True)
    gelu(False, dst)
    heaviside_init()
    heaviside(param)
    is_inf_init()
    is_inf(dst)
    is_posinf_init()
    is_posinf(dst)
    is_neginf_init()
    is_neginf(dst)
    is_finite_init()
    is_finite(dst)
    is_na_n(dst)
    i0init()
    i0(dst)
    logical_not_unary_init()
    logical_not_unary(dst)
    recip_init()
    recip(dst)
    sign_init()
    sign(dst)
    sqrt_init()
    sqrt(dst)
    r_sqrt_init(True)
    r_sqrt(False, dst)
    sigmoid_init()
    sigmoid(dst)
    log_init()
    log(dst)
    log_with_base_init()
    log_with_base(dst, log_base)
    power_init()
    power(dst, power_)
    r_sub_init()
    r_sub(dst, param)
    sign_bit_init()
    sign_bit(dst)
    square_init()
    square(dst)
    reduce(cb0, cb1, tile0, tile1, dst)
    transpose_wh_init(in_cb, out_cb)
    transpose_wh(cb, tile, dst)
    tanh_init()
    tanh(dst)
    tan_init()
    tan(dst)
    sin_init()
    sin(dst)
    cos_init()
    cos(dst)
    asin_init()
    asin(dst)
    atan_init()
    atan(dst)
    acos_init()
    acos(dst)
    ltz_init()
    ltz(dst)
    eqz_init()
    eqz(dst)
    lez_init()
    lez(dst)
    gtz_init()
    gtz(dst)
    gez_init()
    gez(dst)
    nez_init()
    nez(dst)
    unary_ne_init()
    unary_ne(dst, param)
    unary_gt_init()
    unary_gt(dst, param)
    unary_lt_init()
    unary_lt(dst, param)
    tilize_init(in_cb, block, out_cb)
    tilize_init_short(in_cb, block, out_cb)
    tilize_init_short_with_dt(old_in_cb, new_in_cb, block, out_cb)
    tilize_block(in_cb, block, out_cb)
    tilize_uninit(in_cb, out_cb)
    tilize_uninit_with_dt(old_in_cb, new_in_cb, out_cb)
    untilize_init(in_cb, out_cb)
    untilize_init_short(in_cb)
    untilize_block(27, in_cb, block, out_cb)
    untilize_uninit(in_cb)


# Constructs currently implemented:
#   - assignment
#   - ints, floats, mixed
#   - nested binary operations (+, *, /)
#   - reading from variables
#   - nested for loops with range(a, b)
#   - for loops reading from loop variable
#   - nested if statements (and, or, not)
#   - type coercion

# TODO: Constructs not implemented
#     floating point comparisons
#     module operator
#     lists
#     TT-related features

single_assignment()
multiple_assignment()
simple_binop()
read_variable()
overwriting_binop()
nested_binops()
for_loop()
for_loop_use_i()
for_loop_overwriting()
nested_for()
floating_point()
if_statement()
evaluate_bool()
evaluate_bool_nested()
if_elif()
if_elif_else()
if_elif_else_blocks()
boolean_operators()
less_than()
less_than_or_eq()
greater_than()
greater_than_or_eq()
bool_not()
sint()
subtraction()
float_subtraction()
implicit_cast()
division()
bool_assign()
func_call()
adv_arg_eval()
call_noc_funcs()
call_compute_funcs()
