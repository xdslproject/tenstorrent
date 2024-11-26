import tenstorrent as tt


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
def implicit_cast():
    a = 13.4 * 3


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
def sint_comparisons_cast():
    a = -5
    if a < 10:
        a = 10

    if a > 10:
        a = 10

    if a >= 20:
        a = 20

    if a <= 20:
        a = 20


@tt.data_in
def division():
    a = 4
    a = 10 / 2  # a: must be implicitly casted to float, so must 10, 2


@tt.data_in
def subtraction():
    a = 4
    a = 5 - 10  # a: must be implicitly casted to int32 from uint32


# Constructs currently implemented:
#   - assignment
#   - ints, floats, not mixing
#   - nested binary operations (+, *)
#   - reading from variables
#   - nested for loops with range(a, b)
#   - for loops reading from loop variable
#   - nested if statements

# TODO: Constructs we should handle
#     if statements (>=, <, etc. for floats, 'not')
#     binary operations (division, subtraction) [may require implicit casting?]
#     lists?
#     TT-related features
#     nice to write outputs to files data_in.out (C++) data_in.log (all) to
#     let git / other regression testing system

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
# sint()
# sint_comparisons_cast()
# implicit_cast()
# division()
# subtraction()
