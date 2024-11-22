from frontend.decorators import data_in


@data_in
def single_assignment():
    a = 13  # Assign (Const)


@data_in
def multiple_assignment():
    a = 1  # Assign (Const)
    b = 4  # Assign (Const)


@data_in
def simple_binop():
    a = 2 + 3  # Assign (BinOp (Const Const))


@data_in
def read_variable():
    a = 5       # Assign (Const)
    b = a + 1   # Assign (BinOp (Name Const))


@data_in
def overwriting_binop():
    a = 5
    a = a + 1


@data_in
def nested_binops():
    a = 1 + 2 * 3 + 4


@data_in
def for_loop():
    for i in range(3, 5):
        a = 10
        b = 20
        c = 30
        d = 40
        a = 50


@data_in
def for_loop_use_i():
    for i in range(3, 5):
        a = i


@data_in
def nested_for():
    for i in range(3, 5):
        for j in range(7, 9):
            a = 10


@data_in
def floating_point():
    a = 27.3 + 41.2
    b = 16.2 * 13.1


@data_in
def implicit_cast():
    a = 13.4 * 3


@data_in
def if_statement():
    if True:
        a = 5


@data_in
def evaluate_bool():
    a = 5
    if a == 6:
        a = 2


@data_in
def if_elif():
    a = 5
    if a == 7:
        a = 2
    elif a == 5:
        a = 1


@data_in
def if_elif_else():
    a = 5
    if a == 7:
        a = 2
    elif a == 5:
        a = 1
    else:
        a = 10


@data_in
def boolean_operators():
    a = 7
    b = 3
    c = 9
    if a == 7 and b == 3 or c == 2:
        a = 1
        b = 1
        c = 1


# TODO: Constructs we should handle
#     floating points
#     if statements
#     TT-related features
#     Nested for loops currently aren't correct (they create redundant vars)
#     Note: in Python, loops and if-statement blocks don't create a new scope
#     so every variable declared in a loop is like declaring out of loop in C++

single_assignment()
multiple_assignment()
simple_binop()
read_variable()
overwriting_binop()
nested_binops()
for_loop()
for_loop_use_i()
# floating_point()
# implicit_cast()
# if_statement()
# evaluate_bool()
# if_elif()
# if_elif_else()
# boolean_operators()
# nested_for()
