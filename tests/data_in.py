from frontend.decorators import data_in


@data_in
def single_assignment():
    a = 0  # Assign (Const)


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
    for i in range(0, 5):
        a = 1
        b = 2
        c = 3
        d = 4
        a = 5


@data_in
def for_loop_use_i():
    for i in range(3, 5):
        a = i

# TODO: Constructs we should handle
#     for loops: updating global values
#     floating points
#     casting operations (/) -> Python implicit, C++ explicit
#     if statements
#     TT-related features
#     Note: in Python, loops and if-statement blocks don't create a new scope
#     so every variable declared in a loop is like declaring out of loop in C++


single_assignment()
multiple_assignment()
simple_binop()
read_variable()
overwriting_binop()
nested_binops()
for_loop()
