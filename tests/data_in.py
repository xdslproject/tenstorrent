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
    # semantics: each iteration create a new memref, and assign a constant to it
    for i in range(0, 5):
        a = 1

# TODO: Constructs we should handle
#     for loops
#     floating points
#     casting operations (/) -> Python implicit, C++ explicit
#     if statements
#     TT-related features


single_assignment()
multiple_assignment()
simple_binop()
read_variable()
overwriting_binop()
nested_binops()
for_loop()
