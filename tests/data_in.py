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
    # TODO: think about how to handle this case
    # think have to handle just binary operation, constant on its own probably
    # isnt valid C/C++ code?
    a = 2 + 3  # Assign (BinOp (Const Const))


@data_in
def read_variable():
    a = 5       # Assign (Const)
    b = a + 1   # Assign (BinOp (Name Const))


@data_in
def overwriting_binop():
    a = 5
    a = a + 1
    # TODO: incorrect, should replace value of a0
    #     actually incorrect in the SSA gen, allocates new memory for a instead
    #     of reusing its location


single_assignment()
# multiple_assignment()
# simple_binop()
# read_variable()
# overwriting_binop()
