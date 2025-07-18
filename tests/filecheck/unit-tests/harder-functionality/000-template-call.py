# RUN: python3.13 %s | filecheck %s
from tenstorrent.frontend import tt


@tt.compute
def call_with_template():
    tt.exp(True, 3)


# CHECK: "comp.exp_tile"(%0) <{fast_and_approx = true}> : (i32) -> ()
