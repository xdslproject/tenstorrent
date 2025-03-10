from tenstorrent.frontend import tt


@tt.compute
def void_compute():
    print("Hello, I am running a void compute kernel")
