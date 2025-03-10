from tenstorrent.frontend import tt

# Nothing to move. Print respond message.
# Make sure to export TT_METAL_DPRINT_CORES=0,0 before runtime.


@tt.data_in
def void_dataflow0():
    print("Hello, I am running a void data movement kernel on NOC 0.")


@tt.data_out
def void_dataflow1():
    print("Hello, I am running a void data movement kernel on NOC 1.")
