from tenstorrent.frontend import tt


@tt.compute
def add_two_int_tiles():
    cb0 = 0
    cb1 = 1
    cb_out = 16

    tt.binary_op_init_common(cb0, cb1, cb_out)
    tt.add_init(cb0, cb1, False)

    # wait for a single block of tiles in each input CB
    tt.cb_wait_front(cb0, 1)
    tt.cb_wait_front(cb1, 1)

    # acquire 8 tile registers
    tt.regs_acquire()

    # add the first tiles in cb0 and cb1, storing the result tile
    tt.add(cb0, cb1, 0, 0, 0)

    # commit the result, signals the packer
    tt.regs_commit()

    tt.regs_wait()  # the packer core will wait here
    tt.pack_tile(False, 0, cb_out, 0)
    tt.regs_release()  # and release here

    tt.cb_pop_front(cb0, 1)
    tt.cb_pop_front(cb1, 1)
    tt.cb_push_back(cb_out, 1)
