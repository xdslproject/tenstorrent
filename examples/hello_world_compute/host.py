from tenstorrent.frontend import tt


@tt.host
def hello_world():
    core = tt.Core(0, 0)
    device = tt.CreateDevice(0)
    cq = tt.GetCommandQueue(device)
    program = tt.CreateProgram()

    void_compute_kernel = tt.Kernel(
        program, void_compute, core, tt.MathFidelity.HiFi4, False, False, []
    )

    tt.SetRuntimeArgs(program, void_compute_kernel, core)
    tt.EnqueueProgram(cq, program, False)
    print(
        "Hello, Core {0, 0} on Device 0, I am sending you a compute kernel. Standby awaiting communication."
    )

    tt.Finish(cq)
    print("Thank you, Core {0, 0} on Device 0, for the completed task")
    tt.CloseDevice(device)
