from tenstorrent.frontend import tt


@tt.host
def main():
    core = tt.Core(0, 0)
    device = tt.CreateDevice(0)

    cq = tt.GetCommandQueue(device)
    program = tt.CreateProgram()

    dataflow_0 = tt.Kernel(program, void_dataflow, core, tt.DataMovement_0, 0)
    dataflow_1 = tt.Kernel(program, void_dataflow, core, tt.DataMovement_1, 1)

    tt.SetRuntimeArgs(program, dataflow_0, core)
    tt.SetRuntimeArgs(program, dataflow_1, core)

    tt.EnqueueProgram(cq, program, False)
    print(
        "Hello, Core {0, 0} on Device 0, I am sending you some data. Standby awaiting communication."
    )

    tt.Finish(cq)
    print("Thank you, Core {0, 0} on Device 0, for the completed task.")

    tt.CloseDevice(device)
