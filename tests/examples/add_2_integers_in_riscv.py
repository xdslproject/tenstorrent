import tenstorrent.frontend.tt

# Sends over two arrays of integers and each element is summed by the RV data movement core
# and then the result for each element is written into the destination memory which is read
# by the host

@tt.data_in
def single_assignment(src0_dram:uint, src1_dram:uint, dst_dram:uint, src0_dram_id:uint, src1_dram_id:uint, dst_dram_id:uint):
    src0_dram_noc_addr = get_noc_addr_from_bank_id(True, src0_dram_id, src0_dram)
    src1_dram_noc_addr = get_noc_addr_from_bank_id(True, src1_dram_id, src1_dram)
    dst_dram_noc_addr = get_noc_addr_from_bank_id(True, dst_dram_id, dst_dram)

    src0_data=[0]*100
    src1_data=[0]*100
    dst_data=[0]*100

    noc_async_read(src0_dram_noc_addr, src0_data, 100)
    noc_async_read(src1_dram_noc_addr, src1_data, 100)
    noc_async_read_barrier()

    for x in range(0, 100):
      dst_data[x]=src0_data[x]+src1_data[x]

    noc_async_write(dst_data, dst_dram_noc_addr, 100)
    noc_async_write_barrier()  

@tt.host
def host_code():
  core = tt.Core(0,0)
  single_tile_size = 2 * 1024

  device=tt.CreateDevice(0)
  command_queue=tt.GetCommandQueue(device)

  program=tt.CreateProgram()
  
  dram_configuration=tt.DRAMConfig(single_tile_size, single_tile_size)
  src0_dram_buffer=tt.CreateBuffer(dram_configuration)
  src1_dram_buffer=tt.CreateBuffer(dram_configuration)
  dst_dram_buffer=tt.CreateBuffer(dram_configuration)

  host_src0=[0]*100
  host_src1=[0]*100
  host_dst=[0]*100

  for i in range(0, 100):
    host_src0[i]=i
    host_src1[i]=100-i

  tt.EnqueueWriteBuffer(command_queue, src0_dram_buffer, host_src0, False);
  tt.EnqueueWriteBuffer(command_queue, src1_dram_buffer, host_src1, False);

  kernel = tt.Kernel(program, single_assignment, core, tt.DataMovement_0, 0)
  tt.SetRuntimeArgs(program, kernel, core, tt.GetMemoryAddress(src0_dram_buffer), tt.GetMemoryAddress(src1_dram_buffer), tt.GetMemoryAddress(dst_dram_buffer), 0, 0, 0)

  tt.EnqueueProgram(command_queue, program, False)
  tt.Finish(command_queue)

  tt.EnqueueReadBuffer(command_queue, dst_dram_buffer, host_dst, True);

  tt.CloseDevice(device)
  