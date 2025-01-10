builtin.module attributes  {"kernel_type" = "data_in"} {
  func.func @kernel_main() {
    %0 = arith.constant 1 : i32
    %a = memref.alloc() : memref<i32>
    memref.store %0, %a[] : memref<i32>
    %1 = arith.constant 2 : i32
    %b = memref.alloc() : memref<i32>
    memref.store %1, %b[] : memref<i32>
    %2 = arith.constant 3 : i32
    %c = memref.alloc() : memref<i32>
    memref.store %2, %c[] : memref<i32>
    %3 = arith.constant 4 : i32
    %d = memref.alloc() : memref<i32>
    memref.store %3, %d[] : memref<i32>
    %4 = arith.constant 5 : i32
    %e = memref.alloc() : memref<i32>
    memref.store %4, %e[] : memref<i32>
    %5 = arith.constant false
    %f = memref.alloc() : memref<i1>
    memref.store %5, %f[] : memref<i1>
    %6 = arith.constant true
    %g = memref.alloc() : memref<i1>
    memref.store %6, %g[] : memref<i1>
    %7 = memref.load %a[] : memref<i32>
    %8 = memref.load %b[] : memref<i32>
    %9 = memref.load %c[] : memref<i32>
    %10 = memref.load %d[] : memref<i32>
    %11 = memref.load %e[] : memref<i32>
    %12 = memref.load %f[] : memref<i1>
    %13 = memref.load %g[] : memref<i1>
    "dm.noc_async_write_multicast"(%7, %8, %9, %10, %12, %13, %11) : (i32, i32, i32, i32, i1, i1, i32) -> ()
    %14 = memref.load %a[] : memref<i32>
    %15 = memref.load %b[] : memref<i32>
    %16 = memref.load %c[] : memref<i32>
    %17 = memref.load %d[] : memref<i32>
    %18 = memref.load %f[] : memref<i1>
    %19 = memref.load %g[] : memref<i1>
    "dm.noc_semaphore_set_multicast"(%14, %15, %16, %18, %19, %17) : (i32, i32, i32, i1, i1, i32) -> ()
    %20 = memref.load %a[] : memref<i32>
    %21 = memref.load %b[] : memref<i32>
    %22 = memref.load %c[] : memref<i32>
    %23 = memref.load %d[] : memref<i32>
    "dm.noc_async_write"(%20, %21, %22, %23) : (i32, i32, i32, i32) -> ()
    %24 = memref.load %a[] : memref<i32>
    %25 = memref.load %b[] : memref<i32>
    %26 = memref.load %c[] : memref<i32>
    %27 = memref.load %d[] : memref<i32>
    "dm.noc_async_read"(%24, %25, %26, %27) : (i32, i32, i32, i32) -> ()
    %28 = memref.load %a[] : memref<i32>
    %29 = memref.load %b[] : memref<i32>
    "dm.noc_semaphore_set"(%28, %29) : (i32, i32) -> ()
    %30 = memref.load %a[] : memref<i32>
    %31 = memref.load %b[] : memref<i32>
    "dm.noc_semaphore_wait"(%30, %31) : (i32, i32) -> ()
    %32 = memref.load %a[] : memref<i32>
    %33 = memref.load %b[] : memref<i32>
    %34 = memref.load %c[] : memref<i32>
    "dm.noc_semaphore_inc"(%32, %33, %34) : (i32, i32, i32) -> ()
    %35 = memref.load %a[] : memref<i32>
    "dm.noc_async_read_barrier"(%35) : (i32) -> ()
    %36 = memref.load %a[] : memref<i32>
    "dm.noc_async_write_barrier"(%36) : (i32) -> ()
    %37 = memref.load %a[] : memref<i32>
    %38 = memref.load %b[] : memref<i32>
    %39 = memref.load %c[] : memref<i32>
    %40 = "dm.get_noc_addr_from_bank_id"(%37, %38, %39) <{"dram" = true}> : (i32, i32, i32) -> ui64
    func.return
  }
}
