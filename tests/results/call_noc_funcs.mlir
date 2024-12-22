builtin.module attributes  {"kernel_type" = "data_in"} {
  func.func @kernel_main() {
    %0 = arith.constant 1 : i32
    %1 = memref.alloc() : memref<i32>
    memref.store %0, %1[] : memref<i32>
    %2 = arith.constant 2 : i32
    %3 = memref.alloc() : memref<i32>
    memref.store %2, %3[] : memref<i32>
    %4 = arith.constant 3 : i32
    %5 = memref.alloc() : memref<i32>
    memref.store %4, %5[] : memref<i32>
    %6 = arith.constant 4 : i32
    %7 = memref.alloc() : memref<i32>
    memref.store %6, %7[] : memref<i32>
    %8 = arith.constant 5 : i32
    %9 = memref.alloc() : memref<i32>
    memref.store %8, %9[] : memref<i32>
    %10 = arith.constant false
    %11 = memref.alloc() : memref<i1>
    memref.store %10, %11[] : memref<i1>
    %12 = arith.constant true
    %13 = memref.alloc() : memref<i1>
    memref.store %12, %13[] : memref<i1>
    %14 = memref.load %1[] : memref<i32>
    %15 = memref.load %3[] : memref<i32>
    %16 = memref.load %5[] : memref<i32>
    %17 = memref.load %7[] : memref<i32>
    %18 = memref.load %9[] : memref<i32>
    %19 = memref.load %11[] : memref<i1>
    %20 = memref.load %13[] : memref<i1>
    "dm.noc_async_write_multicast"(%14, %15, %16, %17, %19, %20, %18) : (i32, i32, i32, i32, i1, i1, i32) -> ()
    %21 = memref.load %1[] : memref<i32>
    %22 = memref.load %3[] : memref<i32>
    %23 = memref.load %5[] : memref<i32>
    %24 = memref.load %7[] : memref<i32>
    %25 = memref.load %11[] : memref<i1>
    %26 = memref.load %13[] : memref<i1>
    "dm.noc_semaphore_set_multicast"(%21, %22, %23, %25, %26, %24) : (i32, i32, i32, i1, i1, i32) -> ()
    %27 = memref.load %1[] : memref<i32>
    %28 = memref.load %3[] : memref<i32>
    %29 = memref.load %5[] : memref<i32>
    %30 = memref.load %7[] : memref<i32>
    "dm.noc_async_write"(%27, %28, %29, %30) : (i32, i32, i32, i32) -> ()
    %31 = memref.load %1[] : memref<i32>
    %32 = memref.load %3[] : memref<i32>
    %33 = memref.load %5[] : memref<i32>
    %34 = memref.load %7[] : memref<i32>
    "dm.noc_async_read"(%31, %32, %33, %34) : (i32, i32, i32, i32) -> ()
    %35 = memref.load %1[] : memref<i32>
    %36 = memref.load %3[] : memref<i32>
    "dm.noc_semaphore_set"(%35, %36) : (i32, i32) -> ()
    %37 = memref.load %1[] : memref<i32>
    %38 = memref.load %3[] : memref<i32>
    "dm.noc_semaphore_wait"(%37, %38) : (i32, i32) -> ()
    %39 = memref.load %1[] : memref<i32>
    %40 = memref.load %3[] : memref<i32>
    %41 = memref.load %5[] : memref<i32>
    "dm.noc_semaphore_inc"(%39, %40, %41) : (i32, i32, i32) -> ()
    %42 = memref.load %1[] : memref<i32>
    "dm.noc_async_read_barrier"(%42) : (i32) -> ()
    %43 = memref.load %1[] : memref<i32>
    "dm.noc_async_write_barrier"(%43) : (i32) -> ()
    %44 = memref.load %1[] : memref<i32>
    %45 = memref.load %3[] : memref<i32>
    %46 = arith.constant 3 : i32
    %47 = "dm.get_noc_addr_from_bank_id"(%44, %45, %46) <{"dram" = true}> : (i32, i32, i32) -> ui64
    func.return
  }
}
