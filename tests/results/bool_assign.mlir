builtin.module attributes  {"kernel_type" = "data_in"} {
  func.func @kernel_main() {
    %0 = arith.constant true
    %1 = memref.alloc() : memref<i1>
    memref.store %0, %1[] : memref<i1>
    %2 = arith.constant false
    %3 = memref.alloc() : memref<i1>
    memref.store %2, %3[] : memref<i1>
    %4 = memref.load %1[] : memref<i1>
    %5 = memref.load %3[] : memref<i1>
    %6 = arith.andi %4, %5 : i1
    %7 = memref.alloc() : memref<i1>
    memref.store %6, %7[] : memref<i1>
    %8 = memref.load %7[] : memref<i1>
    scf.if %8 {
      %9 = arith.constant false
      memref.store %9, %1[] : memref<i1>
    }
    %10 = arith.constant 5 : i32
    %11 = memref.alloc() : memref<i32>
    memref.store %10, %11[] : memref<i32>
    %12 = memref.load %11[] : memref<i32>
    %13 = arith.constant 10 : i32
    %14 = arith.cmpi ult, %12, %13 : i32
    %15 = memref.load %7[] : memref<i1>
    %16 = arith.ori %14, %15 : i1
    %17 = memref.alloc() : memref<i1>
    memref.store %16, %17[] : memref<i1>
    func.return
  }
}
