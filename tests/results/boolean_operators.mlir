builtin.module attributes  {"kernel_type" = "data_in"} {
  func.func @kernel_main() {
    %0 = arith.constant 7 : i32
    %1 = memref.alloc() : memref<i32>
    memref.store %0, %1[] : memref<i32>
    %2 = arith.constant 3 : i32
    %3 = memref.alloc() : memref<i32>
    memref.store %2, %3[] : memref<i32>
    %4 = arith.constant 9 : i32
    %5 = memref.alloc() : memref<i32>
    memref.store %4, %5[] : memref<i32>
    %6 = memref.load %1[] : memref<i32>
    %7 = arith.constant 7 : i32
    %8 = arith.cmpi eq, %6, %7 : i32
    %9 = memref.load %3[] : memref<i32>
    %10 = arith.constant 3 : i32
    %11 = arith.cmpi eq, %9, %10 : i32
    %12 = arith.andi %8, %11 : i1
    %13 = memref.load %5[] : memref<i32>
    %14 = arith.constant 2 : i32
    %15 = arith.cmpi eq, %13, %14 : i32
    %16 = arith.ori %12, %15 : i1
    scf.if %16 {
      %17 = arith.constant 15 : i32
      memref.store %17, %1[] : memref<i32>
    }
    func.return
  }
}
