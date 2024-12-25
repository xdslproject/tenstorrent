builtin.module attributes  {"kernel_type" = "data_in"} {
  func.func @kernel_main() {
    %0 = arith.constant 7 : i32
    %a = memref.alloc() : memref<i32>
    memref.store %0, %a[] : memref<i32>
    %1 = arith.constant 3 : i32
    %b = memref.alloc() : memref<i32>
    memref.store %1, %b[] : memref<i32>
    %2 = arith.constant 9 : i32
    %c = memref.alloc() : memref<i32>
    memref.store %2, %c[] : memref<i32>
    %3 = memref.load %a[] : memref<i32>
    %4 = arith.constant 7 : i32
    %5 = arith.cmpi eq, %3, %4 : i32
    %6 = memref.load %b[] : memref<i32>
    %7 = arith.constant 3 : i32
    %8 = arith.cmpi eq, %6, %7 : i32
    %9 = arith.andi %5, %8 : i1
    %10 = memref.load %c[] : memref<i32>
    %11 = arith.constant 2 : i32
    %12 = arith.cmpi eq, %10, %11 : i32
    %13 = arith.ori %9, %12 : i1
    scf.if %13 {
      %14 = arith.constant 15 : i32
      memref.store %14, %a[] : memref<i32>
    }
    func.return
  }
}
