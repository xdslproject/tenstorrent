builtin.module attributes  {"kernel_type" = "data_in"} {
  func.func @kernel_main() {
    %0 = arith.constant 5 : i32
    %a = memref.alloc() : memref<i32>
    memref.store %0, %a[] : memref<i32>
    %b = memref.alloc() : memref<i32>
    %1 = memref.load %a[] : memref<i32>
    %2 = arith.constant 7 : i32
    %3 = arith.cmpi eq, %1, %2 : i32
    scf.if %3 {
      %4 = arith.constant 2 : i32
      memref.store %4, %a[] : memref<i32>
      %5 = arith.constant 3 : i32
      memref.store %5, %b[] : memref<i32>
    } else {
      %6 = memref.load %a[] : memref<i32>
      %7 = arith.constant 5 : i32
      %8 = arith.cmpi eq, %6, %7 : i32
      scf.if %8 {
        %9 = arith.constant 1 : i32
        memref.store %9, %a[] : memref<i32>
        %10 = arith.constant 2 : i32
        memref.store %10, %b[] : memref<i32>
      } else {
        %11 = arith.constant 10 : i32
        memref.store %11, %a[] : memref<i32>
        %12 = arith.constant 9 : i32
        memref.store %12, %b[] : memref<i32>
      }
    }
    func.return
  }
}
