builtin.module attributes  {"kernel_type" = "data_in"} {
  func.func @kernel_main() {
    %0 = arith.constant 5 : i32
    %a = memref.alloc() : memref<i32>
    memref.store %0, %a[] : memref<i32>
    %1 = memref.load %a[] : memref<i32>
    %2 = arith.constant 7 : i32
    %3 = arith.cmpi eq, %1, %2 : i32
    scf.if %3 {
      %4 = arith.constant 2 : i32
      memref.store %4, %a[] : memref<i32>
    } else {
      %5 = memref.load %a[] : memref<i32>
      %6 = arith.constant 5 : i32
      %7 = arith.cmpi eq, %5, %6 : i32
      scf.if %7 {
        %8 = arith.constant 1 : i32
        memref.store %8, %a[] : memref<i32>
      } else {
        %9 = arith.constant 10 : i32
        memref.store %9, %a[] : memref<i32>
      }
    }
    func.return
  }
}
