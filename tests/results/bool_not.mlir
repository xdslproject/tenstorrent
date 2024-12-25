builtin.module attributes  {"kernel_type" = "data_in"} {
  func.func @kernel_main() {
    %0 = arith.constant 10 : i32
    %a = memref.alloc() : memref<i32>
    memref.store %0, %a[] : memref<i32>
    %1 = memref.load %a[] : memref<i32>
    %2 = arith.constant 4 : i32
    %3 = arith.cmpi ult, %1, %2 : i32
    %4 = arith.constant true
    %5 = arith.xori %3, %4 : i1
    scf.if %5 {
      %6 = arith.constant 2 : i32
      memref.store %6, %a[] : memref<i32>
    }
    func.return
  }
}
