builtin.module attributes  {"kernel_type" = "data_in"} {
  func.func @kernel_main() {
    %0 = arith.constant true
    %a = memref.alloc() : memref<i1>
    memref.store %0, %a[] : memref<i1>
    %1 = arith.constant false
    %b = memref.alloc() : memref<i1>
    memref.store %1, %b[] : memref<i1>
    %2 = memref.load %a[] : memref<i1>
    %3 = memref.load %b[] : memref<i1>
    %4 = arith.andi %2, %3 : i1
    %c = memref.alloc() : memref<i1>
    memref.store %4, %c[] : memref<i1>
    %5 = memref.load %c[] : memref<i1>
    scf.if %5 {
      %6 = arith.constant false
      memref.store %6, %a[] : memref<i1>
    }
    %7 = arith.constant 5 : i32
    %d = memref.alloc() : memref<i32>
    memref.store %7, %d[] : memref<i32>
    %8 = memref.load %d[] : memref<i32>
    %9 = arith.constant 10 : i32
    %10 = arith.cmpi ult, %8, %9 : i32
    %11 = memref.load %c[] : memref<i1>
    %12 = arith.ori %10, %11 : i1
    %e = memref.alloc() : memref<i1>
    memref.store %12, %e[] : memref<i1>
    func.return
  }
}
