builtin.module attributes  {"kernel_type" = "data_in"} {
  func.func @kernel_main() {
    %0 = arith.constant 0 : i32
    %a = memref.alloc() : memref<i32>
    memref.store %0, %a[] : memref<i32>
    %1 = arith.constant 3 : i32
    %2 = arith.constant 5 : i32
    %3 = arith.constant 1 : i32
    %i = memref.alloc() : memref<i32>
    scf.for %4 = %1 to %2 step %3 {
      memref.store %4, %i[] : memref<i32>
      %5 = memref.load %i[] : memref<i32>
      memref.store %5, %a[] : memref<i32>
    }
    func.return
  }
}
