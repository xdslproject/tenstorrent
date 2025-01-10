builtin.module attributes  {"kernel_type" = "data_in"} {
  func.func @kernel_main() {
    %a = memref.alloc() : memref<i32>
    %0 = arith.constant 3 : i32
    %1 = arith.constant 5 : i32
    %2 = arith.constant 1 : i32
    %i = memref.alloc() : memref<i32>
    scf.for %3 = %0 to %1 step %2 {
      memref.store %3, %i[] : memref<i32>
      %4 = memref.load %i[] : memref<i32>
      memref.store %4, %a[] : memref<i32>
    }
    func.return
  }
}
