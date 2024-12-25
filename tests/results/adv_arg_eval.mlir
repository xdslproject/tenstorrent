builtin.module attributes  {"kernel_type" = "data_in"} {
  func.func @kernel_main() {
    %0 = arith.constant 27 : i32
    %1 = arith.constant 3 : i32
    %2 = arith.addi %0, %1 : i32
    %a = memref.alloc() : memref<i32>
    memref.store %2, %a[] : memref<i32>
    %d = memref.alloc() : memref<i1>
    %3 = arith.constant 8 : i32
    %4 = arith.constant 20 : i32
    %5 = arith.constant 1 : i32
    %i = memref.alloc() : memref<i32>
    scf.for %6 = %3 to %4 step %5 {
      memref.store %6, %i[] : memref<i32>
      %7 = memref.load %a[] : memref<i32>
      %8 = memref.load %i[] : memref<i32>
      %9 = "cb.pages_reservable_at_back"(%7, %8) : (i32, i32) -> i1
      memref.store %9, %d[] : memref<i1>
    }
    func.return
  }
}
