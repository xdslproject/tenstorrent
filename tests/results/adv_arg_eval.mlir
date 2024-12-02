builtin.module {
  func.func @adv_arg_eval() {
    %0 = arith.constant 27 : i32
    %1 = arith.constant 3 : i32
    %2 = arith.addi %0, %1 : i32
    %3 = memref.alloc() : memref<1xi32>
    memref.store %2, %3[] : memref<1xi32>
    %4 = memref.alloc() : memref<1xi1>
    %5 = arith.constant 8 : i32
    %6 = arith.constant 20 : i32
    %7 = arith.constant 1 : i32
    %8 = memref.alloc() : memref<1xi32>
    scf.for %9 = %5 to %6 step %7 {
      memref.store %9, %8[] : memref<1xi32>
      %10 = memref.load %3[] : memref<1xi32>
      %11 = memref.load %8[] : memref<1xi32>
      %12 = "cb.pages_reservable_at_back"(%10, %11) : (i32, i32) -> i1
      memref.store %12, %4[] : memref<1xi1>
    }
  }
}
