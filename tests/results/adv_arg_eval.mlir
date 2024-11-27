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
    %7 = arith.index_cast %5 : i32 to index
    %8 = arith.index_cast %6 : i32 to index
    %9 = arith.constant 1 : index
    %10 = memref.alloc() : memref<1xindex>
    scf.for %11 = %7 to %8 step %9 {
      memref.store %11, %10[] : memref<1xindex>
      %12 = memref.load %3[] : memref<1xi32>
      %13 = memref.load %10[] : memref<1xindex>
      %14 = "cb.pages_reservable_at_back"(%12, %13) : (i32, index) -> i1
      memref.store %14, %4[] : memref<1xi1>
    }
  }
}
