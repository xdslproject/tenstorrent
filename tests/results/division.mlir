builtin.module {
  func.func @division() {
    %0 = arith.constant 4 : i32
    %1 = arith.extf %0 : i32 to f32
    %2 = memref.alloc() : memref<1xf32>
    memref.store %1, %2[] : memref<1xf32>
    %3 = memref.load %2[] : memref<1xf32>
    %4 = arith.constant 2 : i32
    %5 = arith.extf %4 : i32 to f32
    %6 = arith.divf %3, %5 : f32
    memref.store %6, %2[] : memref<1xf32>
  }
}