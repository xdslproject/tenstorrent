builtin.module attributes  {"kernel_type" = "data_in"} {
  func.func @kernel_main() {
    %0 = arith.constant 7 : i32
    %1 = arith.extf %0 : i32 to f32
    %2 = memref.alloc() : memref<f32>
    memref.store %1, %2[] : memref<f32>
    %3 = arith.constant 1.340000e+01 : f32
    %4 = arith.constant 3 : i32
    %5 = arith.extf %4 : i32 to f32
    %6 = arith.mulf %3, %5 : f32
    memref.store %6, %2[] : memref<f32>
    func.return
  }
}
