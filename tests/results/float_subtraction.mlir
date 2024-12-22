builtin.module attributes  {"kernel_type" = "data_in"} {
  func.func @kernel_main() {
    %0 = arith.constant 3.700000e+00 : f32
    %1 = memref.alloc() : memref<f32>
    memref.store %0, %1[] : memref<f32>
    %2 = memref.load %1[] : memref<f32>
    %3 = arith.constant 1.020000e+01 : f32
    %4 = arith.subf %2, %3 : f32
    memref.store %4, %1[] : memref<f32>
    func.return
  }
}
