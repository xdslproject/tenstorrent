builtin.module attributes  {"kernel_type" = "data_in"} {
  func.func @kernel_main() {
    %0 = arith.constant 3.700000e+00 : f32
    %a = memref.alloc() : memref<f32>
    memref.store %0, %a[] : memref<f32>
    %1 = memref.load %a[] : memref<f32>
    %2 = arith.constant 1.020000e+01 : f32
    %3 = arith.subf %1, %2 : f32
    memref.store %3, %a[] : memref<f32>
    func.return
  }
}
