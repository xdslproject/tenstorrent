// RUN: python3.13 tenstorrent/tools/tt-opt %s -t tt-metalium | filecheck %s

builtin.module {
  builtin.module attributes {kernel_type = "data_in"} {
    func.func @kernel_main() {
      %0 = arith.constant 27 : i32
      %1 = arith.constant 9.300000e+00 : f32
      %2 = arith.sitofp %0 : i32 to f32
      %3 = arith.addf %2, %1 : f32
      %a = memref.alloc() : memref<f32>
      memref.store %3, %a[] : memref<f32>
      %4 = arith.constant 45 : i32
      %b = memref.alloc() : memref<i32>
      memref.store %4, %b[] : memref<i32>
      %c = memref.alloc() : memref<i32>
      %d = memref.alloc() : memref<i1>
      %5 = arith.constant 5 : i32
      %6 = arith.constant 23 : i32
      %7 = arith.constant 1 : i32
      %i = memref.alloc() : memref<i32>
      scf.for %8 = %5 to %6 step %7  : i32 {
        memref.store %8, %i[] : memref<i32>
        %9 = memref.load %b[] : memref<i32>
        %10 = arith.constant 12 : i32
        %11 = arith.cmpi ult, %9, %10 : i32
        scf.if %11 {
          %12 = arith.constant 9 : i32
          %13 = arith.constant 17 : i32
          %14 = arith.constant 1 : i32
          %j = memref.alloc() : memref<i32>
          scf.for %15 = %12 to %13 step %14  : i32 {
            memref.store %15, %j[] : memref<i32>
            %16 = memref.load %j[] : memref<i32>
            %17 = arith.constant 1 : i32
            %18 = arith.addi %16, %17 : i32
            memref.store %18, %c[] : memref<i32>
            %19 = arith.constant 4 : i32
            %20 = arith.constant 20 : i32
            %21 = arith.constant 1 : i32
            %k = memref.alloc() : memref<i32>
            scf.for %22 = %19 to %20 step %21  : i32 {
              memref.store %22, %k[] : memref<i32>
              %23 = memref.load %b[] : memref<i32>
              %24 = memref.load %k[] : memref<i32>
              %25 = "cb.pages_reservable_at_back"(%23, %24) : (i32, i32) -> i1
              memref.store %25, %d[] : memref<i1>
            }
          }
        } else {
          %26 = memref.load %b[] : memref<i32>
          %27 = arith.constant 35 : i32
          %28 = arith.cmpi ugt, %26, %27 : i32
          scf.if %28 {
            printf.print_format "Hello world"
          } else {
            %29 = memref.load %b[] : memref<i32>
            %30 = memref.load %b[] : memref<i32>
            "cb.wait_front"(%29, %30) : (i32, i32) -> ()
          }
        }
      }
      func.return
    }
  }
}

// CHECK:      void kernel_main() {
// CHECK-NEXT:     float a = static_cast<float>(27) + 9.300000190734863;
// CHECK-NEXT:     std::int32_t b = 45;
// CHECK-NEXT:     std::int32_t c;
// CHECK-NEXT:     bool d;
// CHECK-NEXT:     std::int32_t i;
// CHECK-NEXT:     for (i = 5; i < 23; i += 1) {
// CHECK-NEXT:         if (b < 12) {
// CHECK-NEXT:             std::int32_t j;
// CHECK-NEXT:             for (j = 9; j < 17; j += 1) {
// CHECK-NEXT:                 c = j + 1;
// CHECK-NEXT:                 std::int32_t k;
// CHECK-NEXT:                 for (k = 4; k < 20; k += 1) {
// CHECK-NEXT:                     d = cb_pages_reservable_at_back(b, k);
// CHECK-NEXT:                 }
// CHECK-NEXT:             }
// CHECK-NEXT:         } else {
// CHECK-NEXT:             if (b > 35) {
// CHECK-NEXT:                 DPRINT_DATA0(DPRINT << "Hello world" << ENDL());
// CHECK-NEXT:             } else {
// CHECK-NEXT:                 cb_wait_front(b, b);
// CHECK-NEXT:             }
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT: }
